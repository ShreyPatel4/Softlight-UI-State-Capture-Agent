import json
import re
from dataclasses import dataclass
from typing import Any, Literal, Optional, Sequence

from sqlalchemy.orm import Session

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from ..config import settings
from ..models import Flow, log_flow_event
from .dom_scanner import CandidateAction
from .task_spec import TaskSpec


def _extract_json(text: str) -> dict | None:
    """
    Extract the last JSON object from noisy LLM output.

    The function strips code fences and ignores any chatter surrounding the JSON
    payload. When multiple JSON-looking regions exist, the last valid object is
    returned. None is returned when parsing fails.
    """

    try:
        def _strip_code_fences(payload: str) -> str:
            fenced = re.findall(r"```(?:json)?\s*(.*?)```", payload, re.DOTALL | re.IGNORECASE)
            if fenced:
                return fenced[-1]
            return payload

        cleaned = _strip_code_fences(text.strip())

        spans: list[str] = []
        depth = 0
        start_idx: int | None = None
        for idx, ch in enumerate(cleaned):
            if ch == "{":
                if depth == 0:
                    start_idx = idx
                depth += 1
            elif ch == "}":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start_idx is not None:
                        spans.append(cleaned[start_idx : idx + 1])

        for candidate in reversed(spans):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
    except Exception:
        return None

    return None


@dataclass
class PolicyDecision:
    action_id: Optional[str]
    action_type: Literal["click", "type"]
    text_to_type: Optional[str]
    done: bool
    capture_before: bool
    capture_after: bool
    label: Optional[str]
    reason: Optional[str]
    should_capture: bool = True


POLICY_SYSTEM_PROMPT = """
Return only a single JSON object that follows this schema:
{
  "action_id": "<one of the provided candidate ids>",
  "action_type": "click" | "type",
  "text_to_type": "<text to type>" | null,
  "done": true | false,
  "capture_before": true | false,
  "capture_after": true | false,
  "label": "<short snake_case label>",
  "reason": "<brief reason>"
}

Rules:
- You will receive generic candidate actions. Some have "action_type": "click" and others have "action_type": "type".
- Choose exactly one of the provided candidates and never invent ids.
- For click actions, ignore text_to_type or set it to null.
- For type actions, set text_to_type to the exact text that should be typed based on the user goal.
- Map goal parts to appropriate fields: names/titles/subjects go to inputs mentioning title, name, or subject; descriptions/details/comments go to inputs mentioning description, details, notes, or comment.
- For creation or confirmation steps, keep done=false until the action would complete the goal.
- If no action is needed or progress cannot be made, set action_id to null and text_to_type to null.
- Always emit JSON only; no prose, no markdown, no code fences.

Example outputs for a generic goal "create item named 'Softlight agent test issue'":
{"action_id":"btn_0","action_type":"click","text_to_type":null,"done":false,"capture_before":true,"capture_after":true,"label":"open_form","reason":"open the creation form"}
{"action_id":"input_1","action_type":"type","text_to_type":"Softlight agent test issue","done":false,"capture_before":false,"capture_after":true,"label":"title_entered","reason":"type the requested title"}
"""


def build_policy_prompt(
    task: TaskSpec,
    app_name: str,
    url: str,
    history_summary: str,
    candidates: Sequence[CandidateAction],
) -> str:
    """
    Build the text prompt for Qwen.
    """
    lines: list[str] = []
    lines.append(POLICY_SYSTEM_PROMPT.strip())
    lines.append("")
    lines.append("User goal:")
    lines.append(task.goal)
    lines.append("")
    lines.append(f"App name: {app_name}")
    lines.append(f"Current URL: {url}")
    lines.append("")
    lines.append("History summary:")
    lines.append(history_summary if history_summary else "(no previous actions)")
    lines.append("")
    lines.append("Candidate actions:")
    for cand in candidates:
        lines.append(
            f"  - id={cand.id}  type={cand.action_type}  description={cand.description}"
        )
    lines.append("")
    lines.append(
        "Return a single JSON object that follows the schema exactly. "
        "Do not include any text before or after the JSON."
    )
    return "\n".join(lines)


def choose_fallback_action(
    goal: str, candidates: Sequence[CandidateAction]
) -> CandidateAction:
    def score(cand: CandidateAction) -> int:
        g = set(goal.lower().split())
        d = set(cand.description.lower().split())
        return len(g & d)

    return max(candidates, key=score)


def create_policy_hf_pipeline(model_name: str | None = None) -> Any:
    model_name = model_name or settings.hf_model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device="cpu",
        max_new_tokens=128,
        do_sample=False,
        temperature=0.0,
        top_p=0.9,
    )


class PolicyLLMClient:
    def __init__(self, hf_pipeline: Any) -> None:
        self.hf_pipeline = hf_pipeline

    def generate(self, prompt: str) -> str:
        return self.hf_pipeline(prompt, num_return_sequences=1)[0]["generated_text"]

    def complete(self, prompt: str) -> str:
        return self.generate(prompt)


def choose_action_with_llm(
    llm: PolicyLLMClient,
    task: TaskSpec,
    app_name: str,
    url: str,
    history_summary: str,
    candidates: Sequence[CandidateAction],
    session: Session | None = None,
    flow: Flow | None = None,
) -> PolicyDecision:
    prompt = build_policy_prompt(task, app_name, url, history_summary, candidates)
    raw = llm.complete(prompt).strip()

    def _warn(message: str) -> None:
        if session and flow:
            snippet = raw[:200].replace("\n", " ")
            log_flow_event(session, flow, "warning", f"{message}: {snippet}")

    data = _extract_json(raw)
    if data is None:
        _warn("LLM output missing valid JSON")
        return PolicyDecision(
            action_id=None,
            action_type="click",
            text_to_type=None,
            done=True,
            capture_before=False,
            capture_after=True,
            label="fallback_capture",
            reason="Fallback decision because model output was not valid JSON",
            should_capture=True,
        )

    action_id = data.get("action_id")
    cand_map = {c.id: c for c in candidates}
    if action_id not in cand_map:
        _warn("LLM output missing or invalid action_id")
        return PolicyDecision(
            action_id=None,
            action_type="click",
            text_to_type=None,
            done=True,
            capture_before=False,
            capture_after=True,
            label="fallback_capture",
            reason="Fallback decision because model output did not match a candidate id",
            should_capture=True,
        )

    cand = cand_map[action_id]
    action_type = data.get("action_type") or cand.action_type

    return PolicyDecision(
        action_id=action_id,
        action_type=action_type,
        text_to_type=data.get("text_to_type") if isinstance(data.get("text_to_type"), str) else None,
        done=bool(data.get("done", False)),
        capture_before=bool(data.get("capture_before", True)),
        capture_after=bool(data.get("capture_after", True)),
        label=data.get("label") or f"after_action_{action_id}",
        reason=data.get("reason"),
        should_capture=True,
    )


class Policy:
    """
    LLM backed policy that chooses the next UI action using a Hugging Face model.
    """

    def __init__(self, model_name: str | None = None) -> None:
        model_name = model_name or settings.hf_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.generator = create_policy_hf_pipeline(model_name)

    def _run_hf(self, prompt: str) -> str:
        out = self.generator(
            prompt,
            num_return_sequences=1,
        )[0]["generated_text"]
        return out

    def _extract_json(self, raw: str) -> dict:
        data = _extract_json(raw.strip())
        if data is None:
            return {}
        return data

    async def choose_action(
        self,
        task: TaskSpec,
        candidates: Sequence[CandidateAction],
        history_summary: str,
        url: str,
    ) -> PolicyDecision:
        """
        Decide the next action. Called from the agent loop.
        """
        prompt = build_policy_prompt(
            task=task,
            app_name=task.app_name,
            url=url,
            history_summary=history_summary,
            candidates=candidates,
        )
        raw = self._run_hf(prompt)
        data = self._extract_json(raw)
        if not data:
            fallback = choose_fallback_action(task.goal, candidates)
            return PolicyDecision(
                action_id=None,
                action_type=fallback.action_type,
                text_to_type=None,
                done=True,
                capture_before=False,
                capture_after=True,
                label="fallback_capture",
                reason="Fallback decision because model output was not valid JSON",
                should_capture=True,
            )

        # Validate action_id
        valid_ids = {c.id for c in candidates}
        if data.get("action_id") not in valid_ids:
            first = candidates[0]
            data["action_id"] = first.id
            data["action_type"] = first.action_type

        # Ensure required keys exist with sane defaults
        data.setdefault("text_to_type", data.get("input_text") or data.get("text"))
        data.setdefault("capture_before", True)
        data.setdefault("capture_after", True)
        data.setdefault("label", f"after_action_{data['action_id']}")
        data.setdefault("done", False)
        data.setdefault("reason", "Model did not provide a reason")

        decision = PolicyDecision(
            action_id=data.get("action_id"),
            action_type=data.get("action_type", "click"),
            text_to_type=data.get("text_to_type") if isinstance(data.get("text_to_type"), str) else None,
            done=bool(data.get("done")),
            capture_before=bool(data.get("capture_before")),
            capture_after=bool(data.get("capture_after")),
            label=data.get("label"),
            reason=data.get("reason"),
            should_capture=True,
        )

        print(
            "[policy] decision:",
            json.dumps(
                {
                    "app": task.app_name,
                    "goal": task.goal,
                    "action_id": decision.action_id,
                    "action_type": decision.action_type,
                    "capture_before": decision.capture_before,
                    "capture_after": decision.capture_after,
                    "label": decision.label,
                    "done": decision.done,
                },
                ensure_ascii=False,
            ),
        )

        return decision
