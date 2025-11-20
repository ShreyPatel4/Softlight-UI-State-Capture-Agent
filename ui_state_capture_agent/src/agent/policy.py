import json
from typing import Dict, List

import openai

from ..config import settings
from .dom_scanner import CandidateAction
from .task_spec import TaskSpec


class Policy:
    def __init__(self) -> None:
        openai.api_key = settings.openai_api_key
        self.client = openai.AsyncOpenAI(api_key=openai.api_key)

    @staticmethod
    def _build_user_message(
        task: TaskSpec, candidates: List[CandidateAction], history_summary: str
    ) -> str:
        lines: List[str] = [
            "Task:",
            f"- App: {task.app_name}",
            f"- Goal: {task.goal}",
            f"- Object type: {task.object_type}",
            "",
            "Recent history summary:",
            history_summary or "(none)",
            "",
            "Candidate actions:",
        ]

        for idx, candidate in enumerate(candidates, start=1):
            lines.append(f"{idx}. {candidate.id}: {candidate.description}")

        lines.append(
            "\nRespond with JSON in the following format:\n"
            "{\n"
            '  "chosen_action_id": "act_3",\n'
            '  "action_type": "click",\n'
            '  "input_text": null,\n'
            '  "capture_before": true,\n'
            '  "capture_after": true,\n'
            '  "state_label_after": "create_modal_open",\n'
            '  "done": false,\n'
            '  "reason": "We need to open the create project modal."\n'
            "}"
        )

        return "\n".join(lines)

    @staticmethod
    def _fallback_action(candidates: List[CandidateAction]) -> Dict:
        if not candidates:
            return {
                "chosen_action_id": None,
                "action_type": None,
                "input_text": None,
                "capture_before": False,
                "capture_after": False,
                "state_label_after": None,
                "done": False,
                "reason": "No candidate actions available.",
            }

        first_candidate = candidates[0]
        return {
            "chosen_action_id": first_candidate.id,
            "action_type": first_candidate.action_type,
            "input_text": None,
            "capture_before": True,
            "capture_after": True,
            "state_label_after": None,
            "done": False,
            "reason": "Falling back to first candidate due to invalid or unparsable model response.",
        }

    async def choose_action(
        self,
        task: TaskSpec,
        candidates: List[CandidateAction],
        history_summary: str,
    ) -> Dict:
        system_prompt = (
            "You are a UI agent deciding which UI element to interact with next. "
            "You must respond only with JSON."
        )
        user_message = self._build_user_message(task, candidates, history_summary)

        try:
            completion = await self.client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
            )
            content = completion.choices[0].message.content if completion.choices else None
        except Exception:  # noqa: BLE001
            return self._fallback_action(candidates)

        if not content:
            return self._fallback_action(candidates)

        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            return self._fallback_action(candidates)

        candidate_ids = {candidate.id for candidate in candidates}
        if result.get("chosen_action_id") not in candidate_ids:
            return self._fallback_action(candidates)

        result.setdefault("done", False)
        return result
