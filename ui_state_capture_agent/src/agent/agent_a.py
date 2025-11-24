from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import List, Optional, Sequence

from pydantic import BaseModel

from ..models import Flow, SessionLocal, Step
from .llm_client import StructuredLLMClient, create_structured_llm_client
from .orchestrator import FlowSummary as OrchestratorFlowSummary
from .orchestrator import run_task_query_async


class PlanStep(BaseModel):
    index: int
    instruction: str
    ui_hint: Optional[str] = None
    stop_condition: Optional[str] = None


class WorkflowPlan(BaseModel):
    app_name: str
    original_question: str
    goal_summary: str
    success_criteria: str
    steps: List[PlanStep]


class StepStateSummary(BaseModel):
    plan_index: int
    description: str
    screenshot_url: str
    state_kind: str
    url: str


class ExecutionTrace(BaseModel):
    flow_id: str
    app_name: str
    plan: WorkflowPlan
    step_states: List[StepStateSummary]
    status: str
    status_reason: str | None


PLANNER_SYSTEM_PROMPT = """You are Agent A in a two agent system.

Your job is to take the user's question about how to perform a task in a web app,
and produce a clear multi step workflow that a separate UI executor agent (Agent B)
can follow.

Agent B can:
* Open a given start URL for an app.
* Click visible buttons and links.
* Type into text fields, textareas, and contenteditable regions.
* Capture screenshots and DOM snapshots at each step.

Agent B does not know the user's question directly.
It only sees: the app name, the start URL, and a short text goal for the current step.

Your output must be a single JSON object with this shape:
{
  "app_name": "<short app name like 'linear' or 'notion'>",
  "goal_summary": "<one sentence summary>",
  "success_criteria": "<what final state indicates success>",
  "steps": [
    {
      "index": 1,
      "instruction": "<what should Agent B try to do now>",
      "ui_hint": "<how to recognize this in the UI>" or null,
      "stop_condition": "<what tells us this step is complete>" or null
    }
  ]
}
Return only the JSON, no commentary."""


EXPLAINER_SYSTEM_PROMPT = """You are Agent A summarizing what Agent B did in the live app.

Explain to the user how to perform the task themselves, step by step, referencing
what actually happened during execution. Mention screenshots by URL in the form
"(see screenshot N)" where N is the 1-based step index when helpful.
Write a clear set of numbered instructions for the user.
"""


@dataclass
class _PlanPromptContext:
    question: str
    app_hint: str


def _normalize_app_name(question: str) -> str:
    lowered = question.lower()
    for candidate in ["linear", "notion", "asana", "slack", "github", "google"]:
        if candidate in lowered:
            return candidate
    return "unknown"


def build_planner_prompt(question: str, *, app_hint: Optional[str] = None) -> str:
    context = _PlanPromptContext(question=question, app_hint=app_hint or _normalize_app_name(question))
    lines = [PLANNER_SYSTEM_PROMPT.strip(), "", f"Question: {context.question}"]
    if context.app_hint:
        lines.append(f"App hint: {context.app_hint}")
    lines.append("")
    lines.append("Produce the JSON plan now.")
    return "\n".join(lines)


def build_explainer_prompt(trace: ExecutionTrace) -> str:
    plan_json = trace.plan.model_dump()
    steps_payload: list[dict[str, str]] = []
    for idx, step in enumerate(trace.step_states, start=1):
        steps_payload.append(
            {
                "index": idx,
                "description": step.description,
                "screenshot_url": step.screenshot_url,
                "state_kind": step.state_kind,
                "url": step.url,
            }
        )

    payload = {
        "original_question": trace.plan.original_question,
        "plan": plan_json,
        "execution": {
            "status": trace.status,
            "status_reason": trace.status_reason,
            "steps": steps_payload,
        },
    }

    lines = [EXPLAINER_SYSTEM_PROMPT.strip(), "", json.dumps(payload, indent=2)]
    return "\n".join(lines)


def _build_screenshot_url(flow_id: str, step_index: int) -> str:
    return f"/assets/{flow_id}/{step_index}/screenshot"


def _summarize_step(step: Step) -> str:
    if step.description:
        return step.description
    return step.state_label


def _collect_step_states(flow_id: str, plan: WorkflowPlan) -> list[StepStateSummary]:
    session = SessionLocal()
    try:
        steps: Sequence[Step] = (
            session.query(Step).filter(Step.flow_id == flow_id).order_by(Step.step_index.asc()).all()
        )
        summaries: list[StepStateSummary] = []
        for step in steps:
            plan_index = min(step.step_index, len(plan.steps)) if plan.steps else 0
            summaries.append(
                StepStateSummary(
                    plan_index=plan_index,
                    description=_summarize_step(step),
                    screenshot_url=_build_screenshot_url(flow_id, step.step_index),
                    state_kind=step.state_kind or "unknown",
                    url=step.url,
                )
            )
        return summaries
    finally:
        session.close()


class AgentA:
    def __init__(self, planner_llm: StructuredLLMClient, explainer_llm: StructuredLLMClient):
        self.planner_llm = planner_llm
        self.explainer_llm = explainer_llm

    async def plan_workflow(self, question: str) -> WorkflowPlan:
        prompt = build_planner_prompt(question)
        raw = await self.planner_llm.generate_json(prompt)
        plan = WorkflowPlan(
            app_name=raw.get("app_name", _normalize_app_name(question)),
            original_question=question,
            goal_summary=raw["goal_summary"],
            success_criteria=raw["success_criteria"],
            steps=[PlanStep(**s) for s in raw["steps"]],
        )
        return plan

    async def execute_with_agent_b(self, plan: WorkflowPlan) -> ExecutionTrace:
        query = f"{plan.app_name}: {plan.goal_summary}"
        flow_summary: OrchestratorFlowSummary = await run_task_query_async(query)

        step_states = await asyncio.to_thread(_collect_step_states, flow_summary.id, plan)
        status_reason: str | None = None
        session = SessionLocal()
        try:
            flow: Flow | None = session.query(Flow).get(flow_summary.id)
            if flow:
                status_reason = flow.status_reason
        finally:
            session.close()

        trace = ExecutionTrace(
            flow_id=str(flow_summary.id),
            app_name=plan.app_name,
            plan=plan,
            step_states=step_states,
            status=flow_summary.status,
            status_reason=status_reason,
        )
        return trace

    async def explain_to_user(self, trace: ExecutionTrace) -> str:
        prompt = build_explainer_prompt(trace)
        return await self.explainer_llm.generate_text(prompt)

    async def handle_user_question(self, question: str) -> tuple[ExecutionTrace, str]:
        plan = await self.plan_workflow(question)
        trace = await self.execute_with_agent_b(plan)
        explanation = await self.explain_to_user(trace)
        return trace, explanation


def create_default_agent_a() -> AgentA:
    planner_llm = create_structured_llm_client(max_new_tokens=512)
    explainer_llm = create_structured_llm_client(max_new_tokens=512)
    return AgentA(planner_llm=planner_llm, explainer_llm=explainer_llm)
