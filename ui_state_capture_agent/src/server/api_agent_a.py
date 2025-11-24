from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from ..agent.agent_a import AgentA, StepStateSummary, create_default_agent_a

router = APIRouter()

_agent_a: AgentA = create_default_agent_a()


class AgentARequest(BaseModel):
    question: str


class AgentAResponse(BaseModel):
    flow_id: str
    status: str
    status_reason: str | None
    steps: list[StepStateSummary]
    explanation: str


@router.post("/agent_a/run", response_model=AgentAResponse)
async def run_agent_a(payload: AgentARequest) -> AgentAResponse:
    trace, explanation = await _agent_a.handle_user_question(payload.question)
    return AgentAResponse(
        flow_id=trace.flow_id,
        status=trace.status,
        status_reason=trace.status_reason,
        steps=trace.step_states,
        explanation=explanation,
    )
