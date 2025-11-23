import uuid

from src.agent.dom_scanner import CandidateAction
from src.agent.policy import PolicyDecision, _extract_json, choose_action_with_llm
from src.agent.task_spec import TaskSpec
from src.models import FlowLog


class DummyLLM:
    def __init__(self, output: str):
        self.output = output

    def complete(self, prompt: str) -> str:  # noqa: ARG002
        return self.output


class DummySession:
    def __init__(self, flow=None):
        self.logs = []
        self.flow = flow

    def add(self, obj):
        if isinstance(obj, FlowLog):
            self.logs.append(obj)

    def commit(self):
        return None

    def refresh(self, obj, *_args, **_kwargs):
        return obj

    def get(self, *_args, **_kwargs):
        return self.flow


def test_extract_json_variants():
    assert _extract_json('{"a": 1}') == {"a": 1}
    fenced = """```json\n{\n  \"a\": 2\n}\n```"""
    assert _extract_json(fenced) == {"a": 2}
    noisy = "Here is the result: {\"a\":3} and some trailing text"
    assert _extract_json(noisy) == {"a": 3}


def test_choose_action_valid_json():
    candidates = [CandidateAction(id="btn_0", action_type="click", locator="locator", description="button A")]
    llm = DummyLLM('{"action_id": "btn_0", "action_type": "click", "done": false}')
    task = TaskSpec(original_query="", app_name="linear", goal="go", start_url="http://example.com")

    decision = choose_action_with_llm(
        llm,
        task,
        task.app_name,
        "http://example.com",
        "",
        candidates,
    )

    assert isinstance(decision, PolicyDecision)
    assert decision.action_id == "btn_0"
    assert decision.label == "after_action_btn_0"
    assert decision.text_to_type is None


def test_choose_action_with_text_to_type():
    candidates = [CandidateAction(id="input_1", action_type="type", locator="locator", description="Text input labeled 'Title'")]
    llm = DummyLLM('{"action_id": "input_1", "action_type": "type", "text_to_type": "Some value", "done": false}')
    task = TaskSpec(original_query="", app_name="linear", goal="go", start_url="http://example.com")

    decision = choose_action_with_llm(
        llm,
        task,
        task.app_name,
        "http://example.com",
        "",
        candidates,
    )

    assert decision.text_to_type == "Some value"


def test_choose_action_fallback_logs_warning():
    candidates = [CandidateAction(id="btn_0", action_type="click", locator="locator", description="button A")]
    llm = DummyLLM("nonsense without json")
    task = TaskSpec(original_query="", app_name="linear", goal="go", start_url="http://example.com")
    flow = type("Flow", (), {"id": uuid.uuid4()})()
    session = DummySession(flow)

    decision = choose_action_with_llm(
        llm,
        task,
        task.app_name,
        "http://example.com",
        "",
        candidates,
        session=session,
        flow=flow,
    )

    assert decision.action_id is None
    assert decision.done is True
    assert session.logs
    assert any("LLM output missing valid JSON" in log.message for log in session.logs)


def test_choose_action_invalid_json_sets_text_none():
    candidates = [CandidateAction(id="btn_0", action_type="click", locator="locator", description="button A")]
    llm = DummyLLM("{not json}")
    task = TaskSpec(original_query="", app_name="linear", goal="go", start_url="http://example.com")

    decision = choose_action_with_llm(
        llm,
        task,
        task.app_name,
        "http://example.com",
        "",
        candidates,
    )

    assert decision.text_to_type is None
