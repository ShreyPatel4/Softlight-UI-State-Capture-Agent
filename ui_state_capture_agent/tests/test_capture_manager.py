import sys
import asyncio
import asyncio
from types import SimpleNamespace

from src.agent.capture import CaptureManager
from src.models import Step


class DummySession:
    def __init__(self):
        self.steps = []

    def query(self, *_args, **_kwargs):
        session = self

        class Q:
            def filter(self, *_):
                return self

            def order_by(self, *_):
                return self

            def first(self):
                if not session.steps:
                    return None
                max_index = max(step.step_index for step in session.steps)
                return (max_index,)

        return Q()

    def add(self, obj, *_args, **_kwargs):
        if isinstance(obj, Step):
            self.steps.append(obj)
        return None

    def commit(self):
        return None

    def refresh(self, obj, *_args, **_kwargs):
        return obj


class DummyStorage:
    def __init__(self):
        self.saved_keys = []

    def save_bytes(self, key, *_args, **_kwargs):
        self.saved_keys.append(key)
        return None


class DummyPage:
    url = "http://example.com"

    async def screenshot(self, full_page: bool = True):
        return b"image"

    async def content(self):
        return "<html></html>"

def test_capture_step_accepts_state_kind_and_url_changed():
    capture_manager = CaptureManager(DummySession(), DummyStorage())
    flow = SimpleNamespace(id="flow1", prefix="pref")

    step = asyncio.run(
        capture_manager.capture_step(
            page=DummyPage(),
            flow=flow,
            label="test_state",
            dom_html="<html></html>",
            diff_summary="Minor or no structural change",
            diff_score=0.0,
            action_description="",
            url_changed=True,
            state_kind="dom_change",
        )
    )

    assert step.state_label == "test_state"
    assert step.url_changed is True
    assert step.state_kind == "dom_change"


def test_capture_step_increments_indices_and_keys():
    storage = DummyStorage()
    session = DummySession()
    capture_manager = CaptureManager(session, storage)
    flow = SimpleNamespace(id="flow1", prefix="pref")

    first_step = asyncio.run(
        capture_manager.capture_step(
            page=DummyPage(),
            flow=flow,
            label="first",
            dom_html="<html></html>",
            diff_summary=None,
            diff_score=None,
            action_description="",
            url_changed=True,
            state_kind="dom_change",
        )
    )

    second_step = asyncio.run(
        capture_manager.capture_step(
            page=DummyPage(),
            flow=flow,
            label="second",
            dom_html="<html></html>",
            diff_summary=None,
            diff_score=None,
            action_description="",
            url_changed=False,
            state_kind="dom_change",
        )
    )

    assert first_step.step_index == 1
    assert second_step.step_index == 2
    assert first_step.screenshot_key != second_step.screenshot_key
    assert len(storage.saved_keys) == 4
