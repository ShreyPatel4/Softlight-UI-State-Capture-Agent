import sys
import asyncio
from types import SimpleNamespace

from src.agent.capture import CaptureManager


class DummySession:
    def query(self, *_args, **_kwargs):
        class Q:
            def filter(self, *_):
                return self

            def order_by(self, *_):
                return self

            def first(self):
                return None

        return Q()

    def add(self, *_args, **_kwargs):
        return None

    def commit(self):
        return None

    def refresh(self, obj, *_args, **_kwargs):
        return obj


class DummyStorage:
    def save_bytes(self, *_args, **_kwargs):
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
