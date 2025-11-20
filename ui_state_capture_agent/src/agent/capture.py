from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy.orm import Session

from ..config import settings
from ..models import Flow, Step
from ..storage.base import StorageBackend


class CaptureManager:
    def __init__(self, db_session: Session, storage: StorageBackend) -> None:
        self.db_session = db_session
        self.storage = storage

    def start_flow(self, app_name: str, task_id: str, task_title: str, task_blurb: str) -> Flow:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        prefix = f"{app_name}/{task_id}/{run_id}"

        flow = Flow(
            app_name=app_name,
            task_id=task_id,
            task_title=task_title,
            task_blurb=task_blurb,
            run_id=run_id,
            status="running",
            started_at=datetime.now(timezone.utc),
            bucket=settings.minio_bucket,
            prefix=prefix,
        )
        self.db_session.add(flow)
        self.db_session.commit()
        self.db_session.refresh(flow)
        return flow

    def capture_step(
        self,
        flow: Flow,
        step_index: int,
        state_label: str,
        description: str,
        page_url: str,
        screenshot_bytes: bytes,
        dom_html: str,
    ) -> Step:
        screenshot_key = f"{flow.prefix}/step_{step_index}_screenshot.png"
        dom_key = f"{flow.prefix}/step_{step_index}_dom.html"

        self.storage.save_bytes(screenshot_key, screenshot_bytes)
        self.storage.save_bytes(dom_key, dom_html.encode("utf-8"))

        step = Step(
            flow_id=flow.id,
            step_index=step_index,
            state_label=state_label,
            description=description,
            url=page_url,
            screenshot_key=screenshot_key,
            dom_key=dom_key,
        )
        self.db_session.add(step)
        self.db_session.commit()
        self.db_session.refresh(step)
        return step

    def finish_flow(self, flow: Flow, status: str) -> None:
        flow.status = status
        flow.finished_at = datetime.now(timezone.utc)
        self.db_session.add(flow)
        self.db_session.commit()
