from typing import Optional

from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from .task_spec import TaskSpec
from .dom_scanner import scan_candidate_actions
from .policy import Policy
from .browser import BrowserSession
from .capture import CaptureManager
from ..models import Flow
from .state_diff import diff_dom


async def run_agent_loop(
    task: TaskSpec,
    flow: Flow,
    capture_manager: CaptureManager,
    policy: Policy,
    start_url: str,
    max_steps: int = 15,
) -> None:
    async with BrowserSession() as browser:
        await browser.goto(start_url)

        prev_dom: Optional[str] = None
        history_summary = ""

        for step_index in range(1, max_steps + 1):
            page = browser.page
            if page is None:
                break

            current_dom = await browser.get_dom()
            dom_diff = diff_dom(prev_dom, current_dom)  # noqa: F841

            candidates = await scan_candidate_actions(page)
            print(f"[agent_loop] URL={page.url} candidates={len(candidates)}")
            if not candidates:
                capture_manager.finish_flow(flow, status="no_actions")
                break

            print(f"[agent_loop] history_summary length={len(history_summary or '')}")
            decision = await policy.choose_action(task, candidates, history_summary)

            if decision.get("done"):
                if decision.get("capture_after"):
                    screenshot_bytes = await page.screenshot(full_page=True)
                    capture_manager.capture_step(
                        flow=flow,
                        step_index=step_index,
                        state_label=decision.get("state_label_after", "done"),
                        description=decision.get("reason", ""),
                        page_url=page.url,
                        screenshot_bytes=screenshot_bytes,
                        dom_html=current_dom,
                    )

                capture_manager.finish_flow(flow, status="success")
                break

            if decision.get("capture_before"):
                screenshot_bytes = await page.screenshot(full_page=True)
                capture_manager.capture_step(
                    flow=flow,
                    step_index=step_index,
                    state_label=f"before_{step_index}",
                    description=f"Before action: {decision.get('reason', '')}",
                    page_url=page.url,
                    screenshot_bytes=screenshot_bytes,
                    dom_html=current_dom,
                )

            cand = next((c for c in candidates if c.id == decision.get("chosen_action_id")), candidates[0])

            if decision.get("action_type") == "click":
                locator = page.locator(cand.locator)

                try:
                    if not await locator.is_visible():
                        print(
                            f"[agent_loop] Skipping action {cand.id}: locator {cand.locator} is not visible"
                        )
                        continue
                except PlaywrightTimeoutError:
                    print(
                        f"[agent_loop] Visibility check timed out for {cand.id} ({cand.locator}), skipping"
                    )
                    continue

                try:
                    await locator.click(timeout=2000)
                except PlaywrightTimeoutError:
                    print(
                        f"[agent_loop] Click timed out for {cand.id} ({cand.locator}), skipping this action"
                    )
                    history_summary = (history_summary or "") + (
                        f"\nAction {cand.id} with locator {cand.locator} failed (click timeout)."
                    )
                    continue
            elif decision.get("action_type") == "type":
                await page.locator(cand.locator).fill(decision.get("input_text", ""))

            await page.wait_for_timeout(1000)

            new_dom = await browser.get_dom()

            if decision.get("capture_after"):
                screenshot_bytes = await page.screenshot(full_page=True)
                capture_manager.capture_step(
                    flow=flow,
                    step_index=step_index,
                    state_label=decision.get("state_label_after", f"after_{step_index}"),
                    description=decision.get("reason", ""),
                    page_url=page.url,
                    screenshot_bytes=screenshot_bytes,
                    dom_html=new_dom,
                )

            summary_line = f"{step_index}. {decision.get('reason', '')}".strip()
            history_summary = "\n".join(
                [line for line in [history_summary, summary_line] if line]
            )

            prev_dom = new_dom
        else:
            capture_manager.finish_flow(flow, status="max_steps_reached")
