from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal

from playwright.async_api import Page

ActionType = Literal["click", "type"]


@dataclass
class CandidateAction:
    id: str
    action_type: ActionType
    locator: str
    description: str


async def scan_candidate_actions(page: Page) -> List[CandidateAction]:
    candidates: List[CandidateAction] = []

    button_locator = page.locator("button")
    count = await button_locator.count()
    for i in range(count):
        handle = button_locator.nth(i)
        if not await handle.is_visible():
            continue
        text = (await handle.inner_text() or "").strip()
        locator_str = f"button >> nth={i}"
        desc = f"button with text '{text}'" if text else f"button index {i}"
        candidates.append(
            CandidateAction(
                id=f"btn_{i}",
                locator=locator_str,
                action_type="click",
                description=desc,
            )
        )

    # TODO: extend for links, inputs, etc
    return candidates
