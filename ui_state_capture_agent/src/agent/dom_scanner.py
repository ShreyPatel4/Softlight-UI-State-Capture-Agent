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


async def scan_candidate_actions(page: Page, max_actions: int = 60) -> List[CandidateAction]:
    """
    Generic DOM scanner:
      - visible buttons and role=button
      - visible links
    Good enough for Linear / Notion without hardcoding workflows.
    """
    candidates: List[CandidateAction] = []

    # Buttons + ARIA button-like elements
    button_locator = page.locator("button, [role='button']")
    button_count = await button_locator.count()
    for i in range(button_count):
        if len(candidates) >= max_actions:
            return candidates
        handle = button_locator.nth(i)
        if not await handle.is_visible():
            continue
        text = (await handle.get_attribute("aria-label") or await handle.inner_text() or "").strip()
        locator_str = f"button, [role='button'] >> nth={i}"
        desc = (
            f"button \"{text}\"" if text else f"button index {i}"
        )
        candidates.append(
            CandidateAction(
                id=f"btn_{i}",
                locator=locator_str,
                action_type="click",
                description=desc,
            )
        )

    # Clickable links (Linear/Notion use these a lot for nav)
    link_locator = page.locator("a[href]")
    link_count = await link_locator.count()
    for i in range(link_count):
        if len(candidates) >= max_actions:
            return candidates
        handle = link_locator.nth(i)
        if not await handle.is_visible():
            continue
        text = (await handle.inner_text() or await handle.get_attribute("aria-label") or "").strip()
        locator_str = f"a[href] >> nth={i}"
        desc = f"link \"{text}\"" if text else f"link index {i}"
        candidates.append(
            CandidateAction(
                id=f"link_{i}",
                locator=locator_str,
                action_type="click",
                description=desc,
            )
        )

    # Text inputs and textareas for typing actions
    input_locator = page.locator("input, textarea")
    input_count = await input_locator.count()
    for i in range(input_count):
        if len(candidates) >= max_actions:
            return candidates
        handle = input_locator.nth(i)
        if not await handle.is_visible():
            continue
        placeholder = await handle.get_attribute("placeholder")
        ph = (placeholder or "").strip()
        locator_str = f"input, textarea >> nth={i}"
        desc = (
            f"input placeholder \"{ph}\"" if ph else "input/textarea"
        )
        candidates.append(
            CandidateAction(
                id=f"input_{i}",
                locator=locator_str,
                action_type="type",
                description=desc,
            )
        )

    return candidates
