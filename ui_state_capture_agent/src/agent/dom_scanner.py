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


async def scan_candidate_actions(page: Page, max_actions: int = 40) -> List[CandidateAction]:
    actions: List[CandidateAction] = []

    async def build_text(element_locator, default_locator: str) -> tuple[str, str]:
        text = (await element_locator.inner_text()).strip()
        truncated_text = text[:60]
        if truncated_text:
            locator = f"text={truncated_text}"
        else:
            locator = default_locator
        return truncated_text, locator

    # Collect clickable button actions
    button_locator = page.locator("button")
    button_count = await button_locator.count()
    for i in range(button_count):
        if len(actions) >= max_actions:
            return actions
        element = button_locator.nth(i)
        if not await element.is_visible():
            continue
        locator_str = f"button >> nth={i}"
        text = (await element.inner_text() or "").strip()
        description = (
            f"Click button '{text}'" if text else f"Click button index {i}"
        )
        actions.append(
            CandidateAction(
                id=f"btn_{i}",
                action_type="click",
                locator=locator_str,
                description=description,
            )
        )

    # Collect clickable link actions
    link_locator = page.locator("a[href]")
    link_count = await link_locator.count()
    for i in range(link_count):
        if len(actions) >= max_actions:
            return actions
        element = link_locator.nth(i)
        if not await element.is_visible():
            continue
        text, locator = await build_text(element, f"a[href] >> nth={i}")
        description = f"Click link '{text}'" if text else "Click link"
        actions.append(
            CandidateAction(
                id=f"act_{len(actions) + 1}",
                action_type="click",
                locator=locator,
                description=description,
            )
        )

    # Collect input and textarea actions
    input_locator = page.locator("input, textarea")
    input_count = await input_locator.count()
    for i in range(input_count):
        if len(actions) >= max_actions:
            return actions
        element = input_locator.nth(i)
        if not await element.is_visible():
            continue
        placeholder = await element.get_attribute("placeholder")
        placeholder_text = (placeholder or "").strip()
        truncated_placeholder = placeholder_text[:60]
        locator = (
            f"input[placeholder=\"{truncated_placeholder}\"]"
            if truncated_placeholder
            else f"input, textarea >> nth={i}"
        )
        if truncated_placeholder:
            description = f"Type into input with placeholder '{truncated_placeholder}'"
        else:
            description = "Type into input or textarea"
        actions.append(
            CandidateAction(
                id=f"act_{len(actions) + 1}",
                action_type="type",
                locator=locator,
                description=description,
            )
        )

    return actions
