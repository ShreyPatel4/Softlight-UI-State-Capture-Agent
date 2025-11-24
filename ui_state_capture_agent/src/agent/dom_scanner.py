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
      - visible text inputs / editable regions for typing
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

    text_selector = (
        "input[type='text'], input[type='search'], input[type='email'], "
        "input[type='url'], textarea, [contenteditable='true'], [role='textbox']"
    )
    input_locator = page.locator(text_selector)
    input_count = await input_locator.count()
    for i in range(input_count):
        if len(candidates) >= max_actions:
            return candidates
        handle = input_locator.nth(i)
        if not await handle.is_visible():
            continue

        element_id = (await handle.get_attribute("id") or "").strip()
        label_text = ""
        if element_id:
            label_locator = page.locator(f"label[for=\"{element_id}\"]")
            if await label_locator.count() > 0:
                label_text = (await label_locator.first.inner_text() or "").strip()

        placeholder = (await handle.get_attribute("placeholder") or "").strip()
        aria_label = (await handle.get_attribute("aria-label") or "").strip()

        aria_labelledby = (await handle.get_attribute("aria-labelledby") or "").strip()
        labelledby_text = ""
        if aria_labelledby:
            for ref_id in aria_labelledby.split():
                ref_locator = page.locator(f"#{ref_id}")
                if await ref_locator.count() > 0:
                    labelledby_text = (await ref_locator.first.inner_text() or "").strip()
                    if labelledby_text:
                        break

        aria_describedby = (await handle.get_attribute("aria-describedby") or "").strip()
        describedby_text = ""
        if aria_describedby:
            for ref_id in aria_describedby.split():
                ref_locator = page.locator(f"#{ref_id}")
                if await ref_locator.count() > 0:
                    describedby_text = (await ref_locator.first.inner_text() or "").strip()
                    if describedby_text:
                        break

        ancestor_hint = await handle.evaluate(
            "(el) => { let node = el.parentElement; while (node) { const text = (node.innerText || '').trim(); if (text) { return text; } node = node.parentElement; } return ''; }"
        )
        ancestor_hint = ancestor_hint.strip() if isinstance(ancestor_hint, str) else ""

        tag_name = (await handle.evaluate("(el) => el.tagName.toLowerCase()")) or ""
        contenteditable = bool((await handle.get_attribute("contenteditable")) is not None)
        if tag_name == "textarea":
            base_desc = "Multiline text area"
        elif contenteditable:
            base_desc = "Editable text area"
        elif tag_name == "input":
            input_type = (await handle.get_attribute("type") or "").lower()
            if input_type == "search":
                base_desc = "Search input"
            else:
                base_desc = "Text input"
        else:
            base_desc = "Text input"

        hint = next(
            (
                val
                for val in [
                    label_text or element_id,
                    aria_label,
                    labelledby_text,
                    placeholder,
                    describedby_text,
                    ancestor_hint,
                ]
                if val
            ),
            None,
        )

        semantic_hint = hint or base_desc
        normalized = semantic_hint.lower()
        if any(keyword in normalized for keyword in ["title", "name", "subject"]):
            desc = f"type {semantic_hint}".strip()
        elif any(keyword in normalized for keyword in ["description", "detail", "note", "comment", "summary"]):
            desc = f"type {semantic_hint}".strip()
        elif hint:
            desc = f"type into {semantic_hint}".strip()
        else:
            desc = "type into text input"

        locator_str = f"{text_selector} >> nth={i}"
        candidates.append(
            CandidateAction(
                id=f"input_{i}",
                locator=locator_str,
                action_type="type",
                description=desc,
            )
        )

    return candidates
