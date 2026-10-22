"""JLC Text Concatenate: independently routable and ordered prompt sections.

The backend declares one combined STRING output plus 32 individually addressable
STRING outputs. The frontend exposes only the number selected in Update Slots;
section identity is its fixed 1-based slot index, not its concatenation order.
"""

from __future__ import annotations

import json


MAX_SECTIONS = 32


class JLC_TextConcatenate:
    CATEGORY = "utils"
    FUNCTION = "concatenate"
    RETURN_TYPES = ("STRING",) * (MAX_SECTIONS + 1)
    RETURN_NAMES = ("text",) + tuple(
        f"section_{number:02d}" for number in range(1, MAX_SECTIONS + 1)
    )
    DESCRIPTION = (
        "Edit up to 32 prompt sections; Update Slots controls individual STRING "
        "outputs. Enable toggles affect only combined text. Positive order "
        "numbers sort first, followed by negative orders (-1 last). "
        "Enter inserts a newline."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # This ordinary widget is made invisible by the JS frontend.
                # Keep its name to migrate pre-release string-array workflows.
                "sections": ("STRING", {"default": '[""]'}),
                "number_of_sections": (
                    "INT", {"default": 1, "min": 1, "max": MAX_SECTIONS, "step": 1}
                ),
            }
        }

    def concatenate(self, sections: str, number_of_sections: int = 1):
        try:
            raw = json.loads(sections)
        except (TypeError, ValueError) as exc:
            raise ValueError("Text sections must contain a valid JSON array.") from exc

        if not isinstance(raw, list):
            raise ValueError("Text sections must be a JSON array.")
        if len(raw) > MAX_SECTIONS:
            raise ValueError(f"Text Concatenate supports at most {MAX_SECTIONS} sections.")
        if not 1 <= number_of_sections <= MAX_SECTIONS:
            raise ValueError(f"number_of_sections must be between 1 and {MAX_SECTIONS}.")

        section_texts = [""] * MAX_SECTIONS
        included = []

        for index, entry in enumerate(raw[:number_of_sections]):
            # Compatibility with the original ["first", "second", ...] format.
            if isinstance(entry, str):
                text, enabled, order = entry, True, index + 1
            elif isinstance(entry, dict):
                text = entry.get("text", "")
                enabled = entry.get("enabled", True)
                order = entry.get("order", index + 1)
                if not isinstance(text, str):
                    raise ValueError(f"Section {index + 1}: text must be a string.")
                if not isinstance(enabled, bool):
                    raise ValueError(f"Section {index + 1}: enabled must be a boolean.")
                if isinstance(order, bool) or not isinstance(order, int) or order == 0:
                    raise ValueError(f"Section {index + 1}: order must be a nonzero integer.")
            else:
                raise ValueError(f"Section {index + 1}: expected a string or object.")

            # Individual output is deliberately unaffected by enable/order.
            section_texts[index] = text
            if enabled and text.strip():
                # Positives first (ascending), then negatives (ascending).
                # Physical section index breaks ties deterministically.
                included.append(((order < 0, order, index), text))

        included.sort(key=lambda item: item[0])
        combined = "\n\n".join(text for _, text in included)
        return (combined, *section_texts)
