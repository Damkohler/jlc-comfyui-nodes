"""
JLC Seed Generator
------------------

- JLC ComfyUI Nodes Collection
  - This node is part of the **JLC Custom Nodes for ComfyUI** collection
    developed by **J. L. Córdova**.

- Purpose
    A shared seed source for workflows where multiple samplers or multiple
    inference stages should use the same seed contract for one full prompt
    execution.

- Seed Contract
    ``-1`` is a first-class sentinel meaning **intentionally unseeded**.
    It is passed through unchanged. The JLC frontend companion locks native
    control-after-generate mutation while the sentinel is active, so queued
    prompts continue to receive ``-1`` rather than incrementing, decrementing,
    or randomizing it.

    Nonnegative values are ordinary numeric seeds in the range
    ``0 .. 2^63 - 1``.

- Stable Base Seed Display
    For ordinary nonnegative seeds, ComfyUI's native control-after-generate
    modes remain available (fixed / increment / decrement / randomize), while
    the frontend companion restores the visible seed widget to the user's base
    seed after queue submission. The status panel reports the seed actually
    executed by the backend.

- Randomize Replay
    The ``randomize_replay`` control is a frontend-assisted feature:

      off
          Normal ComfyUI randomize behavior.

      record
          Run normal randomization while recording the actual executed seed
          sequence reported by the backend.

      replay
          Reuse the most recently recorded random sequence in queue order.

    Record/replay is intentionally meaningful only when ComfyUI's native
    control-after-generate mode is ``randomize``. The replay sequence is stored
    in the node's workflow properties by the frontend companion.

- Outputs
    ``seed``
        Small SEED-style dictionary: ``{"seed": value}``.

    ``seed_int``
        Plain integer seed, including ``-1`` when intentionally unseeded.

- Attribution & License
  - Concept and implementation by **J. L. Córdova** with development
    assistance from **ChatGPT (OpenAI)**.

  - Designed for use with:
    https://github.com/comfyanonymous/ComfyUI

  - Copyright (c) 2026 J. L. Córdova
  - Released under the **MIT License**.
"""

from __future__ import annotations

from typing import Any

from ...jlc_custom_nodes_versions import JLC_UTIL_NODES_VERSION

MAX_SEED = 0x7FFFFFFFFFFFFFFF  # 2^63 - 1
UNSEEDED = -1
REPLAY_MODES = ("off", "record", "replay")

MANIFEST = {
    "name": "JLC Seed Generator",
    "version": JLC_UTIL_NODES_VERSION,
    "author": "J. L. Córdova",
    "description": (
        "Shared seed source with a first-class -1 unseeded sentinel, stable "
        "base-seed display, and frontend-assisted randomize record/replay."
    ),
}


class JLC_SeedGenerator:
    """Shared seed source with explicit unseeded and replay-aware semantics."""

    FUNCTION = "generator"
    CATEGORY = "utils/seed"

    RETURN_TYPES = ("SEED", "INT")
    RETURN_NAMES = ("seed", "seed_int")
    OUTPUT_NODE = False

    def __init__(self):
        self.last_seed: int | None = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": UNSEEDED,
                        "max": MAX_SEED,
                        "control_after_generate": True,
                        "tooltip": (
                            "Base seed. Use -1 for intentionally unseeded execution. "
                            "When -1 is active, the JLC frontend locks native seed "
                            "mutation so every queued prompt receives -1. For ordinary "
                            "nonnegative seeds, fixed/increment/decrement/randomize work "
                            "normally while this visible widget is restored to the base value."
                        ),
                    },
                ),
                "randomize_replay": (
                    REPLAY_MODES,
                    {
                        "default": "off",
                        "tooltip": (
                            "Optional companion for native Randomize mode. Off uses normal "
                            "ComfyUI randomization. Record stores the actual executed random "
                            "seed sequence. Replay reuses the stored sequence. Ignored for "
                            "fixed/increment/decrement and while seed = -1."
                        ),
                    },
                ),
                # Reserved display row for the frontend status panel. Keep this
                # as a real widget so LiteGraph reserves vertical space for the
                # custom-drawn panel. The backend intentionally ignores it.
                "spacer": (
                    "STRING",
                    {
                        "default": "──────── JLC seed status ────────",
                        "multiline": False,
                        "tooltip": (
                            "Frontend status panel: executed seed, unseeded sentinel state, "
                            "and randomize record/replay status."
                        ),
                    },
                ),
            }
        }

    @staticmethod
    def _coerce_seed(seed: Any) -> int:
        """Normalize an input while preserving -1 as the sole negative sentinel."""
        if seed is None:
            return 0

        try:
            value = int(seed)
        except (TypeError, ValueError, OverflowError):
            return 0

        if value == UNSEEDED:
            return UNSEEDED

        # Preserve the legacy behavior for invalid negative values: ordinary
        # seeds are clamped to the valid nonnegative range, but -1 is handled
        # explicitly above rather than being created by a generic clamp.
        return max(0, min(value, MAX_SEED))

    @classmethod
    def IS_CHANGED(cls, seed=0, randomize_replay="off", spacer=None):
        """
        Treat the -1 sentinel as volatile.

        An intentionally unseeded downstream consumer may generate a different
        result even when every serialized workflow input is otherwise identical.
        Returning NaN prevents ComfyUI from treating the seed source as a stable
        cached value in that case. Ordinary numeric seeds retain normal caching.
        """
        current_seed = cls._coerce_seed(seed)
        if current_seed == UNSEEDED:
            return float("nan")
        return current_seed

    def generator(self, seed=0, randomize_replay="off", spacer=None):
        current_seed = self._coerce_seed(seed)
        self.last_seed = current_seed

        # The replay selector is frontend behavior. Normalize it here only so a
        # malformed API/workflow value cannot leak surprising state into UI data.
        replay_mode = str(randomize_replay).strip().lower()
        if replay_mode not in REPLAY_MODES:
            replay_mode = "off"

        seed_text = str(current_seed)
        state_text = "unseeded" if current_seed == UNSEEDED else "seeded"

        return {
            "result": (
                {"seed": current_seed},
                current_seed,
            ),
            # ComfyUI forwards these values to node.onExecuted(message).
            # String values preserve the exact decimal representation of large
            # integer seeds across the frontend boundary.
            "ui": {
                "jlc_seed": [seed_text],
                "seed": [seed_text],
                "jlc_seed_state": [state_text],
                "jlc_replay_mode": [replay_mode],
            },
        }


NODE_CLASS_MAPPINGS = {
    "JLC_SeedGenerator": JLC_SeedGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JLC_SeedGenerator": "JLC Seed Generator",
}
