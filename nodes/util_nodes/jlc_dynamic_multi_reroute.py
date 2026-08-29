"""
JLC Dynamic Multi Reroute — Python Declaration Helper
------------------------------------------------------

- JLC ComfyUI Nodes Collection
  - This helper module is part of the **JLC Custom Nodes for ComfyUI**
    collection developed by **J. L. Córdova**.

  - Repository:
    https://github.com/Damkohler/jlc-comfyui-nodes

- Helper Purpose
  - Declares the backend-visible class for:
        • JLC Dynamic Multi Reroute

  - The companion frontend file `web/jlc_dynamic_multi_reroute.js` is the
    primary implementation. It owns dynamic row creation/compaction, stable row
    identities, type inference, socket/link coloring, sizing, serialization,
    and virtual graph-link resolution.

  - This Python module provides:
        • ComfyUI class registration surface
        • one initial wildcard input/output declaration
        • description and release metadata
        • a stateless one-row passthrough fallback if the frontend does not load

- Execution Boundary
  - Under normal operation, the frontend marks the node as virtual and resolves
    every connected output row to the real upstream graph link before the prompt
    reaches backend execution.

  - The reroute therefore does not copy tensors, retain values, or add runtime
    execution stages.

- Dynamic Row Semantics
  - Begins with one wildcard input/output pair.
  - Supports up to twenty-four independent pairs.
  - Keeps one trailing spare pair while capacity remains.
  - Fully disconnected rows may be removed from any position; surviving rows
    compact upward while LiteGraph reindexes their live links.

- Versioning
  - Version is governed by `JLC_UTIL_NODES_VERSION` from
    `jlc_custom_nodes_versions.py`.

- Attribution & License
  - Concept and implementation by **J. L. Córdova**
    with development assistance from **ChatGPT (OpenAI)**.

  - Designed for use with:
    https://github.com/comfyanonymous/ComfyUI

  - Copyright (c) 2026 J. L. Córdova

  - Released under the **MIT License**.
"""

from __future__ import annotations

from ...jlc_custom_nodes_versions import JLC_UTIL_NODES_VERSION


MANIFEST = {
    "name": "JLC Dynamic Multi Reroute",
    "version": JLC_UTIL_NODES_VERSION,
    "author": "J. L. Córdova",
    "description": (
        "Frontend-driven virtual multi-reroute providing one-to-twenty-four "
        "independent wildcard input/output pairs. Rows grow automatically, "
        "fully disconnected rows compact safely from any position, each row "
        "infers and displays its resolved ComfyUI datatype/color, and connected "
        "outputs resolve directly to their real upstream graph links before "
        "backend execution."
    ),
}


class _AnyType(str):
    """A ComfyUI wildcard that compares compatible with every socket type."""

    def __ne__(self, _other: object) -> bool:
        return False


ANY_TYPE = _AnyType("*")


class JLC_DynamicMultiReroute:
    """Frontend-virtual multi-reroute with one initial wildcard pair."""

    CATEGORY = "utils"
    FUNCTION = "passthrough"
    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("value_1",)
    DESCRIPTION = (
        "One-to-twenty-four independent reroute pairs in one compact node. "
        "The frontend automatically adds one spare row, compacts fully "
        "disconnected rows, infers each row's datatype and standard ComfyUI "
        "socket/link color, and resolves every used output directly to its "
        "upstream source before prompt execution."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "value_1": (ANY_TYPE,),
            }
        }

    def passthrough(self, value_1=None):
        """Defensive fallback for the initial row if the frontend fails to load."""
        return (value_1,)
