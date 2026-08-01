"""
JLC Dynamic Multi Set/Get — Python Declaration Helper
------------------------------------------------------

- JLC ComfyUI Nodes Collection
  - This helper module is part of the **JLC Custom Nodes for ComfyUI**
    collection developed by **J. L. Córdova**.

  - Repository:
    https://github.com/Damkohler/jlc-comfyui-nodes

- Helper Purpose
  - Declares the backend-visible classes for:
        • JLC Dynamic Multi Set
        • JLC Dynamic Multi Get

  - The companion frontend file `web/jlc_dynamic_multi_set_get.js` is the
    primary implementation. It owns dynamic rows, channel naming and choices,
    type presentation, workflow metadata, resizing, and virtual graph-link
    resolution.

  - This Python module provides:
        • ComfyUI class registration surfaces
        • two initial wildcard input/output declarations
        • descriptions and release metadata
        • stateless fallback methods if the frontend extension does not load

- Execution Boundary
  - Under normal operation, the frontend marks both classes as virtual nodes
    and resolves connected outputs to the real upstream graph link before the
    prompt reaches backend execution.

  - No Python runtime registry is used to transport Set/Get values.

  - The Multi Set passthrough method is a defensive fallback. The Multi Get
    fallback raises a clear error because reaching backend execution means the
    required frontend virtual-node resolution did not occur.

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
    "name": "JLC Dynamic Multi Set/Get",
    "version": JLC_UTIL_NODES_VERSION,
    "author": "J. L. Córdova",
    "description": (
        "Frontend-driven virtual-node pair providing two-to-sixteen dynamic, "
        "independently named wildcard Set/Get channels. The JavaScript "
        "implementation manages graph-local channel discovery, automatic names, "
        "restricted Get selection, stable rename propagation, dynamic row "
        "growth and compaction, type inference, responsive layout, and stateless "
        "resolution to the real upstream ComfyUI graph links before prompt "
        "submission."
    ),
}


class _AnyType(str):
    """A ComfyUI wildcard that compares compatible with every socket type."""

    def __ne__(self, _other: object) -> bool:
        return False


ANY_TYPE = _AnyType("*")


class JLC_DynamicMultiSet:
    """Frontend-virtual multi-channel Set with two initial wildcard rows."""

    CATEGORY = "utils"
    FUNCTION = "passthrough"
    RETURN_TYPES = (ANY_TYPE, ANY_TYPE)
    RETURN_NAMES = ("value_1", "value_2")
    DESCRIPTION = (
        "Two-to-sixteen independently named wildcard channels. The frontend "
        "resolves passthrough outputs and wireless consumers to real graph "
        "links; no runtime value registry is used."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "value_1": (ANY_TYPE,),
                "value_2": (ANY_TYPE,),
            }
        }

    def passthrough(self, value_1=None, value_2=None):
        return (value_1, value_2)


class JLC_DynamicMultiGet:
    """Frontend-virtual multi-channel Get with two initial wildcard rows."""

    CATEGORY = "utils"
    FUNCTION = "unresolved_frontend_node"
    RETURN_TYPES = (ANY_TYPE, ANY_TYPE)
    RETURN_NAMES = ("value_1", "value_2")
    DESCRIPTION = (
        "Resolves each named channel to a matching KJ Set or JLC Dynamic Multi "
        "Set through ComfyUI's graph-level virtual-node protocol."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    def unresolved_frontend_node(self):
        raise RuntimeError(
            "JLC Dynamic Multi Get is a frontend virtual node. Its web "
            "extension did not resolve this node before prompt submission."
        )
