# jlc_custom_nodes_versions.py

"""
JLC Custom Nodes Version Registry
---------------------------------

Central version registry for the JLC ComfyUI Nodes collection.

For this release, active node families are intentionally kept on a unified
release version to avoid stale per-file manifest drift. Helper/core modules
also use the shared release version unless a future API-specific version split
is needed.

Released under the MIT License as part of the JLC ComfyUI Nodes Collection.
"""

# from ...jlc_custom_nodes_versions import JLC_LORA_LOADER_VERSION

JLC_NON_RECURSIVE_COMP_VERSION = "2.1.0"
JLC_SUPPORT_NODES_VERSION = "1.5.0"

# Active node-family versions
JLC_CONTROLNET_VERSION = JLC_NON_RECURSIVE_COMP_VERSION
JLC_CONTROLNET_AUX_VERSION = "2.1.0"
JLC_LORA_LOADER_VERSION = JLC_SUPPORT_NODES_VERSION
JLC_PADDED_NODES_VERSION = JLC_SUPPORT_NODES_VERSION
JLC_UTIL_NODES_VERSION = "1.6.0"

# Shared helper / engine versions
JLC_MODEL_CACHE_CORE_VERSION = JLC_CONTROLNET_VERSION
JLC_CONTROLNET_HELPERS_VERSION = JLC_NON_RECURSIVE_COMP_VERSION
JLC_LORA_HELPERS_VERSION = JLC_SUPPORT_NODES_VERSION
JLC_ENGINE_HELPERS_VERSION = JLC_SUPPORT_NODES_VERSION