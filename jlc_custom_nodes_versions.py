# jlc_custom_nodes_versions.py

"""
JLC Custom Nodes Version Registry
---------------------------------

Central version registry for the JLC ComfyUI Nodes collection. Active node
families and shared helpers use one release version so their per-module
manifests cannot drift.

Released under the MIT License as part of the JLC ComfyUI Nodes Collection.
"""

# from ...jlc_custom_nodes_versions import JLC_LORA_LOADER_VERSION

JLC_RELEASE_VERSION = "2.3.0"
JLC_NON_RECURSIVE_COMP_VERSION = JLC_RELEASE_VERSION
JLC_SUPPORT_NODES_VERSION = JLC_RELEASE_VERSION

# Active node-family versions
JLC_CONTROLNET_VERSION = JLC_NON_RECURSIVE_COMP_VERSION
JLC_CONTROLNET_AUX_VERSION = JLC_RELEASE_VERSION
JLC_LORA_LOADER_VERSION = JLC_SUPPORT_NODES_VERSION
JLC_PADDED_NODES_VERSION = JLC_SUPPORT_NODES_VERSION
JLC_UTIL_NODES_VERSION = JLC_RELEASE_VERSION

# Shared helper / engine versions
JLC_MODEL_CACHE_CORE_VERSION = JLC_CONTROLNET_VERSION
JLC_CONTROLNET_HELPERS_VERSION = JLC_NON_RECURSIVE_COMP_VERSION
JLC_LORA_HELPERS_VERSION = JLC_SUPPORT_NODES_VERSION
JLC_ENGINE_HELPERS_VERSION = JLC_SUPPORT_NODES_VERSION
