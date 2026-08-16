"""
JLC ComfyUI Custom Nodes
------------------------

- JLC ComfyUI Nodes Collection
  - This node is part of the **JLC Custom Nodes for ComfyUI**
    collection developed by **J. L. Córdova**.

  - Repository
    https://github.com/Damkohler/jlc-comfyui-nodes

  - The JLC nodes focus on practical workflow improvements for
    image generation pipelines, particularly:
        • Flux-based workflows
        • LoRA experimentation
        • advanced inpainting / outpainting pipelines

- This module registers all nodes in the JLC node collection.

- ComfyUI loads the package located in `custom_nodes/<repo_name>/` and reads
  the following mappings:

    NODE_CLASS_MAPPINGS
    NODE_DISPLAY_NAME_MAPPINGS

- Each entry maps an internal class name to the class implementing the node.

- Actual node implementations live in the `nodes/` subdirectory to keep the
  repository organized as the number of nodes grows.

- Author: J. L. Córdova
  License: MIT
"""

import os
from server import PromptServer  # used for static route mounting

# ControlNet Nodes
from .nodes.jlc_padded_image import JLC_PaddedImage
from .nodes.jlc_padded_latent import JLC_PaddedLatent
from .nodes.jlc_controlnet_apply import JLC_ControlNetApply
from .nodes.jlc_controlnet_apply_advanced import JLC_ControlNetApplyAdvanced
from .nodes.jlc_controlnet_composition import JLC_ControlNetComposition
from .nodes.jlc_controlnet_orchestrator import JLC_ControlNetOrchestrator
from .nodes.jlc_controlnet_orchestrator_adv import JLC_ControlNetOrchestratorAdvanced
from .nodes.controlnet_aux_nodes.jlc_controlnet_aux_preproc_wrapper import JLC_DynamicAuxPreprocessorWrapper

# LoRA Loaders
from .nodes.lora_loader_nodes.jlc_dynamic_lora_loader_model_only import JLC_DynamicLoraLoaderModelOnly
from .nodes.lora_loader_nodes.jlc_dynamic_lora_loader_model_clip import JLC_DynamicLoraLoaderModelClip
from .nodes.lora_loader_nodes.jlc_dynamic_lora_loader_shared_block_weight_model_only import JLC_DynamicLoraLoaderSharedBlockWeightModelOnly
from .nodes.lora_loader_nodes.jlc_dynamic_lora_loader_shared_block_weight_model_clip import JLC_DynamicLoraLoaderSharedBlockWeightModelClip
from .nodes.lora_loader_nodes.jlc_dynamic_lora_loader_block_weight_model_only import JLC_DynamicLoraLoaderBlockWeightModelOnly
from .nodes.lora_loader_nodes.jlc_dynamic_lora_loader_block_weight_model_clip import JLC_DynamicLoraLoaderBlockWeightModelClip

# Legacy LoRA wrappers: this file exports two legacy node.
from .nodes.lora_loader_nodes.jlc_lora_loader_legacy_dynamic_wrappers import (
    NODE_CLASS_MAPPINGS as legacy_lora_class_mappings,
    NODE_DISPLAY_NAME_MAPPINGS as legacy_lora_display_name_mappings,
)

# Utility Nodes
from .nodes.util_nodes.jlc_seed_generator import JLC_SeedGenerator
from .nodes.util_nodes.jlc_stage_boundary_vram_cleanup import JLC_StageBoundaryVRAMCleanup
from .nodes.util_nodes.jlc_load_resize_encode_image import JLC_LoadResizeEncodeImage
from .nodes.util_nodes.jlc_resize_image import JLC_ResizeImage
from .nodes.util_nodes.jlc_resize_images import JLC_ResizeImages
from .nodes.util_nodes.jlc_dynamic_multi_set_get import (
     JLC_DynamicMultiGet,
     JLC_DynamicMultiSet,
 )


NODE_CLASS_MAPPINGS = {
    # ControlNet Nodes
    "JLC_PaddedImage": JLC_PaddedImage,
    "JLC_PaddedLatent": JLC_PaddedLatent,
    "JLC_ControlNetApply": JLC_ControlNetApply,
    "JLC_ControlNetApplyAdvanced": JLC_ControlNetApplyAdvanced,
    "JLC_ControlNetComposition": JLC_ControlNetComposition,
    "JLC_ControlNetOrchestrator": JLC_ControlNetOrchestrator,
    "JLC_ControlNetOrchestratorAdvanced": JLC_ControlNetOrchestratorAdvanced,
    "JLC_DynamicAuxPreprocessorWrapper": JLC_DynamicAuxPreprocessorWrapper,

    # LoRA Loaders
    "JLC_DynamicLoraLoaderModelOnly": JLC_DynamicLoraLoaderModelOnly,
    "JLC_DynamicLoraLoaderModelClip": JLC_DynamicLoraLoaderModelClip,
    "JLC_DynamicLoraLoaderSharedBlockWeightModelOnly": JLC_DynamicLoraLoaderSharedBlockWeightModelOnly,
    "JLC_DynamicLoraLoaderSharedBlockWeightModelClip": JLC_DynamicLoraLoaderSharedBlockWeightModelClip,
    "JLC_DynamicLoraLoaderBlockWeightModelOnly": JLC_DynamicLoraLoaderBlockWeightModelOnly,
    "JLC_DynamicLoraLoaderBlockWeightModelClip": JLC_DynamicLoraLoaderBlockWeightModelClip,

    # Legacy LoRA wrappers
    #   - JLC_LoraLoaderTenStack
    #   - JLC_LoraLoaderBlockWeightTwo
    # Delete this merge if you want zero legacy nodes.
    **legacy_lora_class_mappings,

    # Utility Nodes
    "JLC_SeedGenerator": JLC_SeedGenerator,
    "JLC_StageBoundaryVRAMCleanup": JLC_StageBoundaryVRAMCleanup,
    "JLC_LoadAndResizeImage": JLC_LoadResizeEncodeImage,
    "JLC_ResizeImage": JLC_ResizeImage,
    "JLC_ResizeImages": JLC_ResizeImages,
    "JLC_DynamicMultiSet": JLC_DynamicMultiSet,
    "JLC_DynamicMultiGet": JLC_DynamicMultiGet,
}


# Keep \u2003 leading em-space in names to avoid logo overlap.
NODE_DISPLAY_NAME_MAPPINGS = {
    # ControlNet Nodes
    "JLC_PaddedImage": "\u2003JLC Padded Image",
    "JLC_PaddedLatent": "\u2003JLC Padded Latent",
    "JLC_ControlNetApply": "\u2003JLC ControlNet Apply",
    "JLC_ControlNetApplyAdvanced": "\u2003JLC ControlNet Apply (Advanced)",
    "JLC_ControlNetComposition": "\u2003JLC ControlNet Composition",
    "JLC_ControlNetOrchestrator": "\u2003JLC ControlNet Orchestrator",
    "JLC_ControlNetOrchestratorAdvanced": "\u2003JLC ControlNet Orchestrator (Advanced)",
    "JLC_DynamicAuxPreprocessorWrapper": "\u2003JLC Dynamic Aux Preprocessor Wrapper",

    # LoRA Loaders
    "JLC_DynamicLoraLoaderModelOnly": "\u2003JLC LoRA Loader - Multi Model",
    "JLC_DynamicLoraLoaderModelClip": "\u2003JLC LoRA Loader - Multi-Model / CLIP",
    "JLC_DynamicLoraLoaderSharedBlockWeightModelOnly": "\u2003JLC LoRA Loader - Multi-Model / Shared Block Weight",
    "JLC_DynamicLoraLoaderSharedBlockWeightModelClip": "\u2003JLC LoRA Loader - Multi Model / Shared Block Weight + CLIP",
    "JLC_DynamicLoraLoaderBlockWeightModelOnly": "\u2003JLC LoRA Loader - Multi-Model / Block Weight",
    "JLC_DynamicLoraLoaderBlockWeightModelClip": "\u2003JLC LoRA Loader - Multi-Model / CLIP + Block Weight",

    # Legacy LoRA wrappers. Delete this to not load legacy nodes.
    **legacy_lora_display_name_mappings,

    # Utility Nodes
    "JLC_SeedGenerator": "\u2003JLC Seed Generator",
    "JLC_StageBoundaryVRAMCleanup": "\u2003JLC Stage Boundary VRAM Cleanup",
    "JLC_LoadAndResizeImage": "\u2003JLC Load, Resize & Encode Image",
    "JLC_ResizeImage": "\u2003JLC Resize Image",
    "JLC_ResizeImages": "\u2003JLC Resize Multiple Images",
    "JLC_DynamicMultiSet": "\u2003JLC Multi Set",
    "JLC_DynamicMultiGet": "\u2003JLC Multi Get",
}


JLC_NODES_ICON = "🧠"
JLC_NODES_NAME = f"{JLC_NODES_ICON} JLC ComfyUI Nodes"
print(f"{JLC_NODES_NAME} loading...")
for node_name in sorted(NODE_CLASS_MAPPINGS.keys()):
    print(f"  {node_name}")
print(f"{JLC_NODES_NAME} loaded {len(NODE_CLASS_MAPPINGS)} nodes.")


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

# Path to web folder
WEB_DIRECTORY = "./web"
WEB_DIR = os.path.join(os.path.dirname(__file__), "web")

# Mount it into ComfyUI frontend
ps = PromptServer.instance

if os.path.exists(WEB_DIR):
    ps.app.router.add_static(
        "/extensions/JLC-ComfyUI-nodes",
        WEB_DIR,
    )
