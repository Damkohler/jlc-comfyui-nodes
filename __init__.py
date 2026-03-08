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

# __init__.py  (JLC-Tests-comfyui)

from .nodes.jlc_padded_image import JLC_PaddedImage
from .nodes.jlc_padded_latent import JLC_PaddedLatent
from .nodes.jlc_controlnet_apply import JLC_ControlNetApply
from .nodes.jlc_lora_loader_ten_stack import JLC_LoraLoaderTenStack
from .nodes.jlc_lora_loader_block_weight_two import JLC_LoraLoaderBlockWeightTwo

NODE_CLASS_MAPPINGS = {
    "JLC_PaddedImage": JLC_PaddedImage,
    "JLC_PaddedLatent": JLC_PaddedLatent,
    "JLC_ControlNetApply": JLC_ControlNetApply,
    "JLC_LoraLoaderTenStack": JLC_LoraLoaderTenStack,
    "JLC_LoraLoaderBlockWeightTwo": JLC_LoraLoaderBlockWeightTwo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JLC_PaddedImage": "JLC Padded Image",
    "JLC_PaddedLatent": "JLC Padded Latent",
    "JLC_ControlNetApply": "JLC ControlNet Apply",
    "JLC_LoraLoaderTenStack": "JLC 10-LoRA Loader",
    "JLC_LoraLoaderBlockWeightTwo": "JLC 2-LoRA Loader - Block Weight",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]