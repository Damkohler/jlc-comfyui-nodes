"""
JLC 10 LoRA Loader Stack

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


- Node Purpose
  - The **JLC 10 LoRA Loader Stack** node applies up to ten
    LoRA models sequentially to a base model.

  - Each slot contains:
        • a selectable LoRA file
        • an independent strength value

  - Slots operate independently and are evaluated in order.
    Empty slots ("None") or strengths of zero are skipped.

  - If the same LoRA appears in multiple slots, the resulting
    effect is additive in weight space (equivalent to a single
    LoRA applied with the summed strength).


- Attribution & License
  - Concept and implementation by **J. L. Córdova**
    with development assistance from **ChatGPT (OpenAI)**.

  - Conceptual inspiration based on the **"LoRA Loader Stack"**
    node developed by **rgthree** for the ComfyUI ecosystem.

    rgthree repository:
    https://github.com/rgthree/rgthree-comfy

  - Copyright (c) 2026 J. L. Córdova

  - Released under the **MIT License**.
    This permits use, modification, and redistribution of the
    software provided that the copyright notice and license
    information are retained.
"""

import folder_paths
from nodes import LoraLoader

MANIFEST = {
    "name": "JLC 10 LoRA Loader Stack",
    "version": (1, 0, 0),
    "author": "J. L. Córdova",
    "description": "Applies up to 10 LoRAs sequentially with independent strengths."
}

class JLC_LoraLoaderTenStack:
    """
    10-slot LoRA stack loader (independent slots).

    - You can populate any subset of slots (e.g. 2 and 4 only).
    - Skips a slot if lora == "None" or strength == 0.
    - Applies in slot order (01..10).
    """

    FUNCTION = "load_lora"
    CATEGORY = "loaders"

    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")

    MAX_SLOTS = 10

    @classmethod
    def INPUT_TYPES(cls):  # pylint: disable=invalid-name
        lora_choices = ["None"] + folder_paths.get_filename_list("loras")

        required = {
            "model": ("MODEL",),
            "clip": ("CLIP",),
        }

        for i in range(1, cls.MAX_SLOTS + 1):
            suffix = f"{i:02d}"
            required[f"lora_{suffix}"] = (lora_choices,)
            required[f"strength_{suffix}"] = (
                "FLOAT",
                {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01},
            )

        return {"required": required}

    def load_lora(self, model, clip, **kwargs):
        loader = LoraLoader()

        for i in range(1, self.MAX_SLOTS + 1):
            suffix = f"{i:02d}"
            lora_name = kwargs.get(f"lora_{suffix}", "None")
            strength = float(kwargs.get(f"strength_{suffix}", 0.0))

            if lora_name != "None" and strength != 0.0:
                # Same strength applied to model + clip (typical stack behavior)
                model, clip = loader.load_lora(
                    model, clip, lora_name, strength, strength
                )

        return (model, clip)


NODE_CLASS_MAPPINGS = {
    "JLC_LoraLoaderTenStack": JLC_LoraLoaderTenStack,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JLC_LoraLoaderTenStack": "JLC 10-LoRA Loader",
}
