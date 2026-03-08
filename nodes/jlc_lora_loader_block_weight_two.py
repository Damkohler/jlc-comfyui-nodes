"""
JLC - 2 LoRA Loader (Block Weight)

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
  - The **JLC 2 LoRA Loader (Block Weight)** node applies
    up to two LoRA models sequentially while allowing independent
    block-weight vectors for each LoRA.

  - Each slot contains:
        • a selectable LoRA file
        • independent model and CLIP strengths
        • an independent block-weight vector

  - LoRAs are applied sequentially in order, allowing complementary
    LoRAs to be combined while controlling which UNet blocks are
    affected by each.


- Attribution & License
  - This node is adapted from the **LoRA Loader (Block Weight)**
    implementation in the **ComfyUI Inspire Pack** project.

  - Original project
    https://github.com/ltdrdata/ComfyUI-Inspire-Pack

  - The original node was developed by the Inspire Pack contributors
    and released under the **MIT License**.

  - This version modifies and extends the original implementation by:
        • simplifying the interface
        • removing experimental controls (inverse, seed,
          presets, A/B blending, and control-after-generate)
        • adding support for multiple sequential LoRA slots
        • adding independent block-weight vectors per slot
        • integrating with the JLC node collection

  - Special thanks to the Inspire Pack developers for their
    contributions to the ComfyUI ecosystem.

  - Concept modification and implementation by **J. L. Córdova**
    with development assistance from **ChatGPT (OpenAI)**.
  
  - Copyright (c) 2026 J. L. Córdova

  - Released under the **MIT License**.
    This permits use, modification, and redistribution of the
    software provided that the copyright notice and license
    information are retained.
"""

import comfy.lora
import comfy.utils
import folder_paths

MANIFEST = {
    "name": "JLC - 2 LoRA Loader (Block Weight)",
    "version": (1, 0, 0),
    "author": "J. L. Córdova",
    "description": (
        "Applies up to two LoRAs sequentially with independent "
        "model strength, CLIP strength, and block-weight vectors. "
        "Adapted from the Inspire Pack LoRA Block Weight loader."
    ),
}

def _parse_vector_csv(csv_text):
    """
    Numeric-only CSV parser.
    Accepts: "1,0,0,0,1,1,1"
    Returns: list[float]
    """
    if csv_text is None:
        raise ValueError("block_vector is required")

    parts = [p.strip() for p in str(csv_text).strip().split(",") if p.strip() != ""]
    if not parts:
        raise ValueError("block_vector cannot be empty")

    vec = []
    for p in parts:
        try:
            vec.append(float(p))
        except ValueError as exc:
            raise ValueError(
                f"block_vector must be numeric CSV only. Bad token: '{p}'"
            ) from exc

    return vec


def _parse_unet_num(two_chars):
    # Matches InspirePack behavior: "0." -> 0, "12" -> 12
    if len(two_chars) >= 2 and two_chars[1] == ".":
        return int(two_chars[0])
    return int(two_chars)


class JLC_LoraLoaderBlockWeightTwo:
    """
    Two-slot block-weight LoRA loader.

    Slot is skipped if lora == "None" OR both strengths == 0.
    Applies in order: slot_01 then slot_02.
    """

    FUNCTION = "load_loras"
    CATEGORY = "loaders"

    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")

    def __init__(self):
        # Simple cache to avoid reloading the same LoRA each run
        self._loaded = {}  # path -> state_dict

    @classmethod
    def INPUT_TYPES(cls):  # pylint: disable=invalid-name
        lora_choices = ["None"] + folder_paths.get_filename_list("loras")

        required = {
            "model": ("MODEL",),
            "clip": ("CLIP",),

            # Slot 01
            "lora_01": (lora_choices,),
            "strength_model_01": (
                "FLOAT",
                {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01},
            ),
            "strength_clip_01": (
                "FLOAT",
                {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01},
            ),
            "block_vector_01": (
                "STRING",
                {
                    "multiline": False,
                    "default": "1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1",
                },
            ),

            # Slot 02
            "lora_02": (lora_choices,),
            "strength_model_02": (
                "FLOAT",
                {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01},
            ),
            "strength_clip_02": (
                "FLOAT",
                {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01},
            ),
            "block_vector_02": (
                "STRING",
                {
                    "multiline": False,
                    "default": "1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1",
                },
            ),
        }

        return {"required": required}

    def _load_lora_state(self, lora_name):
        lora_path = folder_paths.get_full_path("loras", lora_name)
        if lora_path is None:
            raise ValueError(f"Could not resolve LoRA path for '{lora_name}'")

        if lora_path in self._loaded:
            return self._loaded[lora_path]

        state = comfy.utils.load_torch_file(lora_path, safe_load=True)
        self._loaded[lora_path] = state
        return state

    @staticmethod
    def _compute_block_weights(model, clip, lora_state, vector):
        """
        Reimplements InspirePack LBW's core idea but numeric-only & simplified:
        - vector[0] is base ratio for "others"
        - subsequent entries are consumed per distinct block index encountered
          in the ordered traversal:
          input_blocks -> middle_blocks -> output_blocks -> double_blocks -> single_blocks
        - if vector runs out, reuse last ratio
        """
        key_map = comfy.lora.model_lora_keys_unet(model.model)
        key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, key_map)
        loaded = comfy.lora.load_lora(lora_state, key_map)

        # Sort keys into groups like InspirePack does
        input_blocks = []
        middle_blocks = []
        output_blocks = []
        double_blocks = []
        single_blocks = []
        others = []

        for key, weights in loaded.items():
            k = key[0] if isinstance(key, tuple) else key
            k_unet = k[len("diffusion_model."):] if k.startswith("diffusion_model.") else k

            if k_unet.startswith("input_blocks."):
                num = k_unet[len("input_blocks."):len("input_blocks.") + 2]
                input_blocks.append((k, weights, _parse_unet_num(num)))
            elif k_unet.startswith("middle_block."):
                num = k_unet[len("middle_block."):len("middle_block.") + 2]
                middle_blocks.append((k, weights, _parse_unet_num(num)))
            elif k_unet.startswith("output_blocks."):
                num = k_unet[len("output_blocks."):len("output_blocks.") + 2]
                output_blocks.append((k, weights, _parse_unet_num(num)))
            elif k_unet.startswith("double_blocks."):
                num = k_unet[len("double_blocks."):len("double_blocks.") + 2]
                double_blocks.append((key, weights, _parse_unet_num(num)))
            elif k_unet.startswith("single_blocks."):
                num = k_unet[len("single_blocks."):len("single_blocks.") + 2]
                single_blocks.append((key, weights, _parse_unet_num(num)))
            else:
                others.append((k, weights))

        input_blocks.sort(key=lambda x: x[2])
        middle_blocks.sort(key=lambda x: x[2])
        output_blocks.sort(key=lambda x: x[2])
        double_blocks.sort(key=lambda x: x[2])
        single_blocks.sort(key=lambda x: x[2])

        # Base ratio for "others"
        base_ratio = vector[0]
        if len(vector) == 1:
            # if only base provided, use base everywhere
            vector = [base_ratio, base_ratio]

        block_weights = []  # list of (k, weights, ratio)

        # Apply base ratio to "others"
        if base_ratio != 0:
            for k, weights in others:
                block_weights.append((k, weights, base_ratio))

        # Walk blocks in the prescribed order and consume vector entries per block index
        vector_i = 1
        last_block_num = None
        current_ratio = vector[vector_i] if vector_i < len(vector) else vector[-1]

        for k, weights, block_num in (
            input_blocks + middle_blocks + output_blocks + double_blocks + single_blocks
        ):
            if last_block_num != block_num:
                # consume next vector entry (or reuse last)
                if vector_i < len(vector):
                    current_ratio = vector[vector_i]
                    vector_i += 1
                else:
                    current_ratio = vector[-1]

            last_block_num = block_num

            if current_ratio != 0:
                block_weights.append((k, weights, current_ratio))

        return block_weights

    @staticmethod
    def _apply_block_weights(model, clip, block_weights, strength_model, strength_clip):
        new_model = model.clone()
        new_clip = clip.clone()

        for k, weights, ratio in block_weights:
            if ratio == 0:
                continue

            # Match InspirePack behavior: "text"/"encoder" keys go to clip
            if "text" in k or "encoder" in k:
                if strength_clip != 0.0:
                    new_clip.add_patches({k: weights}, float(strength_clip) * float(ratio))
            else:
                if strength_model != 0.0:
                    new_model.add_patches({k: weights}, float(strength_model) * float(ratio))

        return new_model, new_clip

    def _apply_slot(self, model, clip, lora_name, strength_model, strength_clip, block_vector):
        if lora_name == "None":
            return model, clip
        if float(strength_model) == 0.0 and float(strength_clip) == 0.0:
            return model, clip

        vec = _parse_vector_csv(block_vector)
        lora_state = self._load_lora_state(lora_name)
        block_weights = self._compute_block_weights(model, clip, lora_state, vec)
        return self._apply_block_weights(model, clip, block_weights, strength_model, strength_clip)

    def load_loras(
        self,
        model,
        clip,
        lora_01,
        strength_model_01,
        strength_clip_01,
        block_vector_01,
        lora_02,
        strength_model_02,
        strength_clip_02,
        block_vector_02,
    ):
        # Apply slot 01 then slot 02, stack-style
        model, clip = self._apply_slot(
            model, clip, lora_01, strength_model_01, strength_clip_01, block_vector_01
        )
        model, clip = self._apply_slot(
            model, clip, lora_02, strength_model_02, strength_clip_02, block_vector_02
        )
        return (model, clip)


NODE_CLASS_MAPPINGS = {
    "JLC_LoraLoaderBlockWeightTwo": JLC_LoraLoaderBlockWeightTwo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JLC_LoraLoaderBlockWeightTwo": "JLC 2-LoRA Loader - Block Weight",
}