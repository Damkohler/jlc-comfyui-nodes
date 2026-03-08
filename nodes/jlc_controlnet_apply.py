"""
JLC ControlNet Apply

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
  - The **JLC ControlNet Apply** node applies a ControlNet to both
    positive and negative conditioning streams within a ComfyUI
    workflow.

  - The node is designed to support **daisy-chained ControlNet
    pipelines**, allowing multiple ControlNet nodes to be applied
    sequentially without unnecessary model reloads.

  - When enabled, the node attaches a ControlNet instance to each
    conditioning block while preserving previously attached
    ControlNet stacks.

  - When disabled (or when strength is set to zero), the node
    behaves as a **pass-through**, forwarding conditioning,
    ControlNet, and VAE inputs unchanged. This allows the node
    to be toggled on/off without breaking the surrounding graph.
    This is necessary because bypassing the node breaks the network
    and reroutes the positive clip to both positive and negative inputs
    of the downstream node.


- Attribution & License
  - Concept and implementation by **J. L. Córdova**
    with development assistance from **ChatGPT (OpenAI)**.

  - This node is adapted from the **ControlNetApply** node
    included in the core **ComfyUI** project.

    ComfyUI repository:
    https://github.com/comfyanonymous/ComfyUI

  - Copyright (c) 2026 J. L. Córdova

  - Released under the **MIT License**.
    This permits use, modification, and redistribution of the
    software provided that the copyright notice and license
    information are retained.
"""

import torch

MANIFEST = {
    "name": "JLC ControlNet Apply",
    "version": (1, 0, 1),
    "author": "J. L. Córdova",
    "description": "A simple mod to core ComfyUI node to minimize model reloads in multiple ControlNet workflows.",
}

class JLC_ControlNetApply:
    FUNCTION = "apply_controlnet"
    CATEGORY = "conditioning/controlnet"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": True}),
                "image": ("IMAGE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "control_net": ("CONTROL_NET",),
                "vae": ("VAE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            }
        }

    # CONTROL_NET and VAE pass thru:
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "CONTROL_NET", "VAE")
    RETURN_NAMES = ("positive", "negative", "control_net", "vae")

    def apply_controlnet(
        self,
        enabled,
        positive,
        negative,
        control_net,
        image,
        strength,
        start_percent,
        end_percent,
        vae,
        extra_concat=None,
    ):
        if (not enabled) or strength == 0:
            return (positive, negative, control_net, vae)

        if extra_concat is None:
            extra_concat = []

        control_hint = image.movedim(-1, 1)
        cnets = {}

        out = []
        for conditioning in (positive, negative):
            c = []
            for t in conditioning:
                d = t[1].copy()

                prev_cnet = d.get("control", None)
                if prev_cnet in cnets:
                    c_net = cnets[prev_cnet]
                else:
                    c_net = (
                        control_net.copy()
                        .set_cond_hint(
                            control_hint,
                            strength,
                            (start_percent, end_percent),
                            vae,
                            extra_concat=extra_concat,
                        )
                    )
                    c_net.set_previous_controlnet(prev_cnet)
                    cnets[prev_cnet] = c_net

                d["control"] = c_net
                d["control_apply_to_uncond"] = False
                c.append([t[0], d])

            out.append(c)

        # CONTROL_NET and VAE can link forward to the next call.
        return (out[0], out[1], control_net, vae)


NODE_CLASS_MAPPINGS = {
    "JLC_ControlNetApply": JLC_ControlNetApply,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JLC_ControlNetApply": "JLC ControlNet Apply",
}
