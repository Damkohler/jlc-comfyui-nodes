"""
JLC Padded Latent (Inpaint/Outpaint)

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
  - Combines the functionality of **JLC Padded Image** with
    inpaint conditioning preparation to produce a ready-to-sample
    latent for inpainting and outpainting workflows.

  - Major capabilities include:
        • Padding and rescaling an input image onto a new canvas
        • Deterministic placement using offset positioning
        • Automatic generation of a unified inpaint/outpaint mask
        • Optional union of generated mask with a user-provided mask
        • Direct creation of the inpaint-ready latent using a VAE
        • Injection of inpaint conditioning metadata required by
          compatible diffusion pipelines

  - The node returns conditioned positive and negative conditioning,
    the prepared latent, the unified mask, image dimensions, and
    optional debug outputs such as the padded image and padded flag.


- Attribution & License
  - Concept and implementation by **J. L. Córdova**
    with development assistance from **ChatGPT (OpenAI)**.

  - The padding, placement, and mask-generation logic are original
    to the **JLC Padded Image** node.

  - The inpaint conditioning behavior implemented here follows the
    field conventions used by ComfyUI's built-in
    **InpaintModelConditioning** node (e.g., concat_latent_image,
    concat_mask, noise_mask).

  - Copyright (c) 2026 J. L. Córdova

  - Released under the **MIT License**.
    This permits use, modification, and redistribution of the
    software provided that the copyright notice and license
    information are retained.
"""

import torch
import node_helpers
from .jlc_padded_image import JLC_PaddedImage

MANIFEST = {
    "name": "JLC Padded Latent (Inpaint/Outpaint)",
    "version": (1, 0, 0),
    "author": "J. L. Córdova",
    "description": (
        "Combines JLC Padded Image with inpaint conditioning. "
        "Returns conditioned positive/negative, latent, mask, dimensions, "
        "and optional debug outputs."
    ),
}

class JLC_PaddedLatent:
    def __init__(self):
        self.padded_image_node = JLC_PaddedImage()

    FUNCTION = "build_padded_latent"
    CATEGORY = "conditioning/inpaint"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "image": ("IMAGE",),
                "scaleFac": ("FLOAT", {
                    "default": 0.5, "min": 0.1, "max": 1.0, "step": 0.01,
                }),
                "maxCanvas": ("INT", {
                    "default": 1152, "min": 512, "max": 8192, "step": 8,
                }),
                "newAspectRat": (
                    [
                        "16:9", "8:5", "3:2", "4:3", "1:1",
                        "3:4", "2:3", "5:8", "9:16",
                    ],
                    {"default": "3:4"},
                ),
                "offsetX": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                }),
                "offsetY": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                }),
                "feathering": ("INT", {
                    "default": 8, "min": 0, "max": 256, "step": 1,
                }),
                "seamFixPx": ("INT", {
                    "default": 8, "min": 0, "max": 64, "step": 1,
                }),
                "noise_mask": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Attach noise_mask to the latent so sampling happens "
                        "primarily inside the mask. May help or hurt depending "
                        "on model/workflow."
                    ),
                }),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = (
        "CONDITIONING",
        "CONDITIONING",
        "LATENT",
        "MASK",
        "INT",
        "INT",
        "IMAGE",
        "BOOLEAN",
    )

    RETURN_NAMES = (
        "positive",
        "negative",
        "latent",
        "mask",
        "width",
        "height",
        "padded_image",
        "padded?",
    )

    def build_padded_latent(
        self,
        positive,
        negative,
        vae,
        image,
        scaleFac,
        maxCanvas,
        newAspectRat,
        offsetX,
        offsetY,
        feathering,
        seamFixPx,
        noise_mask=True,
        mask=None,
    ):
        padded_image, padded_mask, width, height, padded_q = (
            self.padded_image_node.scale_pad_offset_img(
                image=image,
                scaleFac=scaleFac,
                maxCanvas=maxCanvas,
                newAspectRat=newAspectRat,
                offsetX=offsetX,
                offsetY=offsetY,
                feathering=feathering,
                seamFixPx=seamFixPx,
                mask=mask,
            )
        )

        positive_out, negative_out, latent_out = self.encode_inpaint(
            positive=positive,
            negative=negative,
            pixels=padded_image,
            vae=vae,
            mask=padded_mask,
            noise_mask=noise_mask,
        )

        return (
            positive_out,
            negative_out,
            latent_out,
            padded_mask,
            width,
            height,
            padded_image,
            padded_q,
        )

    def encode_inpaint(
        self,
        positive,
        negative,
        pixels,
        vae,
        mask,
        noise_mask=True,
    ):
        """
        Adapted from ComfyUI's InpaintModelConditioning behavior.

        - Builds concat_latent_image from a masked/neutralized copy of pixels
        - Builds output latent from original pixels
        - Adds concat_latent_image + concat_mask to both conditionings
        - Optionally adds noise_mask to latent dict
        """
        x = (pixels.shape[1] // 8) * 8
        y = (pixels.shape[2] // 8) * 8

        mask = torch.nn.functional.interpolate(
            mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])),
            size=(pixels.shape[1], pixels.shape[2]),
            mode="bilinear",
        )

        orig_pixels = pixels
        pixels = orig_pixels.clone()

        if pixels.shape[1] != x or pixels.shape[2] != y:
            x_offset = (pixels.shape[1] % 8) // 2
            y_offset = (pixels.shape[2] % 8) // 2
            pixels = pixels[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
            mask = mask[:, :, x_offset:x + x_offset, y_offset:y + y_offset]

        m = (1.0 - mask.round()).squeeze(1)

        for i in range(3):
            pixels[:, :, :, i] -= 0.5
            pixels[:, :, :, i] *= m
            pixels[:, :, :, i] += 0.5

        concat_latent = vae.encode(pixels)
        orig_latent = vae.encode(orig_pixels)

        out_latent = {
            "samples": orig_latent,
        }

        if noise_mask:
            out_latent["noise_mask"] = mask

        out = []
        for conditioning in [positive, negative]:
            c = node_helpers.conditioning_set_values(
                conditioning,
                {
                    "concat_latent_image": concat_latent,
                    "concat_mask": mask,
                },
            )
            out.append(c)

        return (out[0], out[1], out_latent)


NODE_CLASS_MAPPINGS = {
    "JLC_PaddedLatent": JLC_PaddedLatent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JLC_PaddedLatent": "JLC Padded Latent",
}