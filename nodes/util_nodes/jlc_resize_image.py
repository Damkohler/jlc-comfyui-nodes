"""
JLC Resize Image
----------------

- JLC ComfyUI Nodes Collection
  - This node is part of the **JLC Custom Nodes for ComfyUI**
    collection developed by **J. L. Córdova**.

  - Repository
    https://github.com/Damkohler/jlc-comfyui-nodes

  - The JLC nodes focus on practical workflow improvements for image-generation
    pipelines, particularly:
        • general workflow utilities
        • Flux-based workflows
        • LoRA experimentation
        • ControlNet preparation
        • advanced inpainting / outpainting pipelines

- Node Purpose
  - The **JLC Resize Image** node is a tensor-based resize utility intended for
    mid-workflow use.

  - It accepts an IMAGE input and an optional aligned MASK input, preserves the
    source aspect ratio, and uses the same shared resize math as
    **JLC Load & Resize Image**.

  - The node intentionally has no file-loader widget and no internal preview.

- Image and Mask Contract
  - Valid IMAGE plus MASK:
        • resize both to exactly aligned output dimensions

  - Valid IMAGE without MASK:
        • resize the image
        • return an aligned all-zero MASK

  - Runtime None IMAGE:
        • pass through None for IMAGE and MASK
        • return width and height as 0

  - IMAGE tensors use ComfyUI BHWC layout.
  - MASK tensors use ComfyUI BHW layout.
  - The optional MASK is resized with the same interpolation method and target
    geometry as the IMAGE.

- Resize and Frontend Contract
  - Every resize mode preserves the source aspect ratio before the final
    `divisible_by` adjustment.

  - Frontend JavaScript exposes only the numeric widget used by the selected
    resize mode while retaining the complete backend input schema.

  - Deliberate runtime None passthrough supports dynamic workflow branches whose
    upstream nodes disable slots by returning None.

- Versioning
  - Version is governed by `JLC_UTIL_NODES_VERSION` from
    `jlc_custom_nodes_versions.py`.

- Attribution & License
  - Concept and implementation by **J. L. Córdova**
    with code assistance from **OpenAI's ChatGPT and Codex**.

  - Designed for interoperability with:
    https://github.com/comfyanonymous/ComfyUI

  - Copyright (c) 2026 J. L. Córdova

  - Released under the **MIT License**.
"""

from __future__ import annotations

import torch
from nodes import MAX_RESOLUTION

from ...jlc_custom_nodes_versions import JLC_UTIL_NODES_VERSION
from .jlc_load_resize_encode_image import (
    RESIZE_MODES,
    SCALE_METHODS,
    calculate_target_dimensions,
    _resize_image,
    _resize_mask,
)


MANIFEST = {
    "name": "JLC Resize Image",
    "version": JLC_UTIL_NODES_VERSION,
    "author": "J. L. Córdova",
    "description": (
        "Resize an incoming IMAGE tensor and optional aligned MASK with "
        "aspect-ratio-preserving math, final divisible-by alignment, and "
        "deliberate None passthrough for dynamic workflow branches."
    ),
}


class JLC_ResizeImage:
    FUNCTION = "resize_image"
    CATEGORY = "utils/image"
    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("image", "mask", "width", "height")
    OUTPUT_NODE = False

    DESCRIPTION = (
        "Resize an incoming IMAGE tensor and optional aligned MASK with "
        "aspect-ratio-preserving math and final divisible-by alignment. If the "
        "connected IMAGE evaluates to None, the node deliberately passes None "
        "through so downstream ControlNet / Reference nodes can handle "
        "disabled-slot behavior correctly."
    )
    SEARCH_ALIASES = [
        "resize image",
        "jlc resize",
        "aspect ratio resize",
        "resize image tensor",
        "resize image and mask",
        "none passthrough resize",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Incoming IMAGE tensor to resize. If the upstream "
                            "connected node deliberately outputs None at runtime, "
                            "this node passes None through."
                        ),
                    },
                ),
                "resize_by": (
                    RESIZE_MODES,
                    {
                        "default": "scale longer dimension",
                        "tooltip": (
                            "Aspect-ratio-preserving resize policy."
                        ),
                    },
                ),
                "multiplier": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.01,
                        "max": 8.0,
                        "step": 0.01,
                        "tooltip": "Scale factor; 2.0 doubles both dimensions.",
                    },
                ),
                "longer_size": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 1,
                        "max": MAX_RESOLUTION,
                        "step": 1,
                        "tooltip": "Target size of the source image's longer edge.",
                    },
                ),
                "shorter_size": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 1,
                        "max": MAX_RESOLUTION,
                        "step": 1,
                        "tooltip": "Target size of the source image's shorter edge.",
                    },
                ),
                "width": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 1,
                        "max": MAX_RESOLUTION,
                        "step": 1,
                        "tooltip": (
                            "Target width; height is calculated from aspect ratio."
                        ),
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 1,
                        "max": MAX_RESOLUTION,
                        "step": 1,
                        "tooltip": (
                            "Target height; width is calculated from aspect ratio."
                        ),
                    },
                ),
                "megapixels": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.01,
                        "max": 64.0,
                        "step": 0.01,
                        "tooltip": (
                            "Target total megapixels using 1024×1024 per megapixel."
                        ),
                    },
                ),
                "scale_method": (
                    SCALE_METHODS,
                    {
                        "default": "area",
                        "tooltip": (
                            "Interpolation method. Area is generally strong for "
                            "downscaling; Lanczos is often useful for upscaling."
                        ),
                    },
                ),
                "divisible_by": (
                    "INT",
                    {
                        "default": 16,
                        "min": 1,
                        "max": 512,
                        "step": 1,
                        "tooltip": (
                            "After aspect-ratio calculation, round width and "
                            "height down to this multiple. Use 1 to disable."
                        ),
                    },
                ),
            },
            "optional": {
                "mask": (
                    "MASK",
                    {
                        "tooltip": (
                            "Optional aligned MASK tensor to resize with the "
                            "IMAGE. When omitted or None, the node returns an "
                            "all-zero mask at the resized image dimensions."
                        ),
                    },
                ),
            },
        }

    def resize_image(
        self,
        image,
        resize_by,
        multiplier,
        longer_size,
        shorter_size,
        width,
        height,
        megapixels,
        scale_method,
        divisible_by,
        mask=None,
    ):
        # Deliberate runtime NONE passthrough for dynamic workflows.
        if image is None:
            return (None, None, 0, 0)

        if not isinstance(image, torch.Tensor) or image.ndim != 4:
            raise ValueError(
                "JLC Resize Image expected an IMAGE tensor in BHWC layout "
                "or a deliberate runtime None."
            )

        source_height = int(image.shape[1])
        source_width = int(image.shape[2])

        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                raise TypeError(
                    "JLC Resize Image expected optional mask to be a MASK "
                    "tensor in BHW layout or None."
                )
            if mask.ndim == 2:
                mask_height, mask_width = int(mask.shape[0]), int(mask.shape[1])
            elif mask.ndim == 3:
                mask_height, mask_width = int(mask.shape[1]), int(mask.shape[2])
            else:
                raise ValueError(
                    "JLC Resize Image expected optional mask in BHW layout; "
                    f"got shape {tuple(mask.shape)}."
                )
            if (mask_height, mask_width) != (source_height, source_width):
                raise ValueError(
                    "JLC Resize Image requires IMAGE and MASK source dimensions "
                    "to match before resizing. "
                    f"image={source_width}x{source_height}, "
                    f"mask={mask_width}x{mask_height}."
                )

        target_width, target_height = calculate_target_dimensions(
            source_width,
            source_height,
            resize_by,
            multiplier=multiplier,
            longer_size=longer_size,
            shorter_size=shorter_size,
            width=width,
            height=height,
            megapixels=megapixels,
            divisible_by=divisible_by,
        )

        resized_image = _resize_image(
            image,
            target_width,
            target_height,
            scale_method,
        )

        # Resize the optional incoming mask in lockstep. The shared helper
        # returns an aligned all-zero mask when no mask was supplied.
        resized_mask = _resize_mask(
            mask,
            image_batch=int(resized_image.shape[0]),
            target_width=target_width,
            target_height=target_height,
            scale_method=scale_method,
            dtype=resized_image.dtype,
        )

        return (
            resized_image,
            resized_mask,
            int(target_width),
            int(target_height),
        )
