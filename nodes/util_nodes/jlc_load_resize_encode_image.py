"""
JLC Load, Resize & Encode Image
-------------------------------

- JLC ComfyUI Nodes Collection
  - This node is part of the **JLC Custom Nodes for ComfyUI**
    collection developed by **J. L. Córdova**.

  - Repository
    https://github.com/Damkohler/jlc-comfyui-nodes

  - The JLC nodes focus on practical workflow improvements for
    image-generation pipelines, particularly:
        • general workflow utilities
        • Flux-based workflows
        • LoRA experimentation
        • ControlNet preparation
        • advanced inpainting / outpainting pipelines

- Node Purpose
  - The **JLC Load, Resize & Encode Image** node combines:
        • ComfyUI's native upload / drag-and-drop image-loader contract
        • aspect-ratio-preserving resize math aligned with ComfyUI's
          `Resize Image/Mask` node
        • aligned IMAGE and MASK resizing
        • explicit final dimension divisibility
        • optional VAE encoding of the final resized IMAGE

  - The VAE input is optional:
        • with no VAE connected, the node behaves as a load-and-resize utility
          and returns `None` on the LATENT output
        • with a VAE connected, the resized IMAGE is encoded through the
          supplied VAE and returned using ComfyUI's standard LATENT contract

  - Existing output ordering is intentionally preserved for workflow
    compatibility. LATENT is appended after the original IMAGE, MASK, width,
    and height outputs.

  - The implementation class and filename expose the expanded encoding
    capability, while the existing registered ComfyUI node identifier
    `JLC_LoadAndResizeImage` should be retained in `NODE_CLASS_MAPPINGS` so
    existing workflows continue to resolve without migration.

  - The node intentionally excludes freeform independent width/height resizing.
    Every exposed resize mode preserves the source aspect ratio before the final
    `divisible_by` adjustment.

  - Supported resize modes:
        • scale by multiplier
        • scale longer dimension
        • scale shorter dimension
        • scale width
        • scale height
        • scale total pixels

  - Frontend JavaScript shows only the numeric widget relevant to the selected
    resize mode. Hidden values remain serialized but are ignored by the backend
    unless their mode is selected again.

- Divisibility Policy
  - The requested aspect-ratio-preserving dimensions are calculated first.
  - Width and height are then rounded down independently to the nearest
    `divisible_by` multiple.
  - This avoids silently enlarging normal images and produces dimensions suitable
    for latent-space and model-alignment requirements.
  - If a calculated edge is smaller than the requested divisor, that edge is
    clamped to one divisor so the output remains valid and divisible.

- Image, Mask, and Latent Contract
  - IMAGE output uses ComfyUI BHWC layout.
  - MASK output uses ComfyUI BHW layout.
  - The alpha-derived mask from the image loader is resized to exactly the same
    output width and height as the image.
  - Images without alpha still return a correctly sized all-zero mask.
  - LATENT is the final output so all original output indices remain unchanged.
  - When VAE is connected, LATENT follows ComfyUI's standard
    `{"samples": tensor}` representation.
  - When VAE is not connected, LATENT is `None`.

- Upstream Inspiration and Attribution
  - The compact load-and-resize workflow concept is inspired by the practical
    convenience of KJNodes' `Load & Resize Image` node.

  - Resize behavior is independently implemented using ComfyUI's public image
    loading and `common_upscale` interfaces and the aspect-ratio math used by the
    native `Resize Image/Mask` implementation.

  - Optional VAE encoding follows ComfyUI's native VAE Encode contract by
    calling `vae.encode(pixels)` on the final resized IMAGE and wrapping the
    encoded tensor as a LATENT samples dictionary.

  - This node does not import or require KJNodes.

- Versioning
  - Version is governed by `JLC_UTIL_NODES_VERSION` from
    `jlc_custom_nodes_versions.py`.

- Attribution & License
  - Concept and implementation by **J. L. Córdova**
    with development assistance from **ChatGPT (OpenAI)**.

  - Designed for interoperability with:
    https://github.com/comfyanonymous/ComfyUI

  - Copyright (c) 2026 J. L. Córdova

  - Released under the **MIT License**.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch

import comfy.utils
from nodes import LoadImage, MAX_RESOLUTION

from ...jlc_custom_nodes_versions import JLC_UTIL_NODES_VERSION


MANIFEST = {
    "name": "JLC Load, Resize & Encode Image",
    "version": JLC_UTIL_NODES_VERSION,
    "author": "J. L. Córdova",
    "description": (
        "Upload- and drag-and-drop-capable image loader with aligned IMAGE and "
        "MASK outputs, aspect-ratio-preserving resize modes modeled after "
        "ComfyUI's native Resize Image/Mask math, dynamic frontend visibility "
        "for mode-specific controls, final width/height rounding to a "
        "user-selected divisible-by value, and optional VAE encoding of the "
        "resized IMAGE to a LATENT output."
    ),
}


RESIZE_MODES = (
    "scale by multiplier",
    "scale longer dimension",
    "scale shorter dimension",
    "scale width",
    "scale height",
    "scale total pixels",
)

SCALE_METHODS = (
    "nearest-exact",
    "bilinear",
    "area",
    "bicubic",
    "lanczos",
)


def _positive_int(value: int | float, name: str) -> int:
    result = int(round(float(value)))
    if result < 1:
        raise ValueError(f"{name} must resolve to at least 1 pixel; got {value!r}.")
    return result


def _round_down_to_multiple(value: int, multiple: int) -> int:
    """Round down while keeping a valid positive divisible dimension."""

    value = max(1, int(value))
    multiple = max(1, int(multiple))
    if multiple == 1:
        return value
    return max(multiple, (value // multiple) * multiple)


def calculate_target_dimensions(
    source_width: int,
    source_height: int,
    resize_by: str,
    *,
    multiplier: float,
    longer_size: int,
    shorter_size: int,
    width: int,
    height: int,
    megapixels: float,
    divisible_by: int,
) -> Tuple[int, int]:
    """Calculate aspect-preserving output dimensions, then apply divisibility."""

    source_width = _positive_int(source_width, "source_width")
    source_height = _positive_int(source_height, "source_height")

    if resize_by == "scale by multiplier":
        if float(multiplier) <= 0:
            raise ValueError("multiplier must be greater than zero.")
        target_width = round(source_width * float(multiplier))
        target_height = round(source_height * float(multiplier))

    elif resize_by == "scale longer dimension":
        requested = _positive_int(longer_size, "longer_size")
        scale = requested / max(source_width, source_height)
        target_width = round(source_width * scale)
        target_height = round(source_height * scale)

    elif resize_by == "scale shorter dimension":
        requested = _positive_int(shorter_size, "shorter_size")
        scale = requested / min(source_width, source_height)
        target_width = round(source_width * scale)
        target_height = round(source_height * scale)

    elif resize_by == "scale width":
        requested = _positive_int(width, "width")
        scale = requested / source_width
        target_width = requested
        target_height = round(source_height * scale)

    elif resize_by == "scale height":
        requested = _positive_int(height, "height")
        scale = requested / source_height
        target_width = round(source_width * scale)
        target_height = requested

    elif resize_by == "scale total pixels":
        if float(megapixels) <= 0:
            raise ValueError("megapixels must be greater than zero.")
        target_pixels = float(megapixels) * 1024.0 * 1024.0
        scale = math.sqrt(target_pixels / (source_width * source_height))
        target_width = round(source_width * scale)
        target_height = round(source_height * scale)

    else:
        raise ValueError(f"Unsupported resize mode: {resize_by!r}.")

    target_width = _positive_int(target_width, "target_width")
    target_height = _positive_int(target_height, "target_height")

    divisor = max(1, int(divisible_by))
    target_width = _round_down_to_multiple(target_width, divisor)
    target_height = _round_down_to_multiple(target_height, divisor)

    if target_width > MAX_RESOLUTION or target_height > MAX_RESOLUTION:
        raise ValueError(
            "Calculated output exceeds ComfyUI MAX_RESOLUTION: "
            f"{target_width}x{target_height}, maximum={MAX_RESOLUTION}."
        )

    return target_width, target_height


def _resize_image(
    image: torch.Tensor,
    target_width: int,
    target_height: int,
    scale_method: str,
) -> torch.Tensor:
    if not isinstance(image, torch.Tensor) or image.ndim != 4:
        raise ValueError(
            "JLC Load, Resize & Encode Image expected an IMAGE tensor "
            "in BHWC layout."
        )

    image_bchw = image.movedim(-1, 1)
    resized = comfy.utils.common_upscale(
        image_bchw,
        target_width,
        target_height,
        scale_method,
        "disabled",
    )
    return resized.movedim(1, -1).contiguous()


def _resize_mask(
    mask: torch.Tensor | None,
    *,
    image_batch: int,
    target_width: int,
    target_height: int,
    scale_method: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    if mask is None:
        return torch.zeros(
            (image_batch, target_height, target_width),
            dtype=dtype,
            device="cpu",
        )

    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"Expected MASK tensor, got {type(mask)!r}.")

    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3:
        raise ValueError(
            "JLC Load, Resize & Encode Image expected a MASK tensor "
            "in BHW layout."
        )

    if mask.shape[0] == 1 and image_batch > 1:
        mask = mask.repeat(image_batch, 1, 1)
    elif mask.shape[0] != image_batch:
        raise ValueError(
            "Loaded IMAGE and MASK batch sizes do not match: "
            f"image_batch={image_batch}, mask_batch={mask.shape[0]}."
        )

    resized = comfy.utils.common_upscale(
        mask.unsqueeze(1),
        target_width,
        target_height,
        scale_method,
        "disabled",
    )
    return resized.squeeze(1).contiguous().clamp_(0.0, 1.0)


def _encode_image(vae, image: torch.Tensor):
    """Encode IMAGE through a supplied ComfyUI VAE into LATENT format."""

    if vae is None:
        return None

    encoded = vae.encode(image)
    return {"samples": encoded}


class JLC_LoadResizeEncodeImage(LoadImage):
    """Load and resize IMAGE/MASK, with optional VAE encoding to LATENT."""

    FUNCTION = "load_resize_encode"
    CATEGORY = "utils/image"
    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "LATENT")
    RETURN_NAMES = ("image", "mask", "width", "height", "latent")
    OUTPUT_NODE = False

    DESCRIPTION = (
        "Upload or drag-and-drop an image, resize IMAGE and MASK together with "
        "aspect-ratio-preserving math and final divisible-by alignment, and "
        "optionally encode the resized IMAGE when a VAE is connected."
    )
    SEARCH_ALIASES = [
        "load resize image",
        "load resize encode",
        "load resize latent",
        "encode image",
        "image to latent",
        "image loader resize",
        "drag drop resize",
        "aspect ratio resize",
        "resize image and mask",
        "vae encode",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        # Reuse the active ComfyUI LoadImage input specification so upload,
        # dropdown, annotated-path, and drag-and-drop behavior remain aligned
        # with the installed ComfyUI version.
        load_inputs = LoadImage.INPUT_TYPES()
        image_spec = load_inputs["required"]["image"]

        return {
            "required": {
                "image": image_spec,
                "resize_by": (
                    RESIZE_MODES,
                    {
                        "default": "scale longer dimension",
                        "tooltip": (
                            "Aspect-ratio-preserving resize policy. The frontend "
                            "shows only the numeric control used by this mode."
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
                        "tooltip": (
                            "Target size of the source image's longer edge."
                        ),
                    },
                ),
                "shorter_size": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 1,
                        "max": MAX_RESOLUTION,
                        "step": 1,
                        "tooltip": (
                            "Target size of the source image's shorter edge."
                        ),
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
                "vae": (
                    "VAE",
                    {
                        "tooltip": (
                            "Optional. When connected, encodes the final resized "
                            "IMAGE and returns it through the LATENT output. "
                            "Without a VAE, LATENT returns None."
                        ),
                    },
                ),
            },
        }

    @classmethod
    def IS_CHANGED(cls, image, **kwargs):
        """Preserve LoadImage file hashing while accepting added inputs.

        ComfyUI includes this node's full input dictionary when checking for
        external changes. The inherited LoadImage.IS_CHANGED method accepts
        only `image`, so resize controls and the optional VAE are absorbed here.
        Normal prompt-input caching still tracks those inputs independently.
        """

        return LoadImage.IS_CHANGED(image)

    @classmethod
    def VALIDATE_INPUTS(cls, image, **kwargs):
        """Delegate image-path validation while accepting added inputs."""

        return LoadImage.VALIDATE_INPUTS(image)

    def load_resize_encode(
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
        vae=None,
    ):
        loaded = super().load_image(image)

        # Current ComfyUI LoadImage returns a two-item tuple. The small fallback
        # keeps this node tolerant of a future result-wrapping migration.
        if isinstance(loaded, dict):
            loaded = loaded.get("result", loaded)
        if hasattr(loaded, "result"):
            loaded = loaded.result

        if not isinstance(loaded, (tuple, list)) or len(loaded) < 2:
            raise RuntimeError(
                "ComfyUI LoadImage returned an unsupported result structure."
            )

        source_image, source_mask = loaded[0], loaded[1]

        source_height = int(source_image.shape[1])
        source_width = int(source_image.shape[2])

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
            source_image,
            target_width,
            target_height,
            scale_method,
        )
        resized_mask = _resize_mask(
            source_mask,
            image_batch=int(resized_image.shape[0]),
            target_width=target_width,
            target_height=target_height,
            scale_method=scale_method,
            dtype=resized_image.dtype,
        )

        latent = _encode_image(vae, resized_image)

        return (
            resized_image,
            resized_mask,
            int(target_width),
            int(target_height),
            latent,
        )
