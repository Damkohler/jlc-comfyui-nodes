"""
JLC Resize Multiple Images
--------------------------

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
  - The **JLC Resize Multiple Images** node resizes from one to five independent
    IMAGE tensors using one shared resize policy.

  - Each active image preserves its own source aspect ratio and is exposed as an
    independent resized output.

  - A convenience batch output follows ComfyUI's legacy `ImageBatch`
    normalization behavior: later active images are resized/cropped to the first
    active resized image's geometry before concatenation.

- Input and Output Contract
  - `slot_count` is authoritative and selects from one to five active IMAGE
    inputs.

  - The backend always preserves this fixed output order:
        • image_1
        • image_2
        • image_3
        • image_4
        • image_5
        • batch

  - Inactive individual outputs return None.

  - Deliberate runtime None inputs remain positionally aligned and are omitted
    from the convenience batch.

- Frontend Contract
  - The companion frontend extension presents only the requested input sockets.

  - Inactive individual output sockets are visually hidden without deleting,
    reordering, or rebuilding the fixed backend output array.

  - The fixed sixth batch socket may be drawn immediately after the final active
    individual output, while its backend index and attached graph links remain
    unchanged.

  - Resize-mode widget visibility uses the same defensive socket-positioning
    strategy as **JLC Resize Image**.

- Versioning
  - Version is governed by `JLC_UTIL_NODES_VERSION` from
    `jlc_custom_nodes_versions.py`.

- Attribution & License
  - Concept and implementation by **J. L. Córdova**
    with code assistance from **OpenAI's ChatGPT and Codex**.

  - The multi-image batch normalization behavior is aligned with ComfyUI's
    legacy `ImageBatch` node.

  - Designed for interoperability with:
    https://github.com/comfyanonymous/ComfyUI

  - Copyright (c) 2026 J. L. Córdova

  - Released under the **MIT License**.
"""

from __future__ import annotations

from typing import Optional

import comfy.utils
import torch
import torch.nn.functional as torch_functional

from ...jlc_custom_nodes_versions import JLC_UTIL_NODES_VERSION
from .jlc_load_and_resize_image import (
    _resize_image,
    calculate_target_dimensions,
)
from .jlc_resize_image import JLC_ResizeImage


MAX_IMAGE_SLOTS = 5


MANIFEST = {
    "name": "JLC Resize Multiple Images",
    "version": JLC_UTIL_NODES_VERSION,
    "author": "J. L. Córdova",
    "description": (
        "Resize up to five independent images with one shared aspect-ratio-"
        "preserving policy, return each result separately, and provide an "
        "optional normalized batch output."
    ),
}


def _validate_image(
    image: Optional[torch.Tensor],
    slot_index: int,
) -> Optional[torch.Tensor]:
    """Validate one active IMAGE slot while preserving deliberate None."""

    if image is None:
        return None
    if not isinstance(image, torch.Tensor) or image.ndim != 4:
        raise ValueError(
            "JLC Resize Multiple Images expected "
            f"image_{slot_index} to be an IMAGE "
            "tensor in BHWC layout or a deliberate runtime None."
        )
    return image


def _match_image_channels(
    first: torch.Tensor,
    second: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad the smaller BHWC channel dimension with opaque values."""

    first_channels = int(first.shape[-1])
    second_channels = int(second.shape[-1])
    if first_channels == second_channels:
        return first, second

    if first_channels < second_channels:
        first = torch_functional.pad(
            first,
            (0, second_channels - first_channels),
            mode="constant",
            value=1.0,
        )
    else:
        second = torch_functional.pad(
            second,
            (0, first_channels - second_channels),
            mode="constant",
            value=1.0,
        )
    return first, second


def _batch_resized_images(
    images: list[Optional[torch.Tensor]],
) -> Optional[torch.Tensor]:
    """Combine active results using the legacy ComfyUI ImageBatch contract."""

    batch: Optional[torch.Tensor] = None

    for image in images:
        if image is None:
            continue
        if batch is None:
            batch = image
            continue

        batch, image = _match_image_channels(batch, image)

        if image.shape[1:] != batch.shape[1:]:
            image = comfy.utils.common_upscale(
                image.movedim(-1, 1),
                int(batch.shape[2]),
                int(batch.shape[1]),
                "bilinear",
                "center",
            ).movedim(1, -1)

        batch = torch.cat((batch, image), dim=0)

    return batch


class JLC_ResizeImages:
    FUNCTION = "resize_images"
    CATEGORY = "utils/image"
    RETURN_TYPES = ("IMAGE",) * (MAX_IMAGE_SLOTS + 1)
    RETURN_NAMES = (
        "image_1",
        "image_2",
        "image_3",
        "image_4",
        "image_5",
        "batch",
    )
    OUTPUT_NODE = False

    DESCRIPTION = (
        "Resize one to five independent IMAGE tensors with one shared policy "
        "while preserving each input's aspect ratio. Individual outputs retain "
        "their own calculated dimensions. The trailing batch output follows "
        "ComfyUI ImageBatch behavior and normalizes later results to the first "
        "active result's dimensions before concatenation."
    )
    SEARCH_ALIASES = [
        "resize multiple images",
        "resize images",
        "multi image resize",
        "batch resize images",
        "jlc resize images",
        "resize five images",
        "dynamic image resize",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        # Reuse the single-image node's controls so defaults, bounds, tooltips,
        # resize modes, and interpolation choices cannot drift between nodes.
        single_inputs = JLC_ResizeImage.INPUT_TYPES()
        image_spec = single_inputs["required"]["image"]
        resize_controls = {
            name: spec
            for name, spec in single_inputs["required"].items()
            if name != "image"
        }

        return {
            "required": {
                "image_1": image_spec,
                "slot_count": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": MAX_IMAGE_SLOTS,
                        "step": 1,
                        "tooltip": (
                            "Number of active image inputs and individual "
                            "outputs. Press Update Visible Slots to apply the "
                            "frontend socket layout."
                        ),
                    },
                ),
                **resize_controls,
            },
            "optional": {
                f"image_{index}": (
                    "IMAGE",
                    {
                        "tooltip": (
                            f"Optional source image for active slot {index}."
                        ),
                    },
                )
                for index in range(2, MAX_IMAGE_SLOTS + 1)
            },
        }

    def resize_images(
        self,
        image_1,
        slot_count,
        resize_by,
        multiplier,
        longer_size,
        shorter_size,
        width,
        height,
        megapixels,
        scale_method,
        divisible_by,
        image_2=None,
        image_3=None,
        image_4=None,
        image_5=None,
    ):
        count = max(1, min(MAX_IMAGE_SLOTS, int(slot_count)))
        supplied_images = (image_1, image_2, image_3, image_4, image_5)
        resized_images: list[Optional[torch.Tensor]] = []

        for slot_index, candidate in enumerate(
            supplied_images[:count],
            start=1,
        ):
            image = _validate_image(candidate, slot_index)
            if image is None:
                resized_images.append(None)
                continue

            source_height = int(image.shape[1])
            source_width = int(image.shape[2])
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

            resized_images.append(
                _resize_image(
                    image,
                    target_width,
                    target_height,
                    scale_method,
                )
            )

        batch = _batch_resized_images(resized_images)

        # Preserve a stable six-output backend schema. The frontend changes
        # only socket presentation; it never reorders these return indices.
        individual_outputs = [*resized_images]
        individual_outputs.extend(
            [None] * (MAX_IMAGE_SLOTS - len(individual_outputs))
        )
        return (*individual_outputs, batch)
