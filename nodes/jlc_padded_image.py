"""
JLC Padded Image (Inpaint / Outpaint Canvas Builder)

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
  - The **JLC Padded Image (Inpaint / Outpaint Canvas Builder)**
    node prepares an input image for inpainting or outpainting 
    workflows by placing a scaled version of the image onto a
    new canvas while generating a corresponding editable mask.

  - The node allows the user to define:
        • canvas size and aspect ratio
        • scaling of the original image
        • horizontal and vertical placement offsets
        • feathering of mask boundaries

  - The padded regions of the canvas are automatically marked
    as editable areas in the generated mask.

  - An optional manual mask may also be supplied, which is
    aligned with the scaled image region and merged with the
    automatically generated padding mask.

  - The node outputs a padded image and a mask aligned with
    the canvas, enabling controlled image editing or expansion
    in downstream nodes.


- Attribution & License
  - Concept and implementation by **J. L. Córdova**
    with development assistance from **ChatGPT (OpenAI)**.

  - This node concept and implementation are original to the
    JLC ComfyUI Nodes collection.

  - Copyright (c) 2026 J. L. Córdova

  - Released under the **MIT License**.
    This permits use, modification, and redistribution of the
    software provided that the copyright notice and license
    information are retained.
"""

import PIL.Image as Image
from PIL import ImageFilter
import numpy as np
import torch

MANIFEST = {
    "name": "JLC Padded Image (Inpaint / Outpaint Canvas Builder)",
    "version": (1, 0, 0),
    "author": "J. L. Córdova",
    "description": (
        "Creates a padded canvas and aligned mask for inpainting and "
        "outpainting workflows. Supports image scaling, offset placement, "
        "aspect ratio control, feathered mask edges, and optional manual "
        "mask integration."
    ),
}

class JLC_PaddedImage:
    def __init__(self):
        pass

    FUNCTION = "scale_pad_offset_img"
    CATEGORY = "image"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
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
            },
            # Optional incoming mask to union with the generated padded/outpaint mask.
            "optional": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "BOOLEAN")
    RETURN_NAMES = ("Padded Image", "Mask", "Width", "Height", "Padded?")

    def scale_pad_offset_img(
        self,
        image,
        scaleFac,
        maxCanvas,
        newAspectRat,
        offsetX,
        offsetY,
        feathering,
        seamFixPx, 
        mask=None,
    ):
        # Convert image tensor -> PIL RGB
        image = self.tensor2pil(image)
        width, height = image.size
        oldAspectRat = height / width

        aspect_ratios = {
            "16:9": 16 / 9,
            "8:5": 8 / 5,
            "3:2": 3 / 2,
            "4:3": 4 / 3,
            "1:1": 1.0,
            "3:4": 3 / 4,
            "2:3": 2 / 3,
            "5:8": 5 / 8,
            "9:16": 9 / 16,
        }
        newAspectRat = aspect_ratios[newAspectRat]

        # Define the new canvas based on maxCanvas and newAspectRat
        if newAspectRat > 1:
            new_dim_x, new_dim_y = int(maxCanvas / newAspectRat), maxCanvas
        else:
            new_dim_x, new_dim_y = maxCanvas, int(newAspectRat * maxCanvas)

        new_dim_x -= new_dim_x % 8
        new_dim_y -= new_dim_y % 8

        # Rescale the input image according to scaleFac to fit entirely inside the new canvas
        if newAspectRat > 1 and oldAspectRat >= 1:
            if newAspectRat > oldAspectRat:
                scaled_dim_x, scaled_dim_y = 1 / newAspectRat, oldAspectRat / newAspectRat
            else:
                scaled_dim_x, scaled_dim_y = 1 / oldAspectRat, 1

        elif newAspectRat >= 1 and oldAspectRat < 1:
            scaled_dim_x, scaled_dim_y = 1 / newAspectRat, oldAspectRat / newAspectRat

        elif newAspectRat <= 1 and oldAspectRat > 1:
            scaled_dim_x, scaled_dim_y = newAspectRat / oldAspectRat, newAspectRat

        elif newAspectRat < 1 and oldAspectRat <= 1:
            if newAspectRat < oldAspectRat:
                scaled_dim_x, scaled_dim_y = newAspectRat / oldAspectRat, newAspectRat
            else:
                scaled_dim_x, scaled_dim_y = 1, oldAspectRat
        else:
            scaled_dim_x, scaled_dim_y = 1, 1

        scaled_dim_x, scaled_dim_y = np.round(
            scaleFac * maxCanvas * np.array([scaled_dim_x, scaled_dim_y])
        ).astype(int).tolist()
        scaled_dim_x -= scaled_dim_x % 8
        scaled_dim_y -= scaled_dim_y % 8

        image_scaled = image.resize((scaled_dim_x, scaled_dim_y), Image.LANCZOS)

        # Define the padding and displacement from the middle of canvas set by offsetX and offsetY
        delta_x, delta_y = new_dim_x - scaled_dim_x, new_dim_y - scaled_dim_y
        offstX = int(offsetX * delta_x)
        offstY = int(offsetY * delta_y)
        offsets_x = (offstX, new_dim_x - scaled_dim_x - offstX)
        offsets_y = (offstY, new_dim_y - scaled_dim_y - offstY)

        paddedQ = True
        if all(v == 0 for v in offsets_x + offsets_y):
            paddedQ = False

        # Pad the image (white canvas)
        image_canvas = np.ones((new_dim_y, new_dim_x, 3), dtype=np.uint8) * 255
        image_canvas = Image.fromarray(image_canvas, mode="RGB")
        image_canvas.paste(image_scaled, (offsets_x[0], offsets_y[0]))

        # Generate the padded/outpaint mask (white=paint, black=keep)
        padded_mask = self.create_mask(
            scaled_dim_x, scaled_dim_y, offsets_x, offsets_y, feathering
        )

        # Optional: align incoming manual mask to the same scaled+offset canvas,
        # then union (max) with padded_mask.
        if mask is not None:
            manual_mask_aligned = self.align_manual_mask_to_canvas(
                mask=mask,
                scaled_dim_x=scaled_dim_x,
                scaled_dim_y=scaled_dim_y,
                new_dim_x=new_dim_x,
                new_dim_y=new_dim_y,
                paste_x=offsets_x[0],
                paste_y=offsets_y[0],
            )

            if manual_mask_aligned is not None:
                padded_mask = self.union_masks(padded_mask, manual_mask_aligned)

        # Seam fix for noise_mask=True workflows.
        # Grow the paint region slightly so the bilinear/rounding transition band
        # falls inside the paint area and gets repainted (removes border lines).
        if seamFixPx and seamFixPx > 0:
            padded_mask = self.grow_mask_white(padded_mask, seamFixPx)

        # Convert outputs back to tensors
        image_padded = self.pil2tensor(image_canvas)
        mask_out = self.pil2masktensor(padded_mask)

        return (
            image_padded.to(dtype=torch.float16),
            mask_out.to(dtype=torch.float16),
            int(new_dim_x),
            int(new_dim_y),
            paddedQ,
        )

    def create_mask(self, scaled_dim_x, scaled_dim_y, offsets_x, offsets_y, feathering):
        msk = np.zeros((scaled_dim_x, scaled_dim_y), dtype=np.float16)

        if 0 < feathering < scaled_dim_x / 2 and feathering < scaled_dim_y / 2:
            msk = msk[feathering-1:-(feathering-1), feathering-1:-(feathering-1)]

            rng = np.arange(1.0 / feathering,
                            1.0 - 1.0 / feathering,
                            1.0 / feathering)
            for val in rng:
                msk = np.pad(msk, ((1, 1), (1, 1)),
                            mode='constant', constant_values=val)

            # NEW: force back to original scaled size (fixes the -2 pixel issue)
            msk = self._fit_to_shape(msk, scaled_dim_x, scaled_dim_y)

        msk = np.pad(msk * msk, (offsets_x, offsets_y),
                    mode='constant', constant_values=1.0)
        mask_image = Image.fromarray((msk.T * 255).astype('uint8'))
        return mask_image

    # --- NEW: manual mask alignment + union helpers ---

    def align_manual_mask_to_canvas(
        self,
        mask,
        scaled_dim_x,
        scaled_dim_y,
        new_dim_x,
        new_dim_y,
        paste_x,
        paste_y,
    ):
        """
        Takes an incoming ComfyUI MASK tensor, resizes it to the scaled image size,
        then pastes it into a new (new_dim_x, new_dim_y) canvas at the same offsets
        used for the image. Returns a PIL L mask (0..255).
        Returns None if the mask is effectively empty.
        """
        manual = self.masktensor2pil(mask)

        # Detect "empty" masks (all black) deterministically.
        # If empty, treat as not provided so behavior matches current node.
        if self.is_pil_mask_empty(manual):
            return None

        # Resize to scaled image region (nearest preserves edges)
        manual_scaled = manual.resize((scaled_dim_x, scaled_dim_y), Image.NEAREST)

        # Paste into a full canvas (black background = keep)
        canvas = Image.new("L", (new_dim_x, new_dim_y), color=0)
        canvas.paste(manual_scaled, (int(paste_x), int(paste_y)))
        return canvas
    
    def _fit_to_shape(self, arr, target_x, target_y):
        """
        arr shape is (x, y). Make it exactly (target_x, target_y)
        via symmetric pad or center-crop (deterministic).
        """
        x, y = arr.shape

        # Pad if too small
        if x < target_x:
            pad_before = (target_x - x) // 2
            pad_after = target_x - x - pad_before
        else:
            pad_before = pad_after = 0

        if y < target_y:
            pad_left = (target_y - y) // 2
            pad_right = target_y - y - pad_left
        else:
            pad_left = pad_right = 0

        if pad_before or pad_after or pad_left or pad_right:
            arr = np.pad(arr, ((pad_before, pad_after), (pad_left, pad_right)),
                        mode="constant", constant_values=0.0)

        # Crop if too large
        x, y = arr.shape
        if x > target_x:
            start = (x - target_x) // 2
            arr = arr[start:start + target_x, :]
        if y > target_y:
            start = (y - target_y) // 2
            arr = arr[:, start:start + target_y]

        return arr    

    def union_masks(self, mask_a_pil, mask_b_pil):
        """
        Pixelwise union: max(A, B) in 8-bit space.
        Both inputs must be 'L' and same size.
        """
        if mask_a_pil.mode != "L":
            mask_a_pil = mask_a_pil.convert("L")
        if mask_b_pil.mode != "L":
            mask_b_pil = mask_b_pil.convert("L")

        a = np.array(mask_a_pil, dtype=np.uint8)
        b = np.array(mask_b_pil, dtype=np.uint8)

        if a.shape != b.shape:
            raise ValueError(
                f"Mask size mismatch for union: {a.shape} vs {b.shape}"
            )

        out = np.maximum(a, b).astype(np.uint8)
        return Image.fromarray(out, mode="L")

    def is_pil_mask_empty(self, mask_pil):
        """
        Returns True if mask is effectively empty (all zeros).
        Deterministic: checks max pixel value.
        """
        if mask_pil.mode != "L":
            mask_pil = mask_pil.convert("L")
        arr = np.array(mask_pil, dtype=np.uint8)
        return int(arr.max()) == 0
    
    def grow_mask_white(self, mask_pil, px):
        """
        Deterministically grows (dilates) the WHITE region of a mask by `px`.
        White = paint, black = keep.
        Uses PIL MaxFilter: stable + fast.
        """
        if mask_pil.mode != "L":
            mask_pil = mask_pil.convert("L")

        # MaxFilter size must be odd and >= 3 to have an effect.
        k = int(px) * 2 + 1
        if k < 3:
            return mask_pil

        return mask_pil.filter(ImageFilter.MaxFilter(size=k))

    # --- Conversion helpers ---

    def tensor2pil(self, image):
        image = image.squeeze(0).to(dtype=torch.float16).cpu().numpy()
        image = (image * 255).clip(0, 255).astype(np.uint8)
        if image.ndim == 3 and image.shape[-1] == 3:
            return Image.fromarray(image, mode="RGB")
        return Image.fromarray(image, mode="L")

    def pil2tensor(self, image):
        image = np.array(image).astype(np.float16) / 255.0
        return torch.from_numpy(image).unsqueeze(0).to(dtype=torch.float16)

    def masktensor2pil(self, mask):
        """
        ComfyUI MASK tensors are typically (1,H,W) float in [0,1].
        Convert to PIL 'L' 0..255.
        """
        m = mask
        # Some masks might arrive with extra dims; normalize deterministically.
        if isinstance(m, torch.Tensor):
            m = m.detach()
        m = m.squeeze()

        m = m.to(dtype=torch.float32).cpu().numpy()
        m = np.clip(m, 0.0, 1.0)
        m = (m * 255.0).round().astype(np.uint8)

        # Ensure 2D
        if m.ndim == 3:
            # If (H,W,1) or similar, collapse last dim.
            m = m[..., 0]

        return Image.fromarray(m, mode="L")

    def pil2masktensor(self, mask_pil):
        """
        PIL 'L' -> MASK tensor (1,H,W) float16 in [0,1]
        """
        if mask_pil.mode != "L":
            mask_pil = mask_pil.convert("L")
        arr = np.array(mask_pil).astype(np.float16) / 255.0
        return torch.from_numpy(arr).unsqueeze(0).to(dtype=torch.float16)


NODE_CLASS_MAPPINGS = {
    "JLC_PaddedImage": JLC_PaddedImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JLC_PaddedImage": "JLC Padded Image",
}