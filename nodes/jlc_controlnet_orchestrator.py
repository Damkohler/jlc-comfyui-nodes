"""
JLC ControlNet Orchestrator
---------------------------

- JLC ComfyUI Nodes Collection
  - This node is part of the **JLC Custom Nodes for ComfyUI**
    collection developed by **J. L. Córdova**.

  - Repository
    https://github.com/Damkohler/jlc-comfyui-nodes

  - The JLC nodes focus on practical workflow improvements for
    image-generation pipelines, particularly:
        • Flux-based workflows
        • LoRA experimentation
        • advanced inpainting / outpainting pipelines
        • multi-ControlNet composition and orchestration

- Node Purpose
  - The **JLC ControlNet Orchestrator** is the external-input interface
    for building weighted multi-ControlNet workflows with the JLC
    non-recursive composition architecture.

  - Unlike **JLC ControlNet Orchestrator (Advanced)**, this node does not
    load ControlNet models internally. It receives ControlNet objects from
    upstream loader nodes, allowing model selection and loading policy to
    remain explicit in the workflow graph.

  - The node provides:
        • multiple externally supplied ControlNet slots
        • independent hint, strength, start, end, and weight controls
          for each active slot
        • isolated per-run ControlNet preparation
        • deliberate null-image handling: a `None` hint disables only its slot
        • optional order bias through the global `alpha` parameter
        • weighted non-recursive fusion through the shared JLC core

  - Each active slot:
        • receives a ControlNet base object from an upstream loader
        • creates an isolated per-run ControlNet copy
        • applies its own hint image, strength, activation range, and VAE
        • remains independent of the other prepared ControlNet instances

  - Multi-ControlNet execution uses the shared JLC linearized,
    non-recursive fusion core:

        combined = Σ [(w_i · alpha^i) · C_i(x)]

    where:
        • C_i(x) is the output of ControlNet slot `i`, evaluated
          independently against the same sampler state
        • w_i is the user-defined slot weight
        • alpha^i applies the optional order bias
        • negative weights and negative alpha values are supported

  - The fusion core:
        • does not build a recursive `previous_controlnet` chain
          between active slots
        • does not mutate upstream ControlNet base objects
        • does not use `deepcopy`
        • evaluates prepared ControlNets independently
        • clones tensors only when taking ownership of output data
        • accumulates subsequent outputs in-place
        • presents one ControlNet-compatible wrapper to the ComfyUI sampler

  - When only one effective ControlNet remains at a final weight of 1.0,
    the node uses the mathematically equivalent native single-ControlNet
    path instead of creating a composed wrapper.

  - Runtime Integration
    - The node exposes child models, hooks, lifecycle methods, and inference
      memory requirements through the interfaces expected by current ComfyUI
      sampler and model-management paths.

    - It is designed to cooperate with normal ComfyUI model management and
      DynamicVRAM behavior. It does not replace ComfyUI loading, offloading,
      patching, or device-residency policy.

    - Because ControlNet models are supplied externally, upstream loader
      nodes remain responsible for model selection, reuse, and residency
      behavior.

  - Workflow Position
    - **JLC ControlNet Orchestrator** is useful when ControlNet loading must
      remain explicit or when externally loaded ControlNet objects are shared
      elsewhere in the workflow.

    - **JLC ControlNet Orchestrator (Advanced)** is the preferred integrated
      interface for new workflows that benefit from internal model selection,
      shared-cache reuse, slot preflight, and `SHARE_PREVIOUS` behavior.

    - Both Orchestrator variants use the same validated non-recursive
      weighted-fusion algorithm.

  - MultiGPU Scope
    - The wrapper includes compatibility shunts required by current ComfyUI
      interfaces but does not implement real MultiGPU ControlNet cloning.

    - Use this node as a single-device ControlNet orchestration path unless
      explicit MultiGPU support is added in a future implementation.

- Attribution & License
  - Concept and implementation by **J. L. Córdova**
    with development assistance from **ChatGPT (OpenAI)**.

  - Inspired by and interoperable with the ControlNet execution model in:
    https://github.com/comfyanonymous/ComfyUI

  - Copyright (c) 2026 J. L. Córdova

  - Released under the **MIT License**.
"""

from __future__ import annotations

from ..jlc_custom_nodes_versions import JLC_CONTROLNET_VERSION

from .jlc_controlnet_nonrecursive_core import (
    DEBUG,
    compose_or_native_fallback,
    prepare_controlnet_copy,
    safe_cnet_name,
)

MANIFEST = {
    "name": "JLC ControlNet Orchestrator",
    "version": JLC_CONTROLNET_VERSION,
    "author": "J. L. Córdova",
    "description": (
        "External-input multi-ControlNet orchestrator using the validated JLC "
        "non-recursive weighted-fusion core. Accepts ControlNet objects from "
        "upstream loaders, applies isolated per-slot conditioning, disables only "
        "the corresponding slot when its runtime hint is None, supports order-biased "
        "weights, routes mathematically equivalent single-ControlNet "
        "cases through the native path, and preserves compatibility with current "
        "ComfyUI sampler and model-management interfaces."
    ),
}

MAX_SLOTS = 5


class JLC_ControlNetOrchestrator:
    FUNCTION = "orchestrate"
    CATEGORY = "conditioning/controlnet"

    @classmethod
    def INPUT_TYPES(cls):
        optional = {}

        for i in range(1, MAX_SLOTS + 1):
            idx = f"{i:02d}"
            optional[f"image_{idx}"] = ("IMAGE", {
                "tooltip": "Control image for this slot. Slot is skipped if this is not connected."
            })
            optional[f"control_net_{idx}"] = ("CONTROL_NET", {
                "tooltip": "ControlNet for this slot. Empty slots reuse the previous active ControlNet where possible."
            })
            optional[f"enabled_{idx}"] = ("BOOLEAN", {"default": True})
            optional[f"strength_{idx}"] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01})
            optional[f"start_{idx}"] = ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001})
            optional[f"end_{idx}"] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001})
            optional[f"weight_{idx}"] = ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01})

        optional["alpha"] = ("FLOAT", {
            "default": 1.0,
            "min": -2.0,
            "max": 2.0,
            "step": 0.01,
            "tooltip": "Order bias. 1.0 = neutral. <1 favors earlier slots. >1 favors later slots. Negative values invert influence.",
        })

        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "slot_count": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": MAX_SLOTS,
                    "step": 1,
                    "tooltip": "Number of ControlNet slots shown in the node.",
                }),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")

    def orchestrate(self, positive, negative, vae, alpha=1.0, **kwargs):
        resolved = []
        current_cnet = None
        inactive_reasons = {}

        for i in range(1, MAX_SLOTS + 1):
            idx = f"{i:02d}"
            enabled = kwargs.get(f"enabled_{idx}", True)
            image = kwargs.get(f"image_{idx}")
            incoming = kwargs.get(f"control_net_{idx}")
            strength = kwargs.get(f"strength_{idx}", 1.0)
            start = kwargs.get(f"start_{idx}", 0.0)
            end = kwargs.get(f"end_{idx}", 1.0)
            weight = kwargs.get(f"weight_{idx}", 1.0)

            # Empty ControlNet slot means implicit SHARE_PREVIOUS, matching the
            # original simple Orchestrator behavior.
            cnet_base = incoming if incoming is not None else current_cnet

            if not enabled:
                inactive_reasons[i] = "disabled"
                continue
            if cnet_base is None:
                inactive_reasons[i] = "no_controlnet"
                continue
            if image is None:
                # Runtime None is the deliberate absent-image signal used by
                # DISABLED/hidden Aux Wrapper outputs. It also covers an
                # unconnected optional image socket. Disable only this slot.
                inactive_reasons[i] = "image_none"
                continue
            if strength == 0:
                inactive_reasons[i] = "strength_zero"
                continue
            if weight == 0:
                inactive_reasons[i] = "weight_zero"
                continue
            if (end - start) <= 0:
                inactive_reasons[i] = "empty_range"
                continue

            current_cnet = cnet_base
            resolved.append({
                "slot": i,
                "base": cnet_base,
                "image": image,
                "strength": strength,
                "start": start,
                "end": end,
                "weight": weight,
            })

        if DEBUG:
            active = [str(item["slot"]) for item in resolved]
            inactive = [str(i) for i in range(1, MAX_SLOTS + 1) if str(i) not in active]
            reason_text = ", ".join(
                f"{slot}:{inactive_reasons.get(slot, 'inactive')}"
                for slot in range(1, MAX_SLOTS + 1)
                if str(slot) not in active
            )
            print(
                f"[JLC-Orchestrator] active={', '.join(active) or 'none'} "
                f"inactive={', '.join(inactive) or 'none'} "
                f"inactive_reasons={reason_text or 'none'} alpha={alpha}"
            )

        if not resolved:
            return (positive, negative)

        prepared_cnets = []
        weights = []
        debug_names = []

        for item in resolved:
            control_hint = item["image"].movedim(-1, 1)
            cnet = prepare_controlnet_copy(
                item["base"],
                control_hint,
                item["strength"],
                (item["start"], item["end"]),
                vae=vae,
            )
            prepared_cnets.append(cnet)
            weights.append(item["weight"])
            debug_names.append(f"slot_{item['slot']:02d}:{safe_cnet_name(cnet)}")

            if DEBUG:
                print(
                    f"[JLC-Orchestrator] prepared slot={item['slot']} "
                    f"type={safe_cnet_name(cnet)} strength={item['strength']} "
                    f"range=({item['start']}, {item['end']}) weight={item['weight']}"
                )

        return compose_or_native_fallback(
            positive,
            negative,
            prepared_cnets,
            weights,
            alpha=alpha,
            debug_label="JLC-Orchestrator",
            debug_names=debug_names,
        )


NODE_CLASS_MAPPINGS = {
    "JLC_ControlNetOrchestrator": JLC_ControlNetOrchestrator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JLC_ControlNetOrchestrator": "JLC ControlNet Orchestrator",
}
