"""
JLC Stage Boundary VRAM Cleanup
-------------------------------

- JLC ComfyUI Nodes Collection
  - This node is part of the **JLC Custom Nodes for ComfyUI** collection
    developed by **J. L. Córdova**.

- Experimental Warning
    This is an experimental utility node.

    It is intended for advanced multi-stage workflows where the user
    deliberately wants to free selected heavy model objects after a stage
    boundary. It may affect model residency, reload behavior, execution time,
    and VRAM usage in ways that depend on ComfyUI's current internal model
    management behavior.

    Use only when the workflow is structured so that the upstream heavy model
    objects are no longer needed after the passthrough boundary. ComfyUI remains
    the authority for model lifecycle management, and this node should be
    treated as a best-effort helper rather than a guaranteed VRAM reset.

- Purpose
    A type-agnostic stage-boundary cleanup passthrough node for multi-stage
    workflows.

    Typical use cases:
        • LATENT boundary:
            Stage 1 creates a latent -> cleanup -> Stage 2 continues denoising.

        • STRING boundary:
            A vision/caption stage finishes prompt generation -> cleanup -> a
            different text encoder/model family consumes the prompt.

        • IMAGE or other boundary:
            Any ordinary ComfyUI value can act as the execution dependency when
            it naturally marks the end of the upstream stage.

    The passthrough value is not interpreted or modified. Its socket type is
    resolved dynamically by the companion frontend extension and the exact same
    Python object is returned after cleanup.

    The first input is a targeted ComfyUI-managed model object. It accepts a
    normal MODEL directly and also accepts wrapper objects such as CLIP or VAE
    when they expose ComfyUI's standard ``patcher`` object. This makes the same
    targeted unload path usable for diffusion models and text encoders without
    falling back to the broad unload-all hammer.

    Cleanup targets are:
        • the connected MODEL / CLIP / VAE patcher and its clones/additional models
        • all currently loaded ComfyUI models, when explicitly requested
        • JLC-managed ControlNet resident cache entries
        • all JLC-managed resident cache entries, when explicitly requested
        • final best-effort allocator cleanup

- Design Notes
    ComfyUI remains the authority for its own model management. This node only
    calls ComfyUI's public-ish model_management helpers when available, and it
    uses the JLC shared model cache core for JLC-owned resident models.

    Both connection sockets are declared as wildcards so normal ComfyUI MODEL,
    CLIP, VAE, STRING, LATENT, IMAGE, and custom types can connect. The frontend
    resolves each socket to its live concrete datatype for normal coloring.

    The physical input order is intentionally MODEL OBJECT first and passthrough
    second. This revision prioritizes clean wiring over compatibility with the
    earlier experimental latent-first layout.

- Attribution & License
  - Concept and implementation by **J. L. Córdova** with development
    assistance from **ChatGPT (OpenAI)**.

  - Designed for use with:
    https://github.com/comfyanonymous/ComfyUI

  - Copyright (c) 2026 J. L. Córdova
  - Released under the **MIT License**.
"""

from __future__ import annotations

from ...jlc_custom_nodes_versions import JLC_UTIL_NODES_VERSION

MANIFEST = {
    "name": "JLC Stage Boundary VRAM Cleanup",
    "version": JLC_UTIL_NODES_VERSION,
    "author": "J. L. Córdova",
    "description": (
        "Type-agnostic, list-aware stage-boundary cleanup passthrough for multi-stage "
        "ComfyUI workflows. A wildcard target accepts MODEL/CLIP/VAE objects "
        "that expose a ComfyUI patcher, while a second dynamic passthrough "
        "socket carries STRING/LATENT/IMAGE or other values unchanged."
    ),
}

import gc
import time
from typing import Any, Optional


class _AnyType(str):
    """A ComfyUI wildcard that compares compatible with every socket type."""

    def __ne__(self, _other: object) -> bool:
        return False


ANY_TYPE = _AnyType("*")


def _first_list_item(value: Any, default: Any = None) -> Any:
    """Return the first item of a ComfyUI list-mapped input, or default."""

    if isinstance(value, list):
        return value[0] if value else default
    return value if value is not None else default


def _as_bool(value: Any, default: bool = False) -> bool:
    """Normalize ordinary or list-wrapped boolean-ish inputs."""

    value = _first_list_item(value, default)
    return bool(value)


def _passthrough_payload(value: Any) -> Any:
    """Keep passthrough payload exactly as received under list mapping."""

    return value

try:
    import comfy.model_management as model_management
except Exception as exc:  # pragma: no cover - depends on ComfyUI runtime
    model_management = None
    _MODEL_MANAGEMENT_IMPORT_ERROR = exc
else:
    _MODEL_MANAGEMENT_IMPORT_ERROR = None


try:
    from ..engines.jlc_model_cache_core import (
        cuda_cleanup as jlc_cuda_cleanup,
        evict_family as jlc_evict_family,
        unload_all as jlc_unload_all,
    )
except Exception as exc:  # pragma: no cover - import layout fallback
    _JLC_CACHE_IMPORT_ERROR = exc
    try:
        # Fallback for unusual import contexts during development/testing.
        from nodes.engines.jlc_model_cache_core import (  # type: ignore
            cuda_cleanup as jlc_cuda_cleanup,
            evict_family as jlc_evict_family,
            unload_all as jlc_unload_all,
        )
    except Exception:
        jlc_cuda_cleanup = None
        jlc_evict_family = None
        jlc_unload_all = None
else:
    _JLC_CACHE_IMPORT_ERROR = None

# End of Import Section
# ###############################################################


def _log(message: str, *, verbose: bool = True) -> None:
    if verbose:
        print(f"[JLC Stage Boundary VRAM Cleanup] {message}")


def _warn(message: str) -> None:
    print(f"[JLC Stage Boundary VRAM Cleanup] Warning: {message}")


def _unwrap_model_patcher(model: Any) -> Optional[Any]:
    """
    Return the ComfyUI ModelPatcher/CoreModelPatcher carried by a model object.

    Supported forms include:
      • MODEL sockets, which normally already carry a patcher
      • CLIP / VAE wrappers exposing ``.patcher``
      • common dict wrappers used by custom nodes

    Unknown objects return ``None`` rather than being handed blindly to ComfyUI
    model-management internals.
    """

    if model is None:
        return None

    # Direct MODEL / ModelPatcher path.
    if hasattr(model, "clone_base_uuid"):
        return model

    # Standard ComfyUI CLIP and VAE wrappers expose their managed patcher here.
    patcher = getattr(model, "patcher", None)
    if patcher is not None and hasattr(patcher, "clone_base_uuid"):
        return patcher

    if isinstance(model, dict):
        for key in ("patcher", "model_patcher", "model", "unet", "clip", "vae"):
            candidate = model.get(key)
            if candidate is None:
                continue
            if hasattr(candidate, "clone_base_uuid"):
                return candidate
            candidate_patcher = getattr(candidate, "patcher", None)
            if candidate_patcher is not None and hasattr(candidate_patcher, "clone_base_uuid"):
                return candidate_patcher

    return None


def _soft_empty_cache(*, verbose: bool = True) -> bool:
    """Call ComfyUI's backend-aware soft cache cleanup when available."""

    if model_management is None:
        if _MODEL_MANAGEMENT_IMPORT_ERROR is not None:
            _warn(f"could not import comfy.model_management: {_MODEL_MANAGEMENT_IMPORT_ERROR}")
        return False

    soft_empty_cache = getattr(model_management, "soft_empty_cache", None)
    if not callable(soft_empty_cache):
        _warn("comfy.model_management.soft_empty_cache() is unavailable.")
        return False

    try:
        try:
            soft_empty_cache(True)
        except TypeError:
            soft_empty_cache()
        _log("ComfyUI soft_empty_cache executed.", verbose=verbose)
        return True
    except Exception as exc:
        _warn(f"ComfyUI soft_empty_cache failed: {exc}")
        return False


def _unload_connected_model(model: Any, *, all_devices: bool, verbose: bool = True) -> bool:
    """Unload one connected MODEL/CLIP/VAE patcher and related models if possible."""

    if model is None:
        _log("No model object connected; targeted ComfyUI unload skipped.", verbose=verbose)
        return False

    if model_management is None:
        if _MODEL_MANAGEMENT_IMPORT_ERROR is not None:
            _warn(f"could not import comfy.model_management: {_MODEL_MANAGEMENT_IMPORT_ERROR}")
        return False

    unload_model_and_clones = getattr(model_management, "unload_model_and_clones", None)
    if not callable(unload_model_and_clones):
        _warn("comfy.model_management.unload_model_and_clones() is unavailable.")
        return False

    target = _unwrap_model_patcher(model)
    if target is None:
        _log("Connected object does not expose a supported ComfyUI model patcher; targeted unload skipped.", verbose=verbose)
        return False

    try:
        unload_model_and_clones(
            target,
            unload_additional_models=True,
            all_devices=bool(all_devices),
        )
        _log(
            "Requested targeted unload of connected model object, clones, and "
            f"additional models; all_devices={bool(all_devices)}.",
            verbose=verbose,
        )
        return True
    except TypeError:
        # Older/forked ComfyUI fallback if the helper exists but has a narrower
        # signature. Keep this conservative rather than inventing our own clone
        # matching logic.
        try:
            unload_model_and_clones(target)
            _log("Requested targeted unload of connected model object using fallback signature.", verbose=verbose)
            return True
        except Exception as exc:
            _warn(f"targeted ComfyUI model-object unload failed with fallback signature: {exc}")
            return False
    except Exception as exc:
        _warn(f"targeted ComfyUI model-object unload failed: {exc}")
        return False


def _unload_all_comfy_models(*, verbose: bool = True) -> bool:
    """Explicit hammer: ask ComfyUI to unload all currently resident models."""

    if model_management is None:
        if _MODEL_MANAGEMENT_IMPORT_ERROR is not None:
            _warn(f"could not import comfy.model_management: {_MODEL_MANAGEMENT_IMPORT_ERROR}")
        return False

    unload_all_models = getattr(model_management, "unload_all_models", None)
    if not callable(unload_all_models):
        _warn("comfy.model_management.unload_all_models() is unavailable.")
        return False

    try:
        unload_all_models()
        _log("Requested ComfyUI unload_all_models().", verbose=verbose)
        return True
    except Exception as exc:
        _warn(f"ComfyUI unload_all_models failed: {exc}")
        return False


def _evict_jlc_controlnet_cache(*, safe: bool, verbose: bool = True) -> int:
    """Evict JLC-managed ControlNet cache entries."""

    if jlc_evict_family is None:
        if _JLC_CACHE_IMPORT_ERROR is not None:
            _warn(f"could not import JLC cache core: {_JLC_CACHE_IMPORT_ERROR}")
        else:
            _warn("JLC cache core evict_family() is unavailable.")
        return 0

    try:
        count = int(
            jlc_evict_family(
                "controlnet",
                reason="stage_boundary_controlnet_evict",
                safe=bool(safe),
            )
        )
        _log(f"Evicted {count} JLC ControlNet cache entr{'y' if count == 1 else 'ies'}.", verbose=verbose)
        return count
    except Exception as exc:
        _warn(f"JLC ControlNet cache eviction failed: {exc}")
        return 0


def _evict_all_jlc_cache(*, safe: bool, verbose: bool = True) -> int:
    """Explicit hammer: evict all JLC-managed resident cache entries."""

    if jlc_unload_all is None:
        if _JLC_CACHE_IMPORT_ERROR is not None:
            _warn(f"could not import JLC cache core: {_JLC_CACHE_IMPORT_ERROR}")
        else:
            _warn("JLC cache core unload_all() is unavailable.")
        return 0

    try:
        count = int(
            jlc_unload_all(
                include_keep=True,
                reason="stage_boundary_jlc_unload_all",
                safe=bool(safe),
            )
        )
        _log(f"Evicted {count} total JLC cache entr{'y' if count == 1 else 'ies'}.", verbose=verbose)
        return count
    except Exception as exc:
        _warn(f"JLC cache unload_all failed: {exc}")
        return 0


def _final_allocator_cleanup(*, safe: bool, verbose: bool = True) -> None:
    """Run backend-aware Comfy cleanup and JLC defensive CUDA cleanup."""

    gc.collect()
    _soft_empty_cache(verbose=verbose)

    if jlc_cuda_cleanup is not None:
        try:
            jlc_cuda_cleanup(
                reason="stage_boundary_final_allocator_cleanup",
                synchronize=True,
                safe=bool(safe),
            )
            _log(f"JLC allocator cleanup executed; safe={bool(safe)}.", verbose=verbose)
            return
        except Exception as exc:
            _warn(f"JLC allocator cleanup failed: {exc}")

    # Last-resort fallback only if the shared cache core could not be imported.
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
            torch.cuda.synchronize()
        gc.collect()
        _log("Fallback torch CUDA allocator cleanup executed.", verbose=verbose)
    except Exception as exc:
        _warn(f"fallback CUDA allocator cleanup failed: {exc}")


class JLC_StageBoundaryVRAMCleanup:
    """Type-agnostic stage-boundary VRAM cleanup passthrough node."""

    FUNCTION = "cleanup"
    CATEGORY = "utils/VRAM"
    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("passthrough",)
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)
    DESCRIPTION = (
        "Runs best-effort VRAM/model cleanup at an explicit workflow stage "
        "boundary, then returns the incoming passthrough value unchanged. "
        "When upstream provides a list (for example one prompt per tile), "
        "cleanup executes once for the whole list rather than once per item. "
        "The first socket accepts a MODEL, CLIP, VAE, or compatible wrapper "
        "that exposes a ComfyUI patcher; the second socket dynamically follows "
        "STRING, LATENT, IMAGE, or other ComfyUI types."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "passthrough": (ANY_TYPE,),
                "unload_connected_model": ("BOOLEAN", {"default": True}),
                "evict_jlc_controlnet_cache": ("BOOLEAN", {"default": False}),
                "evict_all_jlc_model_cache": ("BOOLEAN", {"default": False}),
                "unload_all_comfy_models": ("BOOLEAN", {"default": False}),
                "clear_cuda_allocator": ("BOOLEAN", {"default": True}),
                "safe_cleanup": ("BOOLEAN", {"default": True}),
                "all_devices": ("BOOLEAN", {"default": False}),
                "verbose": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                # Wildcard by design: ComfyUI MODEL, CLIP and VAE wrappers use
                # different socket types but can all expose a managed patcher.
                "model": (ANY_TYPE,),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # This node has intentional side effects, so force execution whenever it
        # sits on an active graph path instead of relying only on cached inputs.
        return time.time()

    def cleanup(
        self,
        passthrough,
        unload_connected_model: bool = True,
        evict_jlc_controlnet_cache: bool = False,
        evict_all_jlc_model_cache: bool = False,
        unload_all_comfy_models: bool = False,
        clear_cuda_allocator: bool = True,
        safe_cleanup: bool = True,
        all_devices: bool = False,
        verbose: bool = True,
        model: Any = None,
    ):
        # INPUT_IS_LIST=True means ComfyUI supplies every input as a list.
        # For the passthrough payload, that is exactly what we want: a gathered
        # list of items can act as one stage-boundary token and should be
        # returned unchanged so downstream scalar nodes can fan back out.
        passthrough_value = _passthrough_payload(passthrough)
        verbose_flag = _as_bool(verbose, True)
        started = time.time()

        item_count = len(passthrough_value) if isinstance(passthrough_value, list) else 1
        _log(
            f"Stage-boundary cleanup triggered by passthrough input; item_count={item_count}.",
            verbose=verbose_flag,
        )

        model_value = _first_list_item(model, None)
        unload_connected_model_flag = _as_bool(unload_connected_model, True)
        evict_jlc_controlnet_cache_flag = _as_bool(evict_jlc_controlnet_cache, False)
        evict_all_jlc_model_cache_flag = _as_bool(evict_all_jlc_model_cache, False)
        unload_all_comfy_models_flag = _as_bool(unload_all_comfy_models, False)
        clear_cuda_allocator_flag = _as_bool(clear_cuda_allocator, True)
        safe_cleanup_flag = _as_bool(safe_cleanup, True)
        all_devices_flag = _as_bool(all_devices, False)

        # ComfyUI model residency cleanup.
        if unload_all_comfy_models_flag:
            _unload_all_comfy_models(verbose=verbose_flag)
        elif unload_connected_model_flag:
            _unload_connected_model(model_value, all_devices=all_devices_flag, verbose=verbose_flag)

        # JLC-owned resident cache cleanup. If the all-cache hammer is selected,
        # do not separately evict the ControlNet family first.
        if evict_all_jlc_model_cache_flag:
            _evict_all_jlc_cache(safe=safe_cleanup_flag, verbose=verbose_flag)
        elif evict_jlc_controlnet_cache_flag:
            _evict_jlc_controlnet_cache(safe=safe_cleanup_flag, verbose=verbose_flag)

        if clear_cuda_allocator_flag:
            _final_allocator_cleanup(safe=safe_cleanup_flag, verbose=verbose_flag)
        else:
            gc.collect()
            _log("Allocator cleanup disabled; Python gc.collect() only.", verbose=verbose_flag)

        elapsed = time.time() - started
        _log(
            f"Cleanup complete in {elapsed:.3f}s. Passing value through unchanged.",
            verbose=verbose_flag,
        )
        return (passthrough_value,)



NODE_CLASS_MAPPINGS = {
    "JLC_StageBoundaryVRAMCleanup": JLC_StageBoundaryVRAMCleanup,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JLC_StageBoundaryVRAMCleanup": "\u2003JLC Stage Boundary VRAM Cleanup",
}
