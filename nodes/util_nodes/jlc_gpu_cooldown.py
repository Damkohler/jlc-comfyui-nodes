"""
JLC GPU Cooldown
----------------

- JLC ComfyUI Nodes Collection
  - This node is part of the **JLC Custom Nodes for ComfyUI** collection
    developed by **J. L. Córdova**.

- Purpose
    Provide an interruptible timer, NVIDIA temperature, or combined pause at a
    workflow dependency boundary. The connected value is returned unchanged,
    including its Python object identity and any list structure.

- Contract
    This node does not unload models, clear VRAM, copy tensors, synchronize
    CUDA, or move values between devices. Temperature modes use the optional
    ``pynvml`` module supplied by ``nvidia-ml-py``. A telemetry failure holds
    execution and retries until the sensor recovers or the prompt is canceled.

- Attribution & License
  - Concept and implementation by **J. L. Córdova** with development
    assistance from **ChatGPT (OpenAI)**.

  - Designed for use with:
    https://github.com/comfyanonymous/ComfyUI

  - Copyright (c) 2026 J. L. Córdova
  - Released under the **MIT License**.
"""
from __future__ import annotations

import math
import time

from ...jlc_custom_nodes_versions import JLC_UTIL_NODES_VERSION

MANIFEST = {
    "name": "JLC GPU Cooldown",
    "version": JLC_UTIL_NODES_VERSION,
    "author": "J. L. Córdova",
    "description": (
        "Interruptible timer and NVIDIA temperature cooldown with generic, "
        "list-aware passthrough and no implicit memory cleanup."
    ),
}


class _AnyType(str):
    def __ne__(self, other):
        return False


ANY_TYPE = _AnyType("*")


class _GPUSelectionError(ValueError):
    """The configured physical NVML index does not exist."""


def _first(value):
    return value[0] if isinstance(value, list) else value


def _interrupt():
    from comfy.model_management import throw_exception_if_processing_interrupted
    throw_exception_if_processing_interrupted()


def _sleep(seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        _interrupt()
        time.sleep(min(0.2, max(0, end - time.monotonic())))


class _Sensor:
    def __init__(self, index):
        self.index = index
        self.nvml = None
        self.handle = None

    def read(self):
        if self.nvml is None:
            import pynvml
            pynvml.nvmlInit()
            self.nvml = pynvml
        if self.handle is None:
            device_count = int(self.nvml.nvmlDeviceGetCount())
            if self.index >= device_count:
                raise _GPUSelectionError(
                    f"GPU index {self.index} is unavailable; NVML reports "
                    f"{device_count} physical device(s)"
                )
            self.handle = self.nvml.nvmlDeviceGetHandleByIndex(self.index)
        value = float(self.nvml.nvmlDeviceGetTemperature(
            self.handle, self.nvml.NVML_TEMPERATURE_GPU))
        if not math.isfinite(value) or not 0 <= value <= 150:
            raise ValueError(f"Invalid temperature: {value}")
        return value

    def close(self):
        if self.nvml is not None:
            try:
                self.nvml.nvmlShutdown()
            except Exception:
                pass
            self.nvml = None
            self.handle = None


class JLC_GPUCooldown:
    CATEGORY = "utils/VRAM"
    FUNCTION = "cooldown"
    RETURN_TYPES = (ANY_TYPE, "FLOAT", "STRING")
    RETURN_NAMES = ("passthrough", "waited_seconds", "status")
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, False, False)
    OUTPUT_NODE = True  # Also usable as a terminal gate after a connected stage.
    DESCRIPTION = "Wait between stages without modifying data or freeing memory. Sensor failure holds until recovery or cancellation."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "passthrough": (ANY_TYPE,),
            "enabled": ("BOOLEAN", {"default": True}),
            "mode": (["timer", "temperature", "combined"], {"default": "combined"}),
            "minimum_wait_seconds": ("FLOAT", {"default": 30.0, "min": 0, "max": 86400, "step": 1}),
            "resume_temperature_c": ("FLOAT", {"default": 70.0, "min": 20, "max": 100, "step": 1}),
            "stable_seconds": ("FLOAT", {"default": 5.0, "min": 0, "max": 600, "step": 1}),
            "poll_interval_seconds": ("FLOAT", {"default": 2.0, "min": 0.2, "max": 60, "step": 0.2}),
            "gpu_index": ("INT", {"default": 0, "min": 0, "max": 31,
                "tooltip": "Physical NVML GPU index, not a remapped CUDA index."}),
        }, "hidden": {"unique_id": "UNIQUE_ID"}}

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def cooldown(self, passthrough, enabled=True, mode="combined",
                 minimum_wait_seconds=30.0, resume_temperature_c=70.0,
                 stable_seconds=5.0, poll_interval_seconds=2.0,
                 gpu_index=0, unique_id=None):
        enabled, mode = _first(enabled), _first(mode)
        minimum, target, stable, poll = map(float, map(_first, (
            minimum_wait_seconds, resume_temperature_c, stable_seconds, poll_interval_seconds)))
        if mode not in ("timer", "temperature", "combined"):
            raise ValueError("Unknown cooldown mode")
        if not all(math.isfinite(v) for v in (minimum, target, stable, poll)) or min(minimum, stable) < 0 or poll < 0.2:
            raise ValueError("Invalid cooldown timing parameters")
        if mode == "timer":
            gpu_index = 0  # Timer mode has no telemetry or GPU-selection dependency.
        else:
            gpu_index = int(_first(gpu_index))
            if not 0 <= gpu_index <= 31:
                raise ValueError("GPU index must be between 0 and 31")

        node_id = _first(unique_id)
        last_log = -math.inf

        def report(message, final=False):
            nonlocal last_log
            now = time.monotonic()
            if final or now - last_log >= 10:
                print(f"[JLC GPU Cooldown #{node_id}] {message}", flush=True)
                last_log = now
            try:
                from server import PromptServer
                PromptServer.instance.send_sync("jlc.gpu_cooldown", {"node": node_id, "text": message})
            except Exception:
                pass  # UI availability must not control the cooling gate.

        if not enabled:
            report("Cooldown disabled", final=True)
            return {"ui": {"text": ["Cooldown disabled"]}, "result": (passthrough, 0.0, "Disabled")}

        started = time.monotonic()
        below_since = None
        sensor = _Sensor(gpu_index) if mode != "timer" else None
        report(f"Cooldown starting in {mode} mode")

        try:
            while True:
                _interrupt()
                now = time.monotonic()
                elapsed = now - started
                timer_ready = mode == "temperature" or elapsed >= minimum
                temperature_ready = mode == "timer"
                message = f"Resting: {elapsed:.1f}s / {minimum:.1f}s"
                if sensor is not None:
                    try:
                        temperature = sensor.read()
                    except _GPUSelectionError:
                        sensor.close()
                        raise
                    except Exception as exc:
                        now = time.monotonic()
                        elapsed = now - started
                        timer_ready = mode == "temperature" or elapsed >= minimum
                        below_since = None
                        sensor.close()
                        message = f"HOLD: sensor unavailable ({exc}); retrying. Cancel to exit. Elapsed {elapsed:.1f}s"
                    else:
                        now = time.monotonic()
                        elapsed = now - started
                        timer_ready = mode == "temperature" or elapsed >= minimum
                        if temperature <= target:
                            if below_since is None:
                                below_since = now
                            temperature_ready = now - below_since >= stable
                        else:
                            below_since = None
                        message = f"GPU {sensor.index}: {temperature:.1f} C | target <= {target:.1f} C | elapsed {elapsed:.1f}s"
                report(message)
                if timer_ready and temperature_ready:
                    status = f"Cooldown complete after {elapsed:.1f}s. {message}"
                    report(status, final=True)
                    return {"ui": {"text": [status]}, "result": (passthrough, elapsed, status)}
                delay = poll
                if mode == "timer":
                    delay = min(poll, max(0, minimum - elapsed))
                _sleep(delay)
        finally:
            if sensor is not None:
                sensor.close()


NODE_CLASS_MAPPINGS = {"JLC_GPUCooldown": JLC_GPUCooldown}
NODE_DISPLAY_NAME_MAPPINGS = {"JLC_GPUCooldown": "\u2003JLC GPU Cooldown"}
