# Changelog

## 2.3.0 - 2026-09-18

### Added

- Added JLC GPU Cooldown with timer, NVIDIA temperature, and combined modes.
  It is a generic, list-aware passthrough gate and performs no implicit VRAM
  cleanup, model unloading, tensor copying, synchronization, or device transfer.
- Added live, non-serialized cooldown status in the frontend.

### Changed

- Included the Seed Generator `-1` intentionally-unseeded sentinel and random
  seed record/replay work added after the 2.2.0 release. The sentinel is passed
  through unchanged and is never substituted with a hidden random seed;
  downstream consumers must implement their own `-1` behavior.
- Marked unseeded Seed Generator executions and GPU Cooldown executions as
  volatile so ComfyUI caching cannot suppress behavior that must run per prompt.

### Fixed

- Count cooldown sensor-call time in monotonic timer and temperature-stability
  calculations.
- Reject an invalid physical NVML GPU index instead of retrying it forever as a
  telemetry outage. Actual NVML initialization and reading failures still hold
  execution until recovery or cancellation.
- Refresh disabled/completed cooldown status, guard duplicate listener
  registration, and clear deferred type-refresh timers when nodes are removed.

### Validation note

The GPU cooldown temperature path was previously live-tested on an NVIDIA
laptop, cooling from 83°C to 70°C in 30.1 seconds and releasing successfully.
That hardware test was supplied as release context and was not rerun during this
release-preparation pass.
