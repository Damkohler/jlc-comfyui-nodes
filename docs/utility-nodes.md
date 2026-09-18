# Utility Nodes

This chapter covers the JLC utility-node family:

- [JLC Seed Generator](#jlc-seed-generator)
- [Randomize Record and Replay](#randomize-record-and-replay)
- [JLC Resize Multiple Images](#jlc-resize-multiple-images)
- [JLC Multi Set/Get](#jlc-multi-setget)
- [JLC Multi Reroute](#jlc-multi-reroute)
- [JLC Boolean Logic (Frontend)](#jlc-boolean-logic-frontend)
- [JLC Stage Boundary VRAM Cleanup](#jlc-stage-boundary-vram-cleanup)
- [JLC GPU Cooldown](#jlc-gpu-cooldown)
- [Choosing the Right Utility Node](#choosing-the-right-utility-node)
- [Example Workflows](#example-workflows)

These nodes are not image-generation algorithms by themselves. They are workflow-support nodes intended to make larger ComfyUI graphs easier to connect, resize, control, run, repeat, debug, or stage.

---

## JLC Seed Generator

**JLC Seed Generator** is a shared seed source for workflows where multiple samplers, captioning stages, or other seed-aware nodes should receive one coordinated seed value during a prompt execution.

Typical use case:

```text
JLC Seed Generator
    ├─ seed_int → seeded stage 1
    ├─ seed_int → seeded stage 2
    └─ seed_int → seeded stage 3
```

This is useful when:

- one seed should feed multiple samplers or inference stages;
- a multi-stage workflow must keep seed ownership in one visible place;
- you want deterministic increment/decrement trials without losing the original starting seed;
- you want to explore random seeds and then replay exactly the same random sequence after changing another parameter;
- a downstream node explicitly supports `-1` as an intentionally unseeded sentinel.

### Outputs

The node returns two seed outputs:

| Output | Purpose |
|---|---|
| `seed` | Small SEED-style dictionary, e.g. `{"seed": 12345}`. |
| `seed_int` | Plain integer seed for nodes that expect an INT seed input. |

In most ordinary ComfyUI seed sockets, `seed_int` is the convenient output.

---

## Seed modes and stable base seed

For ordinary nonnegative seeds, the node supports the familiar control modes:

```text
fixed
increment
decrement
randomize
```

The visible **seed** widget is treated as the user's stable base seed. After queue submission, the frontend restores that visible value instead of leaving the widget advanced to whatever seed happened to be used most recently.

For example:

```text
visible base seed: 1
control after generate: increment

run 1 uses seed 1
run 2 uses seed 2
run 3 uses seed 3
run 4 uses seed 4

visible seed remains 1
```

The status panel reports the seed used by execution while the editable seed widget remains the starting point for the trial.

`fixed`, `increment`, and `decrement` remain deterministic from the visible base seed.

For `randomize`, the JLC frontend assigns the actual nonnegative random seed **before prompt serialization**. This is intentional: it allows the visible base seed to remain stable while still making the submitted random value explicit enough to record and replay reliably.

---

## Intentionally unseeded sentinel: `-1`

`-1` is a first-class JLC sentinel meaning:

```text
intentionally unseeded
```

When the visible seed is `-1`:

- both outputs preserve `-1`;
- the frontend locks seed mutation;
- `fixed`, `increment`, `decrement`, and `randomize` do not alter it;
- every queued prompt continues to receive `-1`;
- randomize record/replay is inactive.

The JLC Seed Generator does **not** translate `-1` into a hidden random number. It passes the sentinel through unchanged.

This is only useful when the downstream consumer defines what `-1` means. A node or engine that requires an ordinary nonnegative seed may not accept the sentinel.

---

## Randomize Record and Replay

Randomize record/replay is implemented and applies only when:

```text
control after generate = randomize
```

The **randomize_replay** selector provides three modes:

```text
off
record
replay
```

### `off`

Each run receives a fresh JLC-generated nonnegative random seed.

The visible base seed remains unchanged, while the status panel shows the seed actually used.

### `record`

Switching into **record** starts a fresh sequence.

For every randomized run, the node:

1. generates the seed before prompt serialization;
2. submits that exact seed to the workflow;
3. immediately appends the same value to the recorded sequence;
4. updates the status panel with the number of captured seeds.

Recording works across both queued submissions and separate presses of **Run**.

Example:

```text
record:
  seed A
  seed B
  seed C
  seed D
```

The recorded sequence is stored in the node's workflow properties so it remains associated with that node/workflow.

### `replay`

Switching into **replay** resets the replay cursor to the beginning of the stored sequence.

Subsequent executions reuse the recorded values in the same order:

```text
replay:
  seed A
  seed B
  seed C
  seed D
```

The replay cursor advances across separate presses of **Run** as well as queued prompts.

If execution continues past the end of the stored sequence, replay cycles back to the beginning. For exact A/B comparisons, queue or run the same number of executions that were originally recorded.

### Empty replay protection

If **replay** is selected but no sequence has been recorded, the node does **not** silently fall back to fresh random seeds.

Instead, it holds the visible base seed and reports that replay is empty. This keeps a mistaken replay request deterministic and obvious.

### Why record/replay only applies to randomize

`fixed`, `increment`, and `decrement` already define deterministic sequences from the visible base seed, so recording them would add no information.

Randomized trials are different: once a useful random sequence has been found, replay lets you change another variable—CFG, steps, denoise strength, LoRA weight, model settings, or another workflow parameter—and compare results against exactly the same sequence of seeds.

Conceptually:

```text
Trial A:
  randomize + record
  seeds A, B, C, D

change one parameter

Trial B:
  randomize + replay
  seeds A, B, C, D
```

This makes randomized parameter studies repeatable without giving up the convenience of random exploration.

---

## Status panel

The frontend companion converts the reserved display row into a compact status panel.

Depending on the active state, it can show:

- the last seed used;
- `UNSEEDED` for the `-1` sentinel;
- `RECORD` with the number of captured seeds;
- `REPLAY` with the current sequence position;
- `REPLAY EMPTY` when replay is requested without a stored sequence.

The panel is informational only. Seed values still leave the node through the `seed` and `seed_int` outputs.

If the frontend companion is unavailable, the reserved row remains a harmless inert widget and is ignored by the backend.

---

## JLC Resize Multiple Images

**JLC Resize Multiple Images** resizes one through five independent IMAGE inputs with the same controls and aspect-ratio-preserving policy used by **JLC Resize Image**.

- `slot_count` selects the active inputs and individual outputs.
- Each active image is resized from its own source dimensions and keeps its own calculated output geometry.
- `image_1` through `image_5` expose the resized results separately.
- The trailing `batch` output provides a convenience batch. When active results differ in size, later images are normalized to the first active result's geometry before concatenation, following ComfyUI's legacy ImageBatch behavior.
- **Update Visible Slots** applies the requested one-to-five socket layout in the frontend. Inactive backend outputs remain positionally stable and return `None`.

Use the individual outputs when each image should retain its calculated dimensions. Use `batch` only when a downstream node requires one normalized IMAGE batch.

---

## JLC Multi Set/Get

**JLC Multi Set** and **JLC Multi Get** are production-ready virtual nodes for replacing large groups of individual wireless Set/Get nodes with one compact pair. Each row is an independently named channel and may carry a different ComfyUI socket type.

- The nodes start with one visible row and grow automatically when the final available row is used, up to twenty-four channels.
- Disconnecting a used row removes that row when safe and compacts the remaining rows while preserving their names, types, stable identities, and physical links.
- Connected unnamed Set rows receive unique default names such as `channel_1`; Set names remain editable.
- Get rows select from available connected channels. A Set rename propagates to Gets bound to that specific Set row.
- Types are inferred dynamically, so a single pair can carry IMAGE, MASK, CONDITIONING, LATENT, VAE, MODEL, INT, custom objects, and other ComfyUI types.
- Values resolve through ComfyUI's virtual graph-link interfaces before backend execution; no Python or JavaScript runtime value registry is used.
- Subgraph input boundaries are delegated to ComfyUI's native executable-graph resolver rather than being treated as ordinary source nodes.

Channel discovery follows lexical workflow scope. A **JLC Multi Get** can discover Sets in its own graph and in ancestor graphs. A Set that exists only inside a child subgraph is not published upward to a Get in the parent graph.

JLC Multi Get can also select ordinary KJ `SetNode` channels when KJNodes is installed. Ordinary KJ `GetNode` does not currently resolve JLC Multi Set rows. The JLC pair works independently when KJNodes is absent.

---

## JLC Multi Reroute

**JLC Multi Reroute** is a frontend virtual node for reducing visible wiring clutter without hiding connections behind wireless channels. It combines a stack of independent reroute nodes into one compact, automatically managed routing strip.

Each row is a strict passthrough pair:

```text
input 1  →  output 1
input 2  →  output 2
input 3  →  output 3
...
```

Key behavior:

- The node starts with one input/output pair and grows automatically as the final available row is used, up to twenty-four pairs.
- While below the twenty-four-row limit, the node keeps one trailing spare pair ready for the next connection.
- Each row is independent and may resolve to a different ComfyUI socket type.
- A row remains while either its input or output side is connected.
- When both sides of a row become disconnected, that row may be removed and the surviving rows compact upward.
- Stable internal row identities allow middle-row compaction without intentionally redirecting the surviving physical links.
- One output may fan out to multiple downstream consumers just like a conventional reroute.
- Socket and link colors follow ComfyUI's normal resolved-type colors, making mixed MODEL, VAE, CONDITIONING, IMAGE, LATENT, scalar, and custom-object lanes readable at a glance.
- The node is virtual: it does not add a Python execution stage and does not copy tensors simply to provide the reroute.
- Subgraph boundaries are resolved through ComfyUI's native virtual-link execution path.

A row with downstream consumers but no valid upstream source is an invalid execution path. Prompt validation reports the error and the affected Multi Reroute receives ComfyUI's normal red error frame so the problem can be located on the canvas.

Use **JLC Multi Reroute** when you want the workflow wiring to remain visible and traceable, but want repeated reroute nodes organized into one compact block. Use **JLC Multi Set/Get** when hiding long connections completely is more valuable than showing their physical path.

---

## JLC Boolean Logic (Frontend)

**JLC Boolean Logic (Frontend)** is a pure client-side virtual node intended specifically for use with the ComfyUI-Switchboard Group Controller and Node Controller nodes. It replaces the earlier dedicated **JLC Frontend Boolean AND** prototype.

The selectable two-input operations are:

- AND
- OR
- XOR
- NAND
- NOR
- XNOR
- A AND NOT B
- B AND NOT A

The node resolves frontend-readable Boolean values in real time and exposes the calculated result synchronously so Switchboard can update controlled groups before prompt execution. Both inputs must be connected and frontend-resolvable. Disconnected, backend-only, or unresolved inputs fail closed to `false`, including for NAND, NOR, and XNOR.

This node is never submitted to the Python backend. It is not intended as a standalone Boolean logic node for ordinary backend-executed workflow decisions.

---

## JLC Stage Boundary VRAM Cleanup

**JLC Stage Boundary VRAM Cleanup** is an experimental, type-agnostic stage-boundary cleanup passthrough for advanced multi-stage workflows.

It is intended for workflows where one stage has finished using heavy resident model objects and a later stage should proceed with lower VRAM pressure, a different model family, or a different type of intermediate value.

Typical uses include:

```text
LATENT boundary:
    Stage 1 sampling
    ↓
    latent
    ↓
    JLC Stage Boundary VRAM Cleanup
    ↓
    Stage 2 sampling
```

```text
STRING boundary:
    vision / caption model
    ↓
    prompt text
    ↓
    JLC Stage Boundary VRAM Cleanup
    ↓
    downstream text encoder or generation stage
```

```text
IMAGE or other boundary:
    any ordinary ComfyUI value
    ↓
    JLC Stage Boundary VRAM Cleanup
    ↓
    downstream stage
```

The passthrough value is not interpreted or modified. The frontend resolves the wildcard passthrough socket to the live connected datatype, so normal ComfyUI values such as `STRING`, `LATENT`, `IMAGE`, and custom types can act as the stage-boundary dependency.

The node is also **list-aware**. When ComfyUI supplies a gathered list—such as one prompt per tile—the cleanup executes once for the whole gathered stage boundary and the list is returned unchanged rather than triggering cleanup separately for every item.

### Experimental warning

This node is experimental.

It may affect model residency, reload behavior, execution time, and VRAM usage in ways that depend on ComfyUI's current model-management internals.

ComfyUI remains the authority for model lifecycle management. This node should be treated as a best-effort cleanup helper, not as a guaranteed VRAM reset.

Use it only when the upstream heavy model objects are genuinely no longer needed after the boundary.

---

## Targeted model-object cleanup

The optional **model** socket is also type-agnostic so the same cleanup path can target more than a conventional diffusion `MODEL`.

The node can recognize:

- a normal ComfyUI `MODEL` / model patcher directly;
- `CLIP` wrappers that expose ComfyUI's managed `patcher`;
- `VAE` wrappers that expose a managed `patcher`;
- compatible wrapper/dictionary objects that expose an identifiable ComfyUI model patcher.

When **unload_connected_model** is enabled, the node asks ComfyUI to unload the resolved target patcher together with its clones and additional models when the current ComfyUI API supports that operation.

This targeted path is useful when only one completed stage should be released and a broad unload-all operation would be unnecessarily destructive.

If the connected object does not expose a supported managed patcher, the targeted unload is skipped rather than blindly passing an unknown object into ComfyUI's model-management internals.

---

## Cleanup targets

The available cleanup paths are:

- the connected `MODEL`, `CLIP`, `VAE`, or compatible managed patcher, including clones/additional models where supported;
- all currently loaded ComfyUI models, when explicitly requested;
- JLC-managed ControlNet resident cache entries;
- all JLC-managed resident cache entries, when explicitly requested;
- final best-effort Python / backend-aware / CUDA allocator cleanup.

The broad cleanup options are intentionally separate from the targeted path so the narrowest useful cleanup can be selected.

### Main controls

| Input | Purpose |
|---|---|
| `passthrough` | Required wildcard stage-boundary value. Passed through unchanged and may be `STRING`, `LATENT`, `IMAGE`, or another ComfyUI type. |
| `unload_connected_model` | Try to unload the optional connected managed model object, its clones, and additional models. |
| `evict_jlc_controlnet_cache` | Evict JLC-managed ControlNet cache entries. |
| `evict_all_jlc_model_cache` | Evict all JLC-managed resident cache entries. |
| `unload_all_comfy_models` | Ask ComfyUI to unload all resident models. This is the broadest ComfyUI-side cleanup option. |
| `clear_cuda_allocator` | Run final best-effort allocator cleanup after model/cache cleanup. |
| `safe_cleanup` | Use the safer cleanup path when supported by the JLC cache helpers. |
| `all_devices` | Apply targeted connected-model unload across devices when supported by ComfyUI. |
| `verbose` | Print cleanup actions and timing information. |
| `model` | Optional wildcard cleanup target. Accepts `MODEL`, `CLIP`, `VAE`, or a compatible wrapper exposing a ComfyUI-managed patcher. |

### Execution behavior

The node intentionally forces execution whenever it lies on an active graph path because cleanup is a side effect and should not be optimized away solely because the passthrough value is cached.

With list-mapped inputs, the node gathers the incoming passthrough items, performs cleanup once, and returns the gathered value unchanged so downstream list handling can continue normally.

Cleanup order is intentionally narrow-to-broad:

1. targeted connected-model unload, unless **unload_all_comfy_models** is selected;
2. JLC ControlNet cache eviction, unless **evict_all_jlc_model_cache** is selected;
3. optional allocator cleanup.

If an "all" option is selected for a cleanup family, the narrower operation from the same family is skipped rather than performed redundantly first.

### Practical guidance

Use the narrowest cleanup that solves the problem.

A common targeted boundary is:

```text
unload_connected_model = true
clear_cuda_allocator = true
```

Connect the completed stage's `MODEL`, `CLIP`, or `VAE`-compatible object to the optional **model** socket, and use the natural stage result—latent, image, string, or another value—as the passthrough dependency.

If the workflow also uses JLC-managed ControlNet residency, add:

```text
evict_jlc_controlnet_cache = true
```

For a much more aggressive boundary:

```text
unload_all_comfy_models = true
evict_all_jlc_model_cache = true
```

The broad options may force later stages to reload models and therefore increase execution time substantially.

This node cannot guarantee a complete VRAM reset. Active references elsewhere in the graph, ComfyUI internals, backend allocator behavior, and third-party model ownership can all affect what remains resident.

---

## JLC GPU Cooldown

**JLC GPU Cooldown** is a generic, interruptible stage pause in `utils/VRAM`.
It returns the exact passthrough object it receives, preserving object identity
and gathered-list structure. It does not clone or detach tensors, unload models,
clear VRAM, synchronize CUDA, or transfer data between devices.

The node is list-aware and accepts `IMAGE`, `LATENT`, `MODEL`, `STRING`, and
other connected ComfyUI types. One gathered list produces one cooldown wait and
is returned with the same list and item identities. Multiple cooldown nodes may
be used in one workflow to pause different dependency boundaries.

### Modes and defaults

| Input | Default | Behavior |
|---|---:|---|
| `mode` | `combined` | `timer`, `temperature`, or both conditions together. |
| `minimum_wait_seconds` | `30` | Required elapsed time in timer and combined modes. |
| `resume_temperature_c` | `70` | Maximum accepted reading in temperature and combined modes. |
| `stable_seconds` | `5` | Time that successful readings must remain at or below the target. |
| `poll_interval_seconds` | `2` | Delay between telemetry checks. |
| `gpu_index` | `0` | Physical NVML GPU index used by temperature modes. |

Timer mode releases after the minimum wait. Temperature mode ignores the timer
and releases only after the GPU has remained at or below the target for the
stability interval. Combined mode requires both conditions. Elapsed and stable
intervals use a monotonic clock, so system-clock changes do not alter the wait.

These defaults are workflow settings, not hardware safety limits.

### Telemetry, selection, and cancellation

Temperature and combined modes require `nvidia-ml-py`, imported as `pynvml`,
and an NVIDIA GPU visible to NVML. Timer mode does not initialize NVML and works
without that dependency. `gpu_index` selects the physical NVML index; it is not
a CUDA index remapped by `CUDA_VISIBLE_DEVICES`. On a multi-NVIDIA system,
select the physical GPU doing the relevant work.

If the selected index does not exist, the node fails with a configuration error
instead of waiting forever. If NVML initialization, handle access, or a
temperature reading otherwise fails, the node resets the stability interval and
**holds execution while retrying**. It does not convert sensor failure into
successful completion and has no automatic timeout release. ComfyUI
Cancel/Interrupt is checked during the wait, including telemetry retry delays.
Clear separately queued prompts as well if they should not run after the
interrupted prompt.

### Wiring and caching

The connected data path defines the pause boundary. For example:

```text
VAE Decode → JLC GPU Cooldown → Save Image
```

Here saving occurs after the cooldown releases. A model connection by itself
does not prove that a sampler using the model has completed, and a cooldown on
one branch does not automatically synchronize unrelated branches. A terminal
cooldown running in parallel with Save Image likewise does not guarantee that
saving finishes first.

The node deliberately reports itself changed on each prompt so that a required
cooldown is not skipped by ComfyUI's cache. Consequently, consumers downstream
of its passthrough output are also treated as dirty and may rerun even when
their other inputs are unchanged. Put the node only on the path that genuinely
needs the pause.

The frontend displays live progress without serializing that transient status.
The outputs are `passthrough`, `waited_seconds`, and `status`. GPU memory may
remain allocated throughout the pause, and background GPU activity is outside
the node's control. Cooldown can reduce between-stage heat accumulation; it is
not a guarantee against driver failures, hardware freezes, or other instability.

---

## Choosing the Right Utility Node

| Need | Recommended Node |
|---|---|
| Feed the same seed into multiple samplers or stages | JLC Seed Generator |
| Keep the visible seed stable while viewing the last seed actually used | JLC Seed Generator |
| Pass an explicit `-1` intentionally-unseeded sentinel to a compatible downstream consumer | JLC Seed Generator |
| Record and replay repeatable randomized seed trials | JLC Seed Generator |
| Resize up to five images with one shared policy while retaining separate outputs | JLC Resize Multiple Images |
| Replace many individual wireless Set/Get nodes with compact mixed-type channels | JLC Multi Set/Get |
| Keep wires visible while consolidating many independent reroute lanes into one compact block | JLC Multi Reroute |
| Gate Switchboard-controlled groups from two frontend-readable Boolean inputs | JLC Boolean Logic (Frontend) |
| Pass a latent, image, string, or other value across a deliberate stage boundary while trying to free selected managed model objects | JLC Stage Boundary VRAM Cleanup |
| Pause one wired dependency path by time, NVIDIA temperature, or both without changing its value or freeing memory | JLC GPU Cooldown |
| Force a guaranteed complete VRAM reset | Not guaranteed by these nodes; restart ComfyUI if a true reset is required |

---

## Example Workflows

No new showcase workflows are included for these utility additions. Consult workflows in the other documentation sections for examples of the broader node collection in use.

---

## Notes for Advanced Users

### Seed dictionary vs. integer output

The `seed` output is a small dictionary for compatibility with seed-style consumers. The `seed_int` output is a plain integer and is usually the easiest connection for standard sampler seed fields.

### Randomize replay is frontend-assisted

Randomize record/replay is implemented by the JLC frontend companion because the seed must be chosen or substituted before ComfyUI serializes the prompt.

The Python node remains the explicit seed source and preserves the `-1` sentinel contract. The frontend owns the randomize sequence state, stores recorded seeds in node workflow properties, and restores the visible base seed after submission.

### Multi Set/Get and Multi Reroute are frontend virtual nodes

These nodes organize graph connectivity; they are not tensor-processing stages. Their connected outputs are resolved to the real upstream execution path before the prompt reaches the Python backend.

For Multi Set/Get, channel discovery is lexical: a Get can see matching Sets in its own graph and ancestor graph scopes. It does not search downward into child subgraphs.

For Multi Reroute, each visible lane remains an ordinary physical graph path. ComfyUI's native virtual-link resolver is responsible for crossing subgraph input boundaries during executable-graph construction.

### Stage cleanup is not a magic memory eraser

The VRAM cleanup node can request targeted `MODEL` / `CLIP` / `VAE`-compatible patcher unloads, broader ComfyUI or JLC cache eviction, and allocator cleanup, but model residency still depends on ComfyUI internals, active graph references, backend behavior, third-party ownership, and selected options.

### Avoid using cleanup too early

Place the cleanup node only after the upstream model objects are truly no longer needed. The passthrough socket may carry any ordinary ComfyUI value, but that value should represent a real execution boundary. If the graph still needs the targeted model object later, ComfyUI may reload it or the workflow may behave unexpectedly.

### List-aware stage boundaries

The cleanup node uses ComfyUI list mapping deliberately. A gathered list can act as one stage-boundary token, allowing cleanup to run once for the whole upstream stage rather than once per list item.

### Verbose mode

Verbose mode is useful while designing workflows because it prints what cleanup actions were requested and how long the cleanup pass took.
