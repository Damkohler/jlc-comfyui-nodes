# Utility Nodes

This chapter covers the JLC utility-node family:

- [JLC Seed Generator](#jlc-seed-generator)
- [Future Seed Generator Direction: Randomize Replay](#future-seed-generator-direction-randomize-replay)
- [JLC Resize Multiple Images](#jlc-resize-multiple-images)
- [JLC Multi Set/Get](#jlc-multi-setget)
- [JLC Multi Reroute](#jlc-multi-reroute)
- [JLC Boolean Logic (Frontend)](#jlc-boolean-logic-frontend)
- [JLC Stage Boundary VRAM Cleanup](#jlc-stage-boundary-vram-cleanup)
- [Choosing the Right Utility Node](#choosing-the-right-utility-node)
- [Example Workflows](#example-workflows)

These nodes are not image-generation algorithms by themselves. They are workflow-support nodes intended to make larger ComfyUI graphs easier to connect, resize, control, run, repeat, debug, or stage.

---

## JLC Seed Generator

**JLC Seed Generator** is a shared seed source for workflows where multiple samplers or multiple inference stages should use the same seed during one full prompt execution.

Typical use case:

```text
JLC Seed Generator
    ├─ seed_int → KSampler seed, stage 1
    └─ seed_int → KSampler seed, stage 2
```

This is useful when:

- one seed should feed multiple KSamplers;
- stage 1 creates a latent;
- stage 2 partially denoises the same latent;
- you want parameter trials without losing track of the starting seed.

### Outputs

The node returns two seed outputs:

| Output | Purpose |
|---|---|
| `seed` | Small SEED-style dictionary, e.g. `{"seed": 12345}`. |
| `seed_int` | Plain integer seed for nodes that expect an INT seed input. |

In most ordinary ComfyUI seed sockets, `seed_int` is the convenient output.

---

## Stable base seed and last-used display

The node uses ComfyUI's native `control_after_generate` seed behavior, but changes the user-facing display behavior.

A normal seed widget often mutates the visible seed after a queue submission. For example:

```text
seed = 1
control after generate = increment
queue count = 4
```

The visible widget may advance as queued prompts are prepared.

**JLC Seed Generator** instead keeps the visible seed input as the user's stable base seed, while the companion frontend display reports the seed actually used by the executed prompt.

Conceptually:

```text
visible base seed: 1

queued prompt 1 uses seed 1
queued prompt 2 uses seed 2
queued prompt 3 uses seed 3
queued prompt 4 uses seed 4

visible seed input is restored to 1
display reports the last seed actually used
```

The intent is to make parameter trials easier. You can queue several variations, abort, adjust parameters, and still see the original seed you started from.

### Display spacer

The node includes a harmless STRING widget row reserved for the frontend seed panel.

If the frontend script is available, it turns that row into a display panel. If the frontend script is unavailable, it remains an inert separator-like text field and is ignored by the backend.

---

## Future Seed Generator Direction: Randomize Replay

The current seed behavior works cleanly for:

```text
fixed
increment
decrement
```

Those modes are deterministic from the visible base seed.

The intended future improvement is an optional replay mechanism for `randomize` mode.

### Current randomize limitation

In `randomize` mode, ComfyUI may generate a different sequence of random seeds for a queued run. The JLC Seed Generator can display the seed actually used, but it does not yet replay an entire randomized sequence.

### Planned direction

The intended future feature is optional and off by default.

During a queued randomized trial, the node/frontend sequence could record the actual seeds used:

```text
randomize run, count 4:
  seed A
  seed B
  seed C
  seed D
```

On a rerun where the visible input seed has not changed, an optional replay mode could reuse the stored sequence:

```text
replay run, count 4:
  seed A
  seed B
  seed C
  seed D
```

Possible future controls might look like:

```text
randomize_replay: off / record / replay
```

or:

```text
repeat_last_random_sequence: true / false
```

### Why this matters

The long-term goal is repeatable randomized parameter trials.

That means a user could explore randomized seeds across a batch, then rerun the same randomized sequence after changing another parameter, without relying on or reverse-engineering ComfyUI's internal random-number behavior.

This planned feature should apply only to `randomize` mode. It is unnecessary for `fixed`, `increment`, and `decrement`.

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

**JLC Stage Boundary VRAM Cleanup** is an experimental latent-passthrough cleanup node for advanced multi-stage workflows.

It is intended for workflows where one stage uses heavy model objects to produce a latent, then a later stage should run with a different model family or reduced resident-memory pressure.

Typical use case:

```text
Stage 1:
    large base model, inpaint model, or ControlNet stack
    ↓
    latent

Boundary:
    JLC Stage Boundary VRAM Cleanup
    ↓
    same latent passed through

Stage 2:
    different model family or partial denoising pass
```

The node returns the same `LATENT` it receives. Its purpose is side-effect cleanup at a deliberate stage boundary.

### Experimental warning

This node is experimental.

It may affect model residency, reload behavior, execution time, and VRAM usage in ways that depend on ComfyUI's current model-management internals.

ComfyUI remains the authority for model lifecycle management. This node should be treated as a best-effort helper, not as a guaranteed VRAM reset.

Use it only when the graph is structured so that the upstream model objects are no longer needed after the latent passthrough point.

---

## Cleanup targets

The robust targets are:

- a connected ComfyUI `MODEL` object and its clones/additional models;
- all currently loaded ComfyUI models, when explicitly requested;
- JLC-managed ControlNet resident cache entries;
- all JLC-managed resident cache entries, when explicitly requested;
- final best-effort Python/CUDA allocator cleanup.

It is intentionally not a generic CLIP/VAE cleanup node.

### Main controls

| Input | Purpose |
|---|---|
| `latent` | Passthrough latent that triggers the cleanup point. |
| `unload_connected_model` | Try to unload the connected optional MODEL and its clones/additional models. |
| `evict_jlc_controlnet_cache` | Evict JLC-managed ControlNet cache entries. |
| `evict_all_jlc_model_cache` | Evict all JLC-managed resident cache entries. |
| `unload_all_comfy_models` | Ask ComfyUI to unload all resident models. This is the broadest ComfyUI-side cleanup option. |
| `clear_cuda_allocator` | Run final best-effort allocator cleanup. |
| `safe_cleanup` | Use the safer cleanup path when supported by JLC cache helpers. |
| `all_devices` | Apply connected-model unload across devices when supported by ComfyUI. |
| `verbose` | Print cleanup status messages. |
| `model` | Optional connected MODEL to target for unload. |

### Execution behavior

The node intentionally forces execution when it is on an active graph path, because cleanup is a side effect. It should not be optimized away simply because the latent is cached.

### Practical guidance

Use the narrowest cleanup that solves the problem.

Start with:

```text
unload_connected_model = true
clear_cuda_allocator = true
```

Then add broader options only when needed:

```text
evict_jlc_controlnet_cache = true
```

or, for more aggressive cleanup:

```text
unload_all_comfy_models = true
evict_all_jlc_model_cache = true
```

The broad options may cause later nodes to reload models, which can increase execution time.

---

## Choosing the Right Utility Node

| Need | Recommended Node |
|---|---|
| Feed the same seed into multiple samplers or stages | JLC Seed Generator |
| Keep the visible seed stable while viewing the last seed actually used | JLC Seed Generator |
| Prepare for future repeatable randomized seed trials | JLC Seed Generator, with planned randomize replay enhancement |
| Resize up to five images with one shared policy while retaining separate outputs | JLC Resize Multiple Images |
| Replace many individual wireless Set/Get nodes with compact mixed-type channels | JLC Multi Set/Get |
| Keep wires visible while consolidating many independent reroute lanes into one compact block | JLC Multi Reroute |
| Gate Switchboard-controlled groups from two frontend-readable Boolean inputs | JLC Boolean Logic (Frontend) |
| Pass a latent across a deliberate stage boundary while trying to free selected model objects | JLC Stage Boundary VRAM Cleanup |
| Force a guaranteed complete VRAM reset | Not guaranteed by these nodes; restart ComfyUI if a true reset is required |

---

## Example Workflows

No new showcase workflows are included for these utility additions. Consult workflows in the other documentation sections for examples of the broader node collection in use.

---

## Notes for Advanced Users

### Seed dictionary vs. integer output

The `seed` output is a small dictionary for compatibility with seed-style consumers. The `seed_int` output is a plain integer and is usually the easiest connection for standard sampler seed fields.

### Randomize replay is not implemented yet

The randomize replay section documents the planned direction. It is included here so the intent of the current seed-display design is clear, but the replay feature itself is not part of the current implementation.

### Multi Set/Get and Multi Reroute are frontend virtual nodes

These nodes organize graph connectivity; they are not tensor-processing stages. Their connected outputs are resolved to the real upstream execution path before the prompt reaches the Python backend.

For Multi Set/Get, channel discovery is lexical: a Get can see matching Sets in its own graph and ancestor graph scopes. It does not search downward into child subgraphs.

For Multi Reroute, each visible lane remains an ordinary physical graph path. ComfyUI's native virtual-link resolver is responsible for crossing subgraph input boundaries during executable-graph construction.

### Stage cleanup is not a magic memory eraser

The VRAM cleanup node can request targeted unloads and allocator cleanup, but model residency remains dependent on ComfyUI internals, active graph references, backend behavior, and selected options.

### Avoid using cleanup too early

Place the cleanup node only after the upstream model objects are truly no longer needed. If the graph still needs those objects later, ComfyUI may reload them or the workflow may behave unexpectedly.

### Verbose mode

Verbose mode is useful while designing workflows because it prints what cleanup actions were requested and how long the cleanup pass took.
