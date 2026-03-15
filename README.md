# JLC ComfyUI Nodes

<p align="center">
  <img src="assets/icons/jlc-comfyui-nodes_Logo-0512.png" width="120">
</p>

[![ComfyUI Registry](https://img.shields.io/badge/ComfyUI-Registry-blue)](https://registry.comfy.org/packages/jlc-comfyui-nodes)
![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom%20Nodes-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-brightgreen)

Custom workflow-oriented nodes for ComfyUI focused on Flux pipelines, LoRA experimentation, ControlNet utilities, and advanced inpainting/outpainting workflows. Developed by
**J. L. Córdova**.

These nodes focus on improving practical workflows for modern image
generation pipelines, particularly:

- Flux workflows
- LoRA experimentation
- advanced inpainting / outpainting pipelines
- structured ControlNet pipelines

Repository  
https://github.com/Damkohler/jlc-comfyui-nodes

---

## Example Workflows
PNG workflows contain the embedded ComfyUI graph and can be dragged directly into the ComfyUI canvas.

### Basic Inpainting / Outpainting Workflow

<p align="center">
  <img src="assets/workflows/jlc_padded_image_Basic_Infill_Outfill.png" width="900">
</p>

<p align="center">
  <a href="assets/workflows/jlc_padded_image_Basic_Infill_Outfill.png">Download PNG</a> •
  <a href="assets/workflows/jlc_padded_image_Basic_Infill_Outfill.json">Download JSON</a>
</p>

---

### Preferred Inpainting / Outpainting Workflow

<p align="center">
  <img src="assets/workflows/jlc_padded_image_Best_Infill_Outfill.png" width="900">
</p>

<p align="center">
  <a href="assets/workflows/jlc_padded_image_Best_Infill_Outfill.png">Download PNG</a> •
  <a href="assets/workflows/jlc_padded_image_Best_Infill_Outfill.json">Download JSON</a>
</p>

*(Other workflows will be added later)*

---

## Table of Contents

- [Installation](#installation)
- [Nodes Included](#nodes-included)
- [Node Descriptions](#node-descriptions)
- [Design Philosophy](#design-philosophy)
- [Compatibility](#compatibility)
- [License](#license)
- [Attribution](#attribution)
- [Author](#author)
- [Future Plans](#future-plans)
- [Contributions](#contributions)

---

## Installation

### Using git:
Clone this repository into your **ComfyUI `custom_nodes` directory**.

```
ComfyUI/
└── custom_nodes/
```

Then run:

```
git clone https://github.com/Damkohler/jlc-comfyui-nodes.git
```

Restart **ComfyUI** after installation.

The nodes will appear in the standard ComfyUI node menu under their
appropriate workflow categories.

### Install via ComfyUI Manager

1. Open **ComfyUI**
2. Open **Manager**
3. Search for **JLC ComfyUI Nodes**
4. Click **Install**

---

# Nodes Included

| Node | Purpose |
|-----|--------|
| **JLC Padded Image** | Canvas preparation for inpainting and outpainting workflows |
| **JLC Padded Latent** | Combined padded-image + latent + mask conditioning pipeline |
| **JLC ControlNet Apply** | Simplified ControlNet application node |
| **JLC 10 LoRA Loader Stack** | Sequential loader for up to 10 LoRAs |
| **JLC LoRA Loader (Block Weight)** | Multi-slot LoRA loader with block weight control |

---

# Node Descriptions

## JLC Padded Image

A utility node that prepares images for **inpainting or outpainting**
by placing them on a new canvas with a specified aspect ratio and size.

### Features

- Canvas resizing with aspect ratio control
- Image placement using offset controls
- High-quality **Lanczos resampling**
- Automatic outpaint mask generation
- Optional manual mask merging
- Deterministic padding behavior

Designed to work particularly well with inpainting models such as:

```
flux1-fill-dev
```

---

## JLC Padded Latent

A higher-level workflow node that combines:

- padded image preparation
- outpaint mask generation and merge with inpaint masks
- inpaint conditioning

### Outputs

- conditioned positive prompt
- conditioned negative prompt
- latent image
- mask
- image dimensions

This node simplifies building **reusable inpainting pipelines**.

---

## JLC ControlNet Apply

A streamlined node for applying **ControlNet conditioning**
within a generation pipeline.

### Design Goals

- simplified parameter handling
- improved workflow clarity
- compatibility with Flux-based pipelines

This node adapts the built-in **ComfyUI ControlNet application logic**
for cleaner integration into custom workflows.

---

## JLC 10 LoRA Loader Stack

Applies up to **ten LoRA models sequentially** to a base model.

### Features

Each slot includes:

- selectable LoRA file
- independent strength control

Slots operate independently and are applied **in order**.

Empty slots or strengths of zero are automatically skipped.

### Inspiration

Concept inspired by the **LoRA Loader Stack** design by:

https://github.com/rgthree

---

## JLC LoRA Loader (Block Weight)

A LoRA loader with **block weight support**, allowing detailed control
over how LoRA influence is distributed across model layers.

### Features

- multiple LoRA slots
- independent model and CLIP strengths
- per-slot block weight vectors
- sequential LoRA application

This node is adapted from the implementation found in the  
**ComfyUI Inspire Pack** project.

Original project:

https://github.com/ltdrdata/ComfyUI-Inspire-Pack

Released under the **MIT License**.

---

# Design Philosophy

The nodes in this repository follow several guiding principles.

## Workflow clarity

Nodes simplify complex workflows rather than introducing additional
abstraction layers.

## Deterministic behavior

Operations such as padding, mask generation, and LoRA stacking are
implemented to behave predictably and consistently.

## Uniform documentation

Each node includes a standardized header structure containing:

- repository attribution
- node purpose description
- third-party attribution where applicable
- license information

## Metadata manifests

Each node includes a `MANIFEST` block containing metadata such as:

- node name
- version
- author
- description

These manifests support future tooling and repository indexing.

---

# Compatibility

Tested with:

- **ComfyUI**
- **Flux-based models**
- **LoRA-enabled pipelines**

---

# License

This repository is released under the **MIT License**.

Some nodes include adapted logic from other open-source projects.
Those sections retain their original attribution and licensing terms.

---

# Attribution

Some node concepts and implementations were inspired by existing
ComfyUI projects.

### rgthree

LoRA Loader Stack concept

https://github.com/rgthree

### ComfyUI Inspire Pack

LoRA Loader (Block Weight) implementation

https://github.com/ltdrdata/ComfyUI-Inspire-Pack

---

# Author

**J. L. Córdova**

GitHub  
https://github.com/Damkohler

---

# Future Plans

The JLC node collection will continue expanding with nodes focused on:

- advanced inpainting/outpainting workflows
- Flux pipeline utilities
- LoRA experimentation tools
- pipeline orchestration helpers

---

# Contributions

Suggestions and improvements are welcome.

Feel free to open issues or submit pull requests.

---