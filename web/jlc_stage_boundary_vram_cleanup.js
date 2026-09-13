/*
 * JLC Stage Boundary VRAM Cleanup — Dynamic Model/CLIP + Passthrough Frontend
 * --------------------------------------------------------------------------
 *
 * Connection layout:
 *   1. model / clip / vae   (targeted object to unload)
 *   2. passthrough          (execution dependency, returned unchanged)
 *
 * Both backend sockets are wildcards. The frontend resolves their visible
 * concrete ComfyUI datatypes from live connections so normal socket/link
 * colors are preserved. The passthrough output mirrors the passthrough input.
 *
 * This revision intentionally uses the physical input order above rather than
 * preserving the earlier experimental latent-first layout.
 *
 * Copyright (c) 2026 J. L. Córdova
 * Released under the MIT License.
 */

import { app } from "../../scripts/app.js";

const CLEANUP_CLASS = "JLC_StageBoundaryVRAMCleanup";
const CLEANUP_CLASS_NORMALIZED = "jlcstageboundaryvramcleanup";
const INSTALL_FLAG = "__jlcStageBoundaryDynamicV2Installed";

function normalizeClassIdentifier(value) {
    return String(value ?? "")
        .replace(/[^a-zA-Z0-9]/g, "")
        .toLowerCase();
}

function isCleanupClass(...values) {
    return values.some(
        (value) => normalizeClassIdentifier(value) === CLEANUP_CLASS_NORMALIZED
    );
}

function normalizedType(value) {
    return typeof value === "string" && value ? value : "*";
}

function concreteType(value) {
    const type = normalizedType(value);
    return type === "*" ? null : type;
}

function linkById(graph, id) {
    if (!graph || id == null) return null;
    if (typeof graph.getLink === "function") return graph.getLink(id) ?? null;

    const links = graph._links ?? graph.links;
    if (links instanceof Map) return links.get(id) ?? null;
    return links?.[id] ?? null;
}

function nodeById(graph, id) {
    if (!graph || id == null) return null;
    return (
        graph.getNodeById?.(id) ??
        graph.getNodeById?.(Number(id)) ??
        graph._nodes?.find((node) => String(node.id) === String(id)) ??
        graph.nodes?.find?.((node) => String(node.id) === String(id)) ??
        null
    );
}

function inputIndex(node, name) {
    return node.inputs?.findIndex((input) => input?.name === name) ?? -1;
}

function modelInputIndex(node) {
    return inputIndex(node, "model");
}

function passthroughInputIndex(node) {
    return inputIndex(node, "passthrough");
}

function passthroughOutputIndex(node) {
    if (!Array.isArray(node.outputs) || node.outputs.length === 0) return -1;
    const named = node.outputs.findIndex((output) => output?.name === "passthrough");
    return named >= 0 ? named : 0;
}

function reorderInputs(node) {
    if (!Array.isArray(node.inputs)) return;

    const modelIndex = modelInputIndex(node);
    const passthroughIndex = passthroughInputIndex(node);
    if (modelIndex < 0 || passthroughIndex < 0) return;
    if (modelIndex === 0 && passthroughIndex === 1) return;

    const modelInput = node.inputs[modelIndex];
    const passthroughInput = node.inputs[passthroughIndex];
    const rest = node.inputs.filter(
        (_input, index) => index !== modelIndex && index !== passthroughIndex
    );

    node.inputs = [modelInput, passthroughInput, ...rest];

    // Preserve already-connected links when loading an earlier experimental
    // graph by making each live link's target_slot follow the reordered array.
    if (node.graph) {
        node.inputs.forEach((input, index) => {
            if (input?.link == null) return;
            const link = linkById(node.graph, input.link);
            if (link) link.target_slot = index;
        });
    }

    node.setDirtyCanvas?.(true, true);
}

function linkedInputType(node, index) {
    const input = node.inputs?.[index];
    if (!input || input.link == null || !node.graph) return null;

    const link = linkById(node.graph, input.link);
    if (!link) return null;

    const source = nodeById(node.graph, link.origin_id);
    return (
        concreteType(source?.outputs?.[link.origin_slot]?.type) ??
        concreteType(link.type)
    );
}

function linkedDownstreamTypes(node) {
    const outputIndex = passthroughOutputIndex(node);
    const output = node.outputs?.[outputIndex];
    if (!output || !Array.isArray(output.links) || !node.graph) return [];

    const result = [];
    for (const linkId of output.links) {
        const link = linkById(node.graph, linkId);
        if (!link) continue;

        const target = nodeById(node.graph, link.target_id);
        const type =
            concreteType(target?.inputs?.[link.target_slot]?.type) ??
            concreteType(link.type);
        if (type) result.push({ linkId, type });
    }
    return result;
}

function typesOverlap(left, right) {
    if (!left || !right || left === "*" || right === "*") return true;

    const leftTypes = new Set(
        String(left)
            .split(",")
            .map((type) => type.trim())
            .filter(Boolean)
    );

    return String(right)
        .split(",")
        .map((type) => type.trim())
        .filter(Boolean)
        .some((type) => leftTypes.has(type));
}

function linkColorForType(type) {
    if (!type || type === "*") return null;

    const globalCanvas = globalThis.LGraphCanvas;
    const liteGraphCanvas = globalThis.LiteGraph?.LGraphCanvas;
    const colors =
        globalCanvas?.link_type_colors ??
        liteGraphCanvas?.link_type_colors ??
        null;

    return colors?.[type] ?? null;
}

function colorLink(graph, linkId, type) {
    const link = linkById(graph, linkId);
    if (!link) return;

    const color = linkColorForType(type);
    if (color) {
        link.color = color;
    } else if (Object.prototype.hasOwnProperty.call(link, "color")) {
        delete link.color;
    }
}

function setModelInputType(node, type) {
    const index = modelInputIndex(node);
    const input = node.inputs?.[index];
    if (!input) return;

    const resolved = normalizedType(type);
    input.type = resolved;
    input.label = "model / clip / vae";

    if (node.graph && input.link != null) {
        colorLink(node.graph, input.link, resolved);
    }
}

function setPassthroughType(node, type) {
    const resolved = normalizedType(type);
    const inputIndexValue = passthroughInputIndex(node);
    const outputIndex = passthroughOutputIndex(node);

    const input = node.inputs?.[inputIndexValue];
    if (input) {
        input.type = resolved;
        input.label = "passthrough";
    }

    const output = node.outputs?.[outputIndex];
    if (output) {
        output.type = resolved;
        output.label = "passthrough";
    }

    if (node.graph) {
        if (input?.link != null) colorLink(node.graph, input.link, resolved);
        for (const linkId of output?.links ?? []) {
            colorLink(node.graph, linkId, resolved);
        }
    }
}

function validatePassthroughConsumers(node, resolvedType) {
    if (!node.graph || !resolvedType || resolvedType === "*") return;

    for (const { linkId, type } of linkedDownstreamTypes(node)) {
        if (typesOverlap(resolvedType, type)) continue;

        console.warn(
            `[JLC Stage Boundary VRAM Cleanup] Disconnecting incompatible ` +
                `passthrough consumer: ${resolvedType} -> ${type}`
        );
        node.graph.removeLink?.(linkId);
    }
}

function refreshTypes(node) {
    if (!node.graph || node.__jlcStageBoundaryRefreshingV2) return;

    node.__jlcStageBoundaryRefreshingV2 = true;
    try {
        reorderInputs(node);

        const modelType = linkedInputType(node, modelInputIndex(node)) ?? "*";
        setModelInputType(node, modelType);

        const passthroughUpstream = linkedInputType(
            node,
            passthroughInputIndex(node)
        );
        const passthroughDownstream = linkedDownstreamTypes(node)[0]?.type ?? null;
        const passthroughType =
            passthroughUpstream ?? passthroughDownstream ?? "*";

        setPassthroughType(node, passthroughType);
        validatePassthroughConsumers(node, passthroughType);
    } finally {
        node.__jlcStageBoundaryRefreshingV2 = false;
    }
}

function scheduleRefresh(node) {
    if (node.__jlcStageBoundaryRefreshTimerV2 != null) {
        clearTimeout(node.__jlcStageBoundaryRefreshTimerV2);
    }

    node.__jlcStageBoundaryRefreshTimerV2 = setTimeout(() => {
        node.__jlcStageBoundaryRefreshTimerV2 = null;
        if (!node.graph) return;
        try {
            refreshTypes(node);
        } catch (error) {
            console.error(
                "[JLC Stage Boundary VRAM Cleanup] Type/order refresh failed",
                error
            );
        }
    }, 0);
}

function installNode(node) {
    if (node[INSTALL_FLAG]) return;
    node[INSTALL_FLAG] = true;

    const original = {
        onAdded: node.onAdded?.bind(node),
        onConfigure: node.onConfigure?.bind(node),
        onConnectionsChange: node.onConnectionsChange?.bind(node),
    };

    // Reorder immediately for newly created unconnected nodes.
    reorderInputs(node);
    setModelInputType(node, "*");
    setPassthroughType(node, "*");

    node.onAdded = function () {
        const result = original.onAdded?.(...arguments);
        reorderInputs(this);
        scheduleRefresh(this);
        return result;
    };

    node.onConfigure = function () {
        const result = original.onConfigure?.(...arguments);
        reorderInputs(this);
        scheduleRefresh(this);
        return result;
    };

    node.onConnectionsChange = function () {
        const result = original.onConnectionsChange?.(...arguments);
        if (!app.configuringGraph) scheduleRefresh(this);
        return result;
    };

    scheduleRefresh(node);
}

app.registerExtension({
    name: "JLC.StageBoundaryVRAMCleanup.DynamicV2",

    beforeRegisterNodeDef(nodeType, nodeData) {
        if (
            !isCleanupClass(
                nodeData?.name,
                nodeData?.display_name,
                nodeType?.comfyClass,
                nodeType?.title
            ) ||
            nodeType.prototype.__jlcStageBoundaryDynamicV2PrototypeWrapped
        ) {
            return;
        }

        nodeType.prototype.__jlcStageBoundaryDynamicV2PrototypeWrapped = true;
        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {
            const result = originalOnNodeCreated?.apply(this, arguments);
            try {
                installNode(this);
            } catch (error) {
                console.error(
                    "[JLC Stage Boundary VRAM Cleanup] Failed to initialize V2 frontend",
                    error
                );
            }
            return result;
        };
    },

    nodeCreated(node) {
        if (
            !isCleanupClass(
                node?.comfyClass,
                node?.constructor?.comfyClass,
                node?.type,
                node?.title
            )
        ) {
            return;
        }

        try {
            installNode(node);
        } catch (error) {
            console.error(
                "[JLC Stage Boundary VRAM Cleanup] Failed to initialize V2 frontend",
                error
            );
        }
    },
});

console.info(
    "[JLC Stage Boundary VRAM Cleanup] dynamic model/clip + passthrough frontend loaded"
);
