/*
 * JLC GPU Cooldown — Dynamic Passthrough and Live Status Frontend
 * ----------------------------------------------------------------
 *
 * Resolves the wildcard passthrough socket to the connected ComfyUI datatype
 * and displays backend cooldown progress without serializing transient status.
 * The backend returns the original value unchanged; this frontend only manages
 * socket presentation and status display.
 *
 * Copyright (c) 2026 J. L. Córdova
 * Released under the MIT License.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const COOLDOWN_CLASS = "JLC_GPUCooldown";
const COOLDOWN_CLASS_NORMALIZED = normalizeClassIdentifier(COOLDOWN_CLASS);
const INSTALL_FLAG = "__jlcGPUCooldownDynamicV2Installed";
const STATUS_LISTENER_FLAG = "__jlcGPUCooldownStatusListenerInstalled";

function normalizeClassIdentifier(value) {
    return String(value ?? "")
        .replace(/[^a-zA-Z0-9]/g, "")
        .toLowerCase();
}

function isCooldownClass(...values) {
    return values.some(
        (value) => normalizeClassIdentifier(value) === COOLDOWN_CLASS_NORMALIZED
    );
}

function firstText(value) {
    if (Array.isArray(value)) return firstText(value[0]);
    if (value === undefined || value === null) return null;
    return String(value);
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

function passthroughInputIndex(node) {
    return inputIndex(node, "passthrough");
}

function passthroughOutputIndex(node) {
    if (!Array.isArray(node.outputs) || node.outputs.length === 0) return -1;
    const named = node.outputs.findIndex((output) => output?.name === "passthrough");
    return named >= 0 ? named : 0;
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
            `[JLC GPU Cooldown] Disconnecting incompatible ` +
                `passthrough consumer: ${resolvedType} -> ${type}`
        );
        node.graph.removeLink?.(linkId);
    }
}

function refreshTypes(node) {
    if (!node.graph || node.__jlcGPUCooldownRefreshingV2) return;

    node.__jlcGPUCooldownRefreshingV2 = true;
    try {


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
        node.__jlcGPUCooldownRefreshingV2 = false;
    }
}

function scheduleRefresh(node) {
    if (node.__jlcGPUCooldownRefreshTimerV2 != null) {
        clearTimeout(node.__jlcGPUCooldownRefreshTimerV2);
    }

    node.__jlcGPUCooldownRefreshTimerV2 = setTimeout(() => {
        node.__jlcGPUCooldownRefreshTimerV2 = null;
        if (!node.graph) return;
        try {
            refreshTypes(node);
        } catch (error) {
            console.error(
                "[JLC GPU Cooldown] Type/order refresh failed",
                error
            );
        }
    }, 0);
}

function setStatus(node, text) {
    if (!node || text == null) return;
    if (!node.__cooldownStatus) {
        node.__cooldownStatus = node.addWidget(
            "text",
            "cooldown_status",
            "",
            () => {},
            { serialize: false }
        );
        node.__cooldownStatus.options.serialize = false;
    }
    node.__cooldownStatus.value = String(text);
    node.setDirtyCanvas?.(true, true);
}

function installNode(node) {
    if (node[INSTALL_FLAG]) return;
    node[INSTALL_FLAG] = true;

    const original = {
        onAdded: node.onAdded?.bind(node),
        onConfigure: node.onConfigure?.bind(node),
        onConnectionsChange: node.onConnectionsChange?.bind(node),
        onExecuted: node.onExecuted?.bind(node),
        onRemoved: node.onRemoved?.bind(node),
    };

    // Reorder immediately for newly created unconnected nodes.
    setPassthroughType(node, "*");

    node.onAdded = function () {
        const result = original.onAdded?.(...arguments);
        scheduleRefresh(this);
        return result;
    };

    node.onConfigure = function () {
        const result = original.onConfigure?.(...arguments);
        scheduleRefresh(this);
        return result;
    };

    node.onConnectionsChange = function () {
        const result = original.onConnectionsChange?.(...arguments);
        if (!app.configuringGraph) scheduleRefresh(this);
        return result;
    };

    node.onExecuted = function (message) {
        const result = original.onExecuted?.(...arguments);
        const text = firstText(message?.text ?? message?.ui?.text);
        if (text != null) setStatus(this, text);
        return result;
    };

    node.onRemoved = function () {
        if (this.__jlcGPUCooldownRefreshTimerV2 != null) {
            clearTimeout(this.__jlcGPUCooldownRefreshTimerV2);
            this.__jlcGPUCooldownRefreshTimerV2 = null;
        }
        return original.onRemoved?.(...arguments);
    };

    scheduleRefresh(node);
}

app.registerExtension({
    name: "JLC.GPUCooldown.DynamicV2",

    beforeRegisterNodeDef(nodeType, nodeData) {
        if (
            !isCooldownClass(
                nodeData?.name,
                nodeData?.display_name,
                nodeType?.comfyClass,
                nodeType?.title
            ) ||
            nodeType.prototype.__jlcGPUCooldownDynamicV2PrototypeWrapped
        ) {
            return;
        }

        nodeType.prototype.__jlcGPUCooldownDynamicV2PrototypeWrapped = true;
        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {
            const result = originalOnNodeCreated?.apply(this, arguments);
            try {
                installNode(this);
            } catch (error) {
                console.error(
                    "[JLC GPU Cooldown] Failed to initialize V2 frontend",
                    error
                );
            }
            return result;
        };
    },

    nodeCreated(node) {
        if (
            !isCooldownClass(
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
                "[JLC GPU Cooldown] Failed to initialize V2 frontend",
                error
            );
        }
    },
});

console.info(
    "[JLC GPU Cooldown] dynamic passthrough and status frontend loaded"
);

if (!api[STATUS_LISTENER_FLAG]) {
    api[STATUS_LISTENER_FLAG] = true;
    api.addEventListener("jlc.gpu_cooldown", ({ detail }) => {
        const node = nodeById(app.graph, detail?.node);
        if (!node || !isCooldownClass(node.comfyClass, node.type)) return;
        setStatus(node, detail?.text);
    });
}
