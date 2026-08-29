/*
 * JLC Dynamic Multi Reroute
 * -------------------------
 *
 * JLC ComfyUI Nodes Collection
 *   This frontend extension is part of the JLC Custom Nodes for ComfyUI
 *   collection developed by J. L. Córdova.
 *
 * Repository:
 *   https://github.com/Damkohler/jlc-comfyui-nodes
 *
 * Node Purpose:
 *   Provides a compact dynamic bank of independent reroute pairs for reducing
 *   workflow wiring clutter without hiding the actual graph connections.
 *
 * Frontend Execution Model:
 *   - The companion Python module supplies one initial wildcard input/output
 *     pair plus ComfyUI registration metadata.
 *   - This JavaScript file is the primary implementation.
 *   - The node is marked as a virtual node.
 *   - Each connected output row exposes its matching input link to ComfyUI's
 *     native virtual-node resolver before prompt submission, including across
 *     subgraph input boundaries.
 *   - No Python execution stage, value registry, tensor copy, or retained
 *     runtime state is used by the normal path.
 *
 * Dynamic Row Semantics:
 *   - Begins with one visible wildcard input/output pair.
 *   - Supports up to twenty-four independent pairs.
 *   - Connecting either side of the final available row creates one spare row.
 *   - A row remains while either its input or any output consumer is linked.
 *   - Once both sides are completely disconnected, that row may disappear from
 *     any position and later rows compact upward.
 *   - Row IDs remain stable while their visible slot numbers are free to change.
 *   - One trailing spare row is retained whenever capacity remains.
 *
 * Type / Color Behavior:
 *   - Every row resolves its own ComfyUI datatype independently.
 *   - An upstream concrete type is authoritative when present; otherwise a
 *     concrete downstream target can establish the row type while wiring
 *     backwards.
 *   - Socket types are updated to the resolved datatype, which gives standard
 *     ComfyUI socket colors.
 *   - Attached links are explicitly recolored from ComfyUI's normal
 *     LGraphCanvas.link_type_colors table when available.
 *   - Incompatible downstream links are removed if a loaded or transient graph
 *     state bypasses normal LiteGraph connection validation.
 *
 * Attribution & License:
 *   Concept and implementation by J. L. Córdova
 *   with development assistance from ChatGPT (OpenAI).
 *
 *   Dynamic-row identity and compaction strategy follows the proven design of
 *   JLC Multi Set/Get. Reroute type/color behavior was informed by public
 *   ComfyUI / rgthree reroute behavior while remaining independently
 *   implemented for this node.
 *
 *   Copyright (c) 2026 J. L. Córdova
 *
 *   Released under the MIT License.
 */

import { app } from "../../scripts/app.js";

const REROUTE_CLASS = "JLC_DynamicMultiReroute";
const REROUTE_CLASS_NORMALIZED = "jlcdynamicmultireroute";

const ROWS_KEY = "jlc_dynamic_reroute_rows";
const FORMAT_KEY = "jlc_dynamic_reroute_format";
const FORMAT_VERSION = 2;

const MIN_ROWS = 1;
const MAX_ROWS = 24;

const DEFAULT_NODE_WIDTH = 220;
const MIN_NODE_WIDTH = 90;
const ROW_HEIGHT = 28;
const ROW_GAP = 2;
const ROW_TOP = 4;
const ROW_BOTTOM = 4;
const BLANK_LABEL = "\u00a0";

const INSTALL_FLAG = "__jlcDynamicMultiRerouteInstalled";

let fallbackRowCounter = 0;

function normalizeClassIdentifier(value) {
    return String(value ?? "")
        .replace(/[^a-zA-Z0-9]/g, "")
        .toLowerCase();
}

function isRerouteClass(...values) {
    return values.some(
        (value) => normalizeClassIdentifier(value) === REROUTE_CLASS_NORMALIZED
    );
}

function newRowId() {
    if (typeof crypto?.randomUUID === "function") {
        return `r_${crypto.randomUUID().replaceAll("-", "")}`;
    }
    fallbackRowCounter += 1;
    return `r_${Date.now().toString(36)}_${fallbackRowCounter.toString(36)}`;
}

function newRow(automatic = false) {
    return {
        id: newRowId(),
        type: "*",
        automatic,
    };
}

function normalizeRows(node) {
    node.properties ??= {};
    let rows = node.properties[ROWS_KEY];
    if (!Array.isArray(rows)) {
        rows = [];
        node.properties[ROWS_KEY] = rows;
    }

    if (rows.length > MAX_ROWS) rows.splice(MAX_ROWS);

    const seen = new Set();
    for (let index = 0; index < rows.length; index += 1) {
        let row = rows[index];
        if (!row || typeof row !== "object") {
            row = {};
            rows[index] = row;
        }

        let id = typeof row.id === "string" && row.id ? row.id : newRowId();
        if (seen.has(id)) id = newRowId();
        seen.add(id);

        row.id = id;
        row.type = normalizedType(row.type);
        row.automatic = row.automatic === true;
    }

    while (rows.length < MIN_ROWS) rows.push(newRow(false));

    node.properties[FORMAT_KEY] = FORMAT_VERSION;
    return rows;
}

function rowSlotName(index) {
    return `value_${index + 1}`;
}

function normalizedType(value) {
    return typeof value === "string" && value ? value : "*";
}

function concreteType(value) {
    const type = normalizedType(value);
    return type === "*" ? null : type;
}

function hasOutputLinks(output) {
    return Array.isArray(output?.links) && output.links.length > 0;
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

function graphCandidates(graph) {
    const result = [];
    const seen = new Set();
    const add = (candidate) => {
        if (!candidate || seen.has(candidate)) return;
        seen.add(candidate);
        result.push(candidate);
    };

    add(graph);
    add(graph?.rootGraph);
    add(graph?._rootGraph);
    add(app.graph);
    add(app.rootGraph);

    return result;
}

function rowPitch() {
    return ROW_HEIGHT + ROW_GAP;
}

function desiredNodeHeight(node) {
    return (
        ROW_TOP +
        normalizeRows(node).length * rowPitch() -
        ROW_GAP +
        ROW_BOTTOM
    );
}

function rowCenterY(index) {
    return ROW_TOP + index * rowPitch() + ROW_HEIGHT / 2;
}

function rowIndexFromLinkInfo(node, linkInfo) {
    if (!linkInfo || node?.id == null) return -1;

    const nodeId = String(node.id);
    let index = -1;

    if (String(linkInfo.target_id) === nodeId) {
        index = Number(linkInfo.target_slot);
    } else if (String(linkInfo.origin_id) === nodeId) {
        index = Number(linkInfo.origin_slot);
    }

    return Number.isInteger(index) && index >= 0 ? index : -1;
}

function rowIdFromLinkInfo(node, linkInfo) {
    const index = rowIndexFromLinkInfo(node, linkInfo);
    return index < 0 ? null : normalizeRows(node)[index]?.id ?? null;
}

function rowIndexById(node, rowId) {
    if (!rowId) return -1;
    return normalizeRows(node).findIndex((row) => row.id === rowId);
}

function rowIsUsed(node, index) {
    return (
        node.inputs?.[index]?.link != null ||
        hasOutputLinks(node.outputs?.[index])
    );
}

function rowIsEmpty(node, index) {
    return !rowIsUsed(node, index);
}

function setRowSocketType(node, index, type) {
    const row = normalizeRows(node)[index];
    if (!row) return;

    const resolvedType = normalizedType(type);
    row.type = resolvedType;

    const input = node.inputs?.[index];
    if (input) {
        input.name = rowSlotName(index);
        input.type = resolvedType;
        input.label = BLANK_LABEL;
    }

    const output = node.outputs?.[index];
    if (output) {
        output.name = rowSlotName(index);
        output.type = resolvedType;
        output.label = BLANK_LABEL;
    }
}

function ensureSlots(node) {
    const rows = normalizeRows(node);

    node.inputs ??= [];
    node.outputs ??= [];

    while (node.inputs.length < rows.length) {
        node.addInput(rowSlotName(node.inputs.length), "*");
    }
    while (node.outputs.length < rows.length) {
        node.addOutput(rowSlotName(node.outputs.length), "*");
    }

    while (
        node.inputs.length > rows.length &&
        node.inputs.at(-1)?.link == null
    ) {
        node.removeInput(node.inputs.length - 1);
    }

    while (
        node.outputs.length > rows.length &&
        !hasOutputLinks(node.outputs.at(-1))
    ) {
        node.removeOutput(node.outputs.length - 1);
    }

    rows.forEach((row, index) => {
        const observedType =
            concreteType(node.inputs?.[index]?.type) ??
            concreteType(node.outputs?.[index]?.type) ??
            row.type ??
            "*";
        setRowSocketType(node, index, observedType);
    });
}

function applyNodeSize(node, requestedSize = node.size, fitHeight = true) {
    const requestedWidth = Number(requestedSize?.[0] ?? node.size?.[0]);
    const width = Math.max(
        MIN_NODE_WIDTH,
        Number.isFinite(requestedWidth) ? requestedWidth : DEFAULT_NODE_WIDTH
    );
    const height = desiredNodeHeight(node);

    node.size ??= [width, height];
    node.size[0] = width;
    node.size[1] = fitHeight ? height : Math.max(height, Number(requestedSize?.[1]) || height);
    node.min_size = [MIN_NODE_WIDTH, height];
    node.setDirtyCanvas?.(true, true);

    return node.size;
}

function linkedInputType(node, index) {
    const input = node.inputs?.[index];
    if (!input || input.link == null || !node.graph) return null;

    const link = linkById(node.graph, input.link);
    if (!link) return null;

    const source = nodeByIdEverywhere(node.graph, link.origin_id);
    return (
        concreteType(source?.outputs?.[link.origin_slot]?.type) ??
        concreteType(link.type)
    );
}

function linkedOutputTypes(node, index) {
    const output = node.outputs?.[index];
    if (!hasOutputLinks(output) || !node.graph) return [];

    const result = [];
    for (const linkId of output.links) {
        const link = linkById(node.graph, linkId);
        if (!link) continue;

        const target = nodeByIdEverywhere(node.graph, link.target_id);
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

function colorAttachedLinks(node, index, type) {
    if (!node.graph) return;

    const inputLinkId = node.inputs?.[index]?.link;
    if (inputLinkId != null) colorLink(node.graph, inputLinkId, type);

    for (const linkId of node.outputs?.[index]?.links ?? []) {
        colorLink(node.graph, linkId, type);
    }
}

function validateOutputLinks(node, index, resolvedType) {
    if (!node.graph || !resolvedType || resolvedType === "*") return;

    const output = node.outputs?.[index];
    if (!hasOutputLinks(output)) return;

    for (const linkId of [...output.links]) {
        const link = linkById(node.graph, linkId);
        if (!link) continue;

        const target = nodeByIdEverywhere(node.graph, link.target_id);
        const targetType = normalizedType(
            target?.inputs?.[link.target_slot]?.type ?? link.type
        );

        if (!typesOverlap(resolvedType, targetType)) {
            console.warn(
                `[JLC Dynamic Multi Reroute] Disconnecting incompatible ` +
                    `downstream link on row ${index + 1}: ${resolvedType} -> ${targetType}`
            );
            node.graph.removeLink?.(linkId);
        }
    }
}

function refreshRowType(node, index) {
    const row = normalizeRows(node)[index];
    if (!row) return;

    const inputType = linkedInputType(node, index);
    const outputTypes = linkedOutputTypes(node, index);
    const outputType = outputTypes[0]?.type ?? null;

    // Upstream is authoritative when present. This matches normal dataflow and
    // lets a loaded graph shed incompatible consumers deterministically.
    const resolvedType = inputType ?? outputType ?? "*";

    setRowSocketType(node, index, resolvedType);
    validateOutputLinks(node, index, resolvedType);
    colorAttachedLinks(node, index, resolvedType);
}

function refreshAllTypes(node) {
    if (
        node.__jlcRefreshingTypes ||
        node.__jlcStructuralUpdate ||
        !node.graph
    ) {
        return;
    }

    node.__jlcRefreshingTypes = true;
    try {
        normalizeRows(node).forEach((_row, index) => refreshRowType(node, index));
        node.setDirtyCanvas?.(true, true);
    } finally {
        node.__jlcRefreshingTypes = false;
    }
}

function rowCanBeRemoved(node, index) {
    return (
        normalizeRows(node)[index] != null &&
        node.inputs?.[index]?.link == null &&
        !hasOutputLinks(node.outputs?.[index])
    );
}

function removeSocketRowAt(node, index, rowId) {
    const rows = normalizeRows(node);
    if (rows.length <= MIN_ROWS) return false;
    if (rows[index]?.id !== rowId) return false;
    if (!rowCanBeRemoved(node, index)) return false;

    // Removing the matching input/output at the same live index lets LiteGraph
    // reindex every surviving later connection. Only after the socket pair has
    // been removed do we delete the stable row record itself.
    node.removeInput(index);
    node.removeOutput(index);

    const authoritativeRows = node.properties[ROWS_KEY];
    const currentIndex = authoritativeRows.findIndex((row) => row.id === rowId);
    if (currentIndex < 0) return false;

    authoritativeRows.splice(currentIndex, 1);
    return true;
}

function compactEmptyRows(node) {
    let rows = normalizeRows(node);
    if (rows.length <= MIN_ROWS) return false;

    const usedIndices = [];
    rows.forEach((_row, index) => {
        if (rowIsUsed(node, index)) usedIndices.push(index);
    });

    // Preserve one already-existing trailing empty row when possible so the
    // spare row retains its stable identity instead of being recreated during
    // unrelated connection changes.
    let spareId = null;
    if (usedIndices.length === 0) {
        spareId = rows[0]?.id ?? null;
    } else {
        const lastUsedIndex = usedIndices.at(-1);
        const lastIndex = rows.length - 1;
        if (lastIndex > lastUsedIndex && rowIsEmpty(node, lastIndex)) {
            spareId = rows[lastIndex]?.id ?? null;
        }
    }

    let changed = false;

    // Remove from bottom to top so each still-pending row index remains valid.
    for (let index = rows.length - 1; index >= 0; index -= 1) {
        rows = normalizeRows(node);
        const row = rows[index];
        if (!row || row.id === spareId) continue;
        if (!rowCanBeRemoved(node, index)) continue;

        if (removeSocketRowAt(node, index, row.id)) changed = true;
    }

    return changed;
}

function appendTrailingSpare(node) {
    const rows = normalizeRows(node);
    if (rows.length >= MAX_ROWS) return false;

    const lastIndex = rows.length - 1;
    if (!rowIsUsed(node, lastIndex)) return false;

    rows.push(newRow(true));
    return true;
}

function ensureExactlyOneUsefulSpare(node) {
    let changed = compactEmptyRows(node);
    changed = appendTrailingSpare(node) || changed;
    return changed;
}

function maintainRows(node) {
    if (node.__jlcStructuralUpdate || !node.graph) return;
    node.__jlcStructuralUpdate = true;

    const graph = node.graph;
    let transactionStarted = false;
    let structureChanged = false;

    try {
        if (typeof graph.beforeChange === "function") {
            graph.beforeChange();
            transactionStarted = true;
        }

        normalizeRows(node);
        ensureSlots(node);
        structureChanged = ensureExactlyOneUsefulSpare(node) || structureChanged;
        normalizeRows(node);
        ensureSlots(node);
        applyNodeSize(node, node.size, true);
    } finally {
        try {
            if (transactionStarted && typeof graph.afterChange === "function") {
                graph.afterChange();
            }
        } finally {
            node.__jlcStructuralUpdate = false;
        }
    }

    refreshAllTypes(node);
    refreshRerouteValidationError(node);

    if (structureChanged) {
        node.setDirtyCanvas?.(true, true);
    }
}

function scheduleMaintainRows(node) {
    if (node.__jlcStructuralUpdate) return;

    if (node.__jlcMaintainTimer != null) {
        clearTimeout(node.__jlcMaintainTimer);
    }

    node.__jlcMaintainTimer = setTimeout(() => {
        node.__jlcMaintainTimer = null;
        if (!node.graph) return;

        try {
            maintainRows(node);
        } catch (error) {
            console.error(
                "[JLC Dynamic Multi Reroute] Structural maintenance failed",
                error
            );
        }
    }, 0);
}

function setRerouteValidationError(node, hasError) {
    const next = hasError === true;
    const stateChanged = node.__jlcRerouteValidationError !== next;
    const oldRenderState = node.has_errors;

    if (next) {
        if (node.__jlcRerouteValidationError !== true) {
            // Only clear ComfyUI's error flag later if this node owns the flag.
            // This avoids masking an unrelated frontend/backend validation error.
            node.__jlcRerouteValidationOwnsErrorFlag = node.has_errors !== true;
        }
        node.__jlcRerouteValidationError = true;
        node.has_errors = true;
    } else {
        node.__jlcRerouteValidationError = false;
        if (node.__jlcRerouteValidationOwnsErrorFlag === true) {
            node.has_errors = false;
        }
        delete node.__jlcRerouteValidationOwnsErrorFlag;
    }

    if (oldRenderState !== node.has_errors) {
        node.graph?.trigger?.("node:property:changed", {
            type: "node:property:changed",
            nodeId: node.id,
            property: "has_errors",
            oldValue: oldRenderState,
            newValue: node.has_errors,
        });
    }

    if (stateChanged || oldRenderState !== node.has_errors) {
        node.setDirtyCanvas?.(true, true);
    }
}

function refreshRerouteValidationError(node) {
    // Like JLC Multi Get, do not paint an intentionally half-wired node red
    // while the user is still constructing it. The red frame becomes active
    // after a real prompt-resolution error, then remains until every
    // downstream-used row has a valid upstream source again.
    if (node.__jlcRerouteValidationError !== true) return;

    const hasUnresolvedRelevantRow = normalizeRows(node).some((_row, index) => {
        return (
            hasOutputLinks(node.outputs?.[index]) &&
            sourceForRow(node, index, false) == null
        );
    });

    setRerouteValidationError(node, hasUnresolvedRelevantRow);
}

function sourceForRow(node, slot, throwOnError) {
    const row = normalizeRows(node)[slot];

    const fail = (message) => {
        if (throwOnError) {
            setRerouteValidationError(node, true);
            throw new Error(message);
        }
        return null;
    };

    if (!row) {
        return fail(`JLC Dynamic Multi Reroute row ${slot + 1} does not exist.`);
    }

    const input = node.inputs?.[slot];
    if (!input || input.link == null) {
        return fail(
            `JLC Dynamic Multi Reroute row ${slot + 1} has a downstream ` +
                `connection but no upstream source.`
        );
    }

    const link = linkById(node.graph, input.link);
    if (!link) {
        return fail(
            `JLC Dynamic Multi Reroute row ${slot + 1} has a missing source link.`
        );
    }

    // Deliberately do NOT resolve link.origin_id here. Inside a ComfyUI
    // subgraph, a link entering from a subgraph input uses the special IO
    // endpoint as its origin rather than an ordinary LGraphNode. ComfyUI's
    // executable-graph compiler knows how to walk that boundary when a
    // virtual node returns its input link via getInputLink(). Treating the IO
    // endpoint as a normal node is what caused the false "missing source node"
    // error in the earlier prototype.
    return { link };
}

function installNode(node) {
    if (node[INSTALL_FLAG]) return;
    node[INSTALL_FLAG] = true;

    node.properties ??= {};
    const hadStoredRows = Array.isArray(node.properties[ROWS_KEY]);
    node.properties[FORMAT_KEY] = FORMAT_VERSION;
    node.properties[ROWS_KEY] ??= [newRow(false)];

    node.isVirtualNode = true;
    node.serialize_widgets = false;
    node.resizable = true;

    const original = {
        onAdded: node.onAdded?.bind(node),
        onConfigure: node.onConfigure?.bind(node),
        onSerialize: node.onSerialize?.bind(node),
        onConnectionsChange: node.onConnectionsChange?.bind(node),
        onResize: node.onResize?.bind(node),
        computeSize: node.computeSize?.bind(node),
        getConnectionPos: node.getConnectionPos?.bind(node),
        onDrawForeground: node.onDrawForeground?.bind(node),
    };

    node.computeSize = function () {
        return [MIN_NODE_WIDTH, desiredNodeHeight(this)];
    };

    node.onResize = function (size) {
        const result = original.onResize?.(...arguments);
        applyNodeSize(this, size ?? this.size, true);
        return result;
    };

    node.getConnectionPos = function (isInput, slot, out) {
        const row = normalizeRows(this)[slot];
        if (!row) {
            return (
                original.getConnectionPos?.(isInput, slot, out) ??
                new Float32Array([this.pos[0], this.pos[1]])
            );
        }

        const result = out ?? new Float32Array(2);
        result[0] = this.pos[0] + (isInput ? 0 : this.size[0]);
        result[1] = this.pos[1] + rowCenterY(slot);
        return result;
    };

    node.onDrawForeground = function (ctx) {
        original.onDrawForeground?.(...arguments);

        // A very subtle separator makes independent reroute lanes readable at
        // a glance without adding labels or consuming horizontal space.
        const rows = normalizeRows(this);
        if (rows.length <= 1) return;

        ctx.save();
        ctx.strokeStyle = "rgba(255,255,255,0.07)";
        ctx.lineWidth = 1;
        for (let index = 1; index < rows.length; index += 1) {
            const y = ROW_TOP + index * rowPitch() - ROW_GAP / 2;
            ctx.beginPath();
            ctx.moveTo(10, y);
            ctx.lineTo(Math.max(10, this.size[0] - 10), y);
            ctx.stroke();
        }
        ctx.restore();
    };

    node.onAdded = function () {
        const result = original.onAdded?.(...arguments);
        setTimeout(() => {
            if (!this.graph) return;
            maintainRows(this);
        }, 0);
        return result;
    };

    node.onConfigure = function () {
        const result = original.onConfigure?.(...arguments);

        this.__jlcStructuralUpdate = true;
        try {
            normalizeRows(this);
            ensureSlots(this);
            applyNodeSize(this, this.size, true);
        } finally {
            this.__jlcStructuralUpdate = false;
        }

        setTimeout(() => {
            if (!this.graph) return;
            maintainRows(this);
        }, 0);

        return result;
    };

    node.onSerialize = function (serialized) {
        const result = original.onSerialize?.(serialized);

        serialized.properties ??= {};
        serialized.properties[FORMAT_KEY] = FORMAT_VERSION;
        serialized.properties[ROWS_KEY] = normalizeRows(this).map((row) => ({
            id: row.id,
            type: row.type,
            automatic: row.automatic,
        }));

        return result;
    };

    node.onConnectionsChange = function (slotType, slot, isConnected, linkInfo) {
        // Capture the stable row identity before the original callback. The
        // visible numeric slot may change later if a different empty row is
        // compacted out.
        const affectedRowId = this.__jlcStructuralUpdate
            ? null
            : rowIdFromLinkInfo(this, linkInfo);

        const result = original.onConnectionsChange?.(...arguments);

        if (app.configuringGraph) return result;
        if (this.__jlcStructuralUpdate) return result;

        // Re-find by stable identity after LiteGraph completes the connection
        // mutation. The maintenance pass may legally move this row upward.
        if (affectedRowId) {
            setTimeout(() => {
                if (!this.graph || this.__jlcStructuralUpdate) return;
                const index = rowIndexById(this, affectedRowId);
                if (index >= 0) refreshRowType(this, index);
                refreshRerouteValidationError(this);
            }, 0);
        }

        scheduleMaintainRows(this);
        return result;
    };

    // Match ComfyUI's native reroute contract: expose the corresponding input
    // link and let the executable-graph compiler resolve it. This is important
    // for subgraphs because the compiler's resolveInput() contains the logic
    // that crosses a subgraph input boundary to the outer graph.
    node.getInputLink = function (slot) {
        const source = sourceForRow(this, slot, true);
        refreshRerouteValidationError(this);
        return source.link;
    };

    normalizeRows(node);
    ensureSlots(node);

    // New nodes start at the compact default width. Saved workflows retain the
    // width LiteGraph restores from their serialized node size.
    if (!hadStoredRows) {
        node.size ??= [DEFAULT_NODE_WIDTH, desiredNodeHeight(node)];
        node.size[0] = DEFAULT_NODE_WIDTH;
    }
    applyNodeSize(node, node.size, true);
}

app.registerExtension({
    name: "JLC.DynamicMultiReroute",

    beforeRegisterNodeDef(nodeType, nodeData) {
        if (
            !isRerouteClass(
                nodeData?.name,
                nodeData?.display_name,
                nodeType?.comfyClass,
                nodeType?.title
            ) ||
            nodeType.prototype.__jlcDynamicMultiReroutePrototypeWrapped
        ) {
            return;
        }

        nodeType.prototype.__jlcDynamicMultiReroutePrototypeWrapped = true;
        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {
            const result = originalOnNodeCreated?.apply(this, arguments);
            try {
                installNode(this);
            } catch (error) {
                console.error(
                    "[JLC Dynamic Multi Reroute] Failed to initialize node",
                    error
                );
            }
            return result;
        };
    },

    // Fallback for frontend builds where the instance callback is more reliable
    // than beforeRegisterNodeDef. INSTALL_FLAG prevents double initialization.
    nodeCreated(node) {
        if (
            !isRerouteClass(
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
                "[JLC Dynamic Multi Reroute] Failed to initialize node",
                error
            );
        }
    },
});

console.info("[JLC Dynamic Multi Reroute] frontend extension loaded");
