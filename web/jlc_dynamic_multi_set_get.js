/*
 * JLC Multi Set/Get
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
 *   Implements the primary behavior for the JLC Multi Set and
 *   JLC Multi Get virtual-node pair.
 *
 *   The nodes provide compact, independently named wireless channels for
 *   reducing long or repetitive workflow connections. A Multi Set row accepts
 *   an upstream value and publishes a channel name; a Multi Get row selects a
 *   published channel and exposes the original upstream value downstream.
 *
 * Frontend Execution Model:
 *   - This JavaScript file is the primary implementation of the node family.
 *   - The companion Python module supplies ComfyUI registration declarations,
 *     one initial wildcard socket, metadata, and stateless fallback methods.
 *   - The nodes are marked as virtual nodes.
 *   - Connected outputs resolve to the real upstream graph link before prompt
 *     submission through ComfyUI's virtual-node interfaces.
 *   - Values are never transported through a JavaScript or Python runtime
 *     registry, avoiding stale cross-prompt state and execution-order coupling.
 *
 * Dynamic Row Semantics:
 *   - Both nodes begin with one visible row and support up to twenty-four rows.
 *   - Connecting the final available row creates a new trailing row.
 *   - Disconnecting a used row removes that specific row when safe and compacts
 *     surviving rows upward while preserving their stable row identities.
 *   - Multi-consumer Get outputs remain until their final link is removed.
 *   - Set rows remain while either the input or passthrough output is linked.
 *   - At least one visible row is retained for an empty or minimal node.
 *
 * Responsive Presentation:
 *   - Full width shows channel widgets plus normal socket-side labels.
 *   - Compact width keeps the widgets and suppresses redundant socket labels.
 *   - Micro width keeps the widgets visible at minimum width; only the socket
 *     columns and compact channel controls consume horizontal space.
 *   - Width changes are presentation-only and fully reversible.
 *
 * Channel Naming and Type Behavior:
 *   - Connected unnamed Set rows receive graph-local default channel names.
 *   - Set channel names remain user-editable and are kept unique in scope.
 *   - Get rows use dynamically generated channel choices rather than arbitrary
 *     free-form names.
 *   - Get selections bind to stable Set-row identities so Set renames propagate
 *     without silently redirecting a channel.
 *   - Wildcard sockets infer and display the resolved ComfyUI data type.
 *
 * Compatibility Boundary:
 *   - The JLC Multi Set/Get pair is independently implemented and does not
 *     import or require KJNodes.
 *   - JLC Multi Get can recognize ordinary KJ SetNode channels when KJNodes is
 *     installed.
 *   - Ordinary KJ GetNode does not resolve JLC Multi Set rows because KJ's Get
 *     lookup accepts only its own single SetNode contract.
 *
 * Attribution & License:
 *   Concept and implementation by J. L. Córdova
 *   with development assistance from ChatGPT (OpenAI).
 *
 *   Inspired by ComfyUI's public frontend-extension and virtual-node model:
 *   https://github.com/comfyanonymous/ComfyUI
 *
 *   The multi-channel Set/Get workflow concept also draws practical design
 *   inspiration from KJNodes while remaining independently implemented.
 *
 *   Copyright (c) 2026 J. L. Córdova
 *
 *   Released under the MIT License.
 */

import { app } from "../../scripts/app.js";

const SET_CLASS = "JLC_DynamicMultiSet";
const GET_CLASS = "JLC_DynamicMultiGet";
const SET_CLASS_NORMALIZED = "jlcdynamicmultiset";
const GET_CLASS_NORMALIZED = "jlcdynamicmultiget";

function normalizeClassIdentifier(value) {
    return String(value ?? "")
        .replace(/[^a-zA-Z0-9]/g, "")
        .toLowerCase();
}

function kindFromIdentifiers(...values) {
    for (const value of values) {
        const normalized = normalizeClassIdentifier(value);
        if (normalized === SET_CLASS_NORMALIZED) return "set";
        if (normalized === GET_CLASS_NORMALIZED) return "get";
    }
    return null;
}

const ROWS_KEY = "jlc_dynamic_rows";
const KIND_KEY = "jlc_dynamic_kind";
const FORMAT_KEY = "jlc_dynamic_format";
const FORMAT_VERSION = 4;

const MIN_ROWS = 1;
const MAX_ROWS = 24;
const DEFAULT_NODE_WIDTH = 360;

// Three-stage responsive presentation:
//   full    -> widgets + normal socket-side labels
//   compact -> widgets remain visible; redundant socket labels disappear
//   micro   -> widgets remain visible at minimum width with very small gutters
//
// Hysteresis keeps the presentation from flickering when the resize handle
// hovers near a boundary.
const FULL_LAYOUT_ENTER_WIDTH = 340;
const FULL_LAYOUT_EXIT_WIDTH = 315;
const MICRO_LAYOUT_ENTER_WIDTH = 205;
const MICRO_LAYOUT_EXIT_WIDTH = 225;
const MIN_NODE_WIDTH = 150;

const MIN_WIDGET_WIDTH = 88;

// In Full mode the gutter is calculated from the actual visible output labels,
// bounded by these limits. This avoids reserving a large fixed empty region for
// hypothetical long names while still protecting real labels such as
// CONDITIONING.
const SET_FULL_GUTTER_MIN = 82;
const SET_FULL_GUTTER_MAX = 178;
const GET_FULL_GUTTER_MIN = 64;
const GET_FULL_GUTTER_MAX = 126;

// In Compact/Micro modes the output-side text is hidden, so only a small pin
// safety gutter is needed.
const SET_COMPACT_GUTTER = 38;
const GET_COMPACT_GUTTER = 30;
const SET_MICRO_GUTTER = 44;
const GET_MICRO_GUTTER = 30;

const ROW_HEIGHT = 26;
const ROW_GAP = 2;
const WIDGET_TOP = 4;
const BLANK_LABEL = "\u00a0";

const WIDGET_FLAG = "__jlcDynamicChannelWidget";
const INSTALL_FLAG = "__jlcDynamicMultiInstalled";

let fallbackRowCounter = 0;

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
        name: "",
        type: "*",
        automatic,
        nameAutomatic: false,
        sourceKey: "",
    };
}

function normalizeRows(node) {
    node.properties ??= {};
    let rows = node.properties[ROWS_KEY];
    if (!Array.isArray(rows)) {
        rows = [];
        node.properties[ROWS_KEY] = rows;
    }

    const seen = new Set();
    if (rows.length > MAX_ROWS) rows.splice(MAX_ROWS);

    for (let index = 0; index < rows.length; index += 1) {
        let candidate = rows[index];
        if (!candidate || typeof candidate !== "object") {
            candidate = {};
            rows[index] = candidate;
        }

        let id =
            typeof candidate?.id === "string" && candidate.id
                ? candidate.id
                : newRowId();
        if (seen.has(id)) id = newRowId();
        seen.add(id);

        candidate.id = id;
        candidate.name =
            candidate.name == null ? "" : String(candidate.name);
        candidate.type =
            typeof candidate.type === "string" && candidate.type
                ? candidate.type
                : "*";
        candidate.automatic = candidate.automatic === true;
        candidate.nameAutomatic = candidate.nameAutomatic === true;
        candidate.sourceKey =
            typeof candidate.sourceKey === "string" ? candidate.sourceKey : "";
    }

    while (rows.length < MIN_ROWS) rows.push(newRow(false));

    node.properties[FORMAT_KEY] = FORMAT_VERSION;
    return rows;
}

function isJlcSet(node) {
    return node?.properties?.[KIND_KEY] === "set";
}

function isJlcGet(node) {
    return node?.properties?.[KIND_KEY] === "get";
}

function rowSlotName(index) {
    return `value_${index + 1}`;
}

function normalizedType(type) {
    return typeof type === "string" && type ? type : "*";
}

function hasOutputLinks(output) {
    return Array.isArray(output?.links) && output.links.length > 0;
}

function linkById(graph, id) {
    if (!graph || id == null) return null;
    if (typeof graph.getLink === "function") return graph.getLink(id) ?? null;
    const links = graph._links ?? graph.links;
    return links instanceof Map ? links.get(id) ?? null : links?.[id] ?? null;
}

function nodeById(graph, id) {
    if (!graph || id == null) return null;
    return (
        graph.getNodeById?.(id) ??
        graph.getNodeById?.(Number(id)) ??
        graph._nodes?.find((node) => String(node.id) === String(id)) ??
        null
    );
}

function rootGraph(graph) {
    return graph?.rootGraph ?? graph?._rootGraph ?? app.rootGraph ?? app.graph ?? graph;
}

function childGraph(node) {
    return node?.subgraph ?? node?._subgraph ?? null;
}

function allGraphs(root) {
    const result = [];
    const pending = root ? [root] : [];
    const seen = new Set();

    while (pending.length) {
        const graph = pending.pop();
        if (!graph || seen.has(graph)) continue;
        seen.add(graph);
        result.push(graph);

        for (const node of graph._nodes ?? graph.nodes ?? []) {
            const child = childGraph(node);
            if (child && !seen.has(child)) pending.push(child);
        }

        const registry = graph._subgraphs ?? graph.subgraphs;
        if (registry?.values) {
            for (const child of registry.values()) {
                if (child && !seen.has(child)) pending.push(child);
            }
        }
    }

    return result;
}

function parentGraph(graph) {
    const root = rootGraph(graph);
    if (!root || graph === root) return null;

    for (const candidate of allGraphs(root)) {
        for (const node of candidate._nodes ?? candidate.nodes ?? []) {
            if (childGraph(node) === graph) return candidate;
        }
    }
    return null;
}

function lexicalScope(graph) {
    const result = [];
    const seen = new Set();
    let current = graph;

    while (current && !seen.has(current)) {
        result.push(current);
        seen.add(current);
        current = parentGraph(current);
    }

    return result;
}

function getRowWidget(node, rowId) {
    return node.widgets?.find(
        (widget) =>
            widget?.[WIDGET_FLAG] === true && widget.__jlcRowId === rowId
    );
}

function syncNames(node) {
    for (const row of normalizeRows(node)) {
        const widget = getRowWidget(node, row.id);
        if (!widget) continue;

        if (isJlcSet(node)) {
            row.name = widget.value == null ? "" : String(widget.value).trim();
            continue;
        }

        row.sourceKey =
            widget.value == null ? "" : String(widget.value);
        const descriptor = channelDescriptorByKey(
            node.graph,
            row.sourceKey,
            false
        );
        if (descriptor) row.name = descriptor.name;
    }
}

function setRowType(node, index, type) {
    const row = normalizeRows(node)[index];
    if (!row) return;

    const valueType = normalizedType(type);
    row.type = valueType;

    const input = node.inputs?.[index];
    if (input) {
        input.name = rowSlotName(index);
        input.type = valueType;
        input.label = BLANK_LABEL;
    }

    const displayName = row.name.trim();
    const fallbackLabel = isJlcSet(node)
        ? `Set ${index + 1}`
        : `Get ${index + 1}`;
    const socketLabel = displayName || fallbackLabel;

    // The Set channel name is already shown inside the row widget and on the
    // passthrough output. Suppress the redundant left-side input label so it
    // cannot collide with the widget when LiteGraph draws sockets at its own
    // fixed vertical slot spacing.
    if (input) {
        input.label = BLANK_LABEL;
    }

    const output = node.outputs?.[index];
    if (output) {
        output.name = rowSlotName(index);
        output.type = valueType;

        const presentationMode =
            node.__jlcResponsiveMode ??
            responsiveModeForWidth(node, node.size?.[0]);

        if (presentationMode !== "full") {
            // Compact and Micro modes deliberately suppress redundant
            // socket-side text. Channel identity remains visible in the widget
            // and socket color still communicates the resolved type.
            output.label = BLANK_LABEL;
        } else {
            // Full mode preserves the established presentation:
            //   Set -> channel name + type
            //   Get -> resolved type only
            output.label = isJlcGet(node)
                ? valueType === "*"
                    ? `Get ${index + 1}`
                    : valueType
                : valueType === "*"
                  ? socketLabel
                  : `${socketLabel} · ${valueType}`;
        }
    }

    const widget = getRowWidget(node, row.id);
    if (widget) {
        widget.label = isJlcSet(node)
            ? `Set ${index + 1} name`
            : `Get ${index + 1} source`;
        widget.tooltip = isJlcSet(node)
            ? `Shared channel name for row ${index + 1}. Resolved type: ${valueType}.`
            : `Selected Set channel for row ${index + 1}. Resolved type: ${valueType}.`;
    }
}

function ensureSlots(node) {
    const rows = normalizeRows(node);

    if (isJlcSet(node)) {
        node.inputs ??= [];
        while (node.inputs.length < rows.length) {
            const index = node.inputs.length;
            node.addInput(rowSlotName(index), "*");
        }
        while (
            node.inputs.length > rows.length &&
            node.inputs.at(-1)?.link == null
        ) {
            node.removeInput(node.inputs.length - 1);
        }
    }

    node.outputs ??= [];
    while (node.outputs.length < rows.length) {
        const index = node.outputs.length;
        node.addOutput(rowSlotName(index), "*");
    }
    while (
        node.outputs.length > rows.length &&
        !hasOutputLinks(node.outputs.at(-1))
    ) {
        node.removeOutput(node.outputs.length - 1);
    }

    rows.forEach((row, index) => {
        const observedType =
            (node.inputs?.[index]?.type &&
                node.inputs[index].type !== "*" &&
                node.inputs[index].type) ||
            (node.outputs?.[index]?.type &&
                node.outputs[index].type !== "*" &&
                node.outputs[index].type) ||
            row.type ||
            "*";
        setRowType(node, index, observedType);
    });
}

function rowPitch() {
    return ROW_HEIGHT + ROW_GAP;
}

function desiredNodeHeight(node) {
    return WIDGET_TOP + normalizeRows(node).length * rowPitch() + 4;
}


function responsiveModeForWidth(node, width = node.size?.[0]) {
    const numericWidth = Number(width);
    const nodeWidth = Number.isFinite(numericWidth)
        ? numericWidth
        : DEFAULT_NODE_WIDTH;
    const previous = node.__jlcResponsiveMode;

    // Hysteresis: once in a mode, require a little travel before crossing back
    // through the same threshold.
    if (previous === "full" && nodeWidth >= FULL_LAYOUT_EXIT_WIDTH) {
        return "full";
    }
    if (previous === "micro" && nodeWidth < MICRO_LAYOUT_EXIT_WIDTH) {
        return "micro";
    }

    if (nodeWidth >= FULL_LAYOUT_ENTER_WIDTH) return "full";
    if (nodeWidth < MICRO_LAYOUT_ENTER_WIDTH) return "micro";
    return "compact";
}

function estimatedCanvasTextWidth(text) {
    // LiteGraph's normal node font is close enough to ~7 px/character for the
    // purpose of reserving label space. Clamp through the mode-specific gutter
    // bounds below so this estimate can never dominate the layout.
    return String(text ?? "").length * 7;
}

function fullOutputLabelForRow(node, row, index) {
    const valueType = normalizedType(row?.type);
    const displayName = String(row?.name ?? "").trim();
    const socketLabel =
        displayName || (isJlcSet(node) ? `Set ${index + 1}` : `Get ${index + 1}`);

    if (isJlcGet(node)) {
        return valueType === "*" ? `Get ${index + 1}` : valueType;
    }

    return valueType === "*"
        ? socketLabel
        : `${socketLabel} · ${valueType}`;
}

function fullWidgetGutter(node) {
    let widest = 0;
    normalizeRows(node).forEach((row, index) => {
        widest = Math.max(
            widest,
            estimatedCanvasTextWidth(fullOutputLabelForRow(node, row, index))
        );
    });

    // Include output-pin/margin breathing room in addition to the text itself.
    const estimated = widest + 30;
    if (isJlcSet(node)) {
        return Math.max(
            SET_FULL_GUTTER_MIN,
            Math.min(SET_FULL_GUTTER_MAX, estimated)
        );
    }
    return Math.max(
        GET_FULL_GUTTER_MIN,
        Math.min(GET_FULL_GUTTER_MAX, estimated)
    );
}

function widgetGutterForMode(node, mode) {
    if (mode === "full") return fullWidgetGutter(node);
    if (mode === "micro") {
        return isJlcSet(node) ? SET_MICRO_GUTTER : GET_MICRO_GUTTER;
    }
    return isJlcSet(node) ? SET_COMPACT_GUTTER : GET_COMPACT_GUTTER;
}

function applyWidgetPresentation(node, mode, width) {
    for (const widget of node.widgets ?? []) {
        if (widget?.[WIDGET_FLAG] !== true) continue;

        // The responsive implementation intentionally keeps widgets
        // alive and visible in every width state.
        widget.hidden = false;
        widget.width = widgetWidthForNode(node, width, mode);
    }
}

function widgetWidthForNode(
    node,
    width = node.size?.[0],
    mode = node.__jlcResponsiveMode ?? responsiveModeForWidth(node, width)
) {
    const nodeWidth = Math.max(
        MIN_NODE_WIDTH,
        Number(width) || DEFAULT_NODE_WIDTH
    );
    const gutter = widgetGutterForMode(node, mode);
    return Math.max(MIN_WIDGET_WIDTH, nodeWidth - gutter);
}

function applyResponsiveLayout(node, requestedSize = node.size, fitHeight = false) {
    const desiredHeight = desiredNodeHeight(node);
    const requestedWidth = Number(requestedSize?.[0] ?? node.size?.[0]);
    const requestedHeight = Number(requestedSize?.[1] ?? node.size?.[1]);
    const width = Math.max(
        MIN_NODE_WIDTH,
        Number.isFinite(requestedWidth) ? requestedWidth : DEFAULT_NODE_WIDTH
    );
    const height = fitHeight
        ? desiredHeight
        : Math.max(
              desiredHeight,
              Number.isFinite(requestedHeight)
                  ? requestedHeight
                  : desiredHeight
          );

    node.size ??= [width, height];
    node.size[0] = width;
    node.size[1] = height;
    node.min_size = [MIN_NODE_WIDTH, desiredHeight];

    const mode = responsiveModeForWidth(node, width);
    const modeChanged = node.__jlcResponsiveMode !== mode;
    node.__jlcResponsiveMode = mode;

    applyWidgetPresentation(node, mode, width);

    // Refresh labels/widths only; graph structure, channel bindings, widget
    // values, and serialization state are untouched by presentation changes.
    normalizeRows(node).forEach((row, index) => {
        setRowType(node, index, row.type);
    });

    if (modeChanged) {
        node.setDirtyCanvas?.(true, true);
    } else {
        node.setDirtyCanvas?.(true, false);
    }
    return node.size;
}

function resizeNode(node) {
    applyResponsiveLayout(node, node.size, true);
}

function removeRowWidgets(node) {
    for (let index = (node.widgets?.length ?? 0) - 1; index >= 0; index -= 1) {
        const widget = node.widgets[index];
        if (widget?.[WIDGET_FLAG] !== true) continue;
        widget.onRemove?.();
        node.widgets.splice(index, 1);
    }
}

function kjSetName(node) {
    if (node?.type !== "SetNode") return null;
    const name = node.widgets?.[0]?.value;
    return name == null ? "" : String(name).trim();
}

function channelKey(kind, _scopeDepth, node, row = null) {
    if (kind === "KJ") {
        return `kj:${String(node.id)}`;
    }
    return `jlc:${String(node.id)}:${row.id}`;
}

function setChannelDescriptors(graph, readyOnly = true) {
    const descriptors = [];

    lexicalScope(graph).forEach((scopedGraph, scopeDepth) => {
        for (const node of scopedGraph._nodes ?? scopedGraph.nodes ?? []) {
            const kjName = kjSetName(node);
            if (kjName) {
                const ready = node.inputs?.[0]?.link != null;
                if (!readyOnly || ready) {
                    descriptors.push({
                        kind: "KJ",
                        graph: scopedGraph,
                        node,
                        slot: 0,
                        rowIndex: 0,
                        rowId: null,
                        name: kjName,
                        type: normalizedType(node.inputs?.[0]?.type),
                        ready,
                        key: channelKey("KJ", scopeDepth, node),
                    });
                }
            }

            if (!isJlcSet(node)) continue;
            syncNames(node);
            normalizeRows(node).forEach((row, rowIndex) => {
                if (!row.name) return;
                const ready = node.inputs?.[rowIndex]?.link != null;
                if (readyOnly && !ready) return;
                descriptors.push({
                    kind: "JLC",
                    graph: scopedGraph,
                    node,
                    slot: rowIndex,
                    rowIndex,
                    rowId: row.id,
                    name: row.name,
                    type: normalizedType(
                        node.inputs?.[rowIndex]?.type ?? row.type
                    ),
                    ready,
                    key: channelKey("JLC", scopeDepth, node, row),
                });
            });
        }
    });

    return descriptors.sort((left, right) => {
        const byName = left.name.localeCompare(right.name);
        return byName || left.key.localeCompare(right.key);
    });
}

function channelDescriptorByKey(graph, key, readyOnly = true) {
    if (!graph || !key) return null;
    return (
        setChannelDescriptors(graph, readyOnly).find(
            (descriptor) => descriptor.key === key
        ) ?? null
    );
}

function legacyDescriptorByName(graph, name, readyOnly = true) {
    if (!graph || !name) return null;
    const matches = setChannelDescriptors(graph, readyOnly).filter(
        (descriptor) => descriptor.name === name
    );
    return matches.length === 1 ? matches[0] : null;
}

function bindLegacyGetRow(node, row) {
    if (!isJlcGet(node) || row.sourceKey || !row.name) return;
    const descriptor = legacyDescriptorByName(node.graph, row.name, true);
    if (descriptor) row.sourceKey = descriptor.key;
}

function usedChannelNames(graph, excludedNode = null, excludedRowId = null) {
    const used = new Set();
    for (const descriptor of setChannelDescriptors(graph, false)) {
        if (
            descriptor.node === excludedNode &&
            descriptor.rowId === excludedRowId
        ) {
            continue;
        }
        used.add(descriptor.name.toLocaleLowerCase());
    }
    return used;
}

function uniqueChannelName(
    graph,
    requestedName,
    excludedNode = null,
    excludedRowId = null
) {
    const base = String(requestedName ?? "").trim();
    if (!base) return "";

    const used = usedChannelNames(graph, excludedNode, excludedRowId);
    if (!used.has(base.toLocaleLowerCase())) return base;

    let suffix = 2;
    while (used.has(`${base}_${suffix}`.toLocaleLowerCase())) suffix += 1;
    return `${base}_${suffix}`;
}

function nextDefaultChannelName(node, rowId) {
    const used = usedChannelNames(node.graph, node, rowId);
    let number = 1;
    while (used.has(`channel_${number}`.toLocaleLowerCase())) number += 1;
    return `channel_${number}`;
}

function commitSetRowName(node, rowId, requestedName, generated = false) {
    if (!isJlcSet(node)) return "";
    const rows = normalizeRows(node);
    const index = rows.findIndex((row) => row.id === rowId);
    if (index < 0) return "";

    const row = rows[index];
    const connected = node.inputs?.[index]?.link != null;
    let desired = String(requestedName ?? "").trim();
    let automaticName = generated;

    if (!desired && connected) {
        desired = nextDefaultChannelName(node, row.id);
        automaticName = true;
    }

    row.name = uniqueChannelName(node.graph, desired, node, row.id);
    row.nameAutomatic = Boolean(row.name && automaticName);

    const widget = getRowWidget(node, row.id);
    if (widget && widget.value !== row.name) widget.value = row.name;

    setRowType(node, index, row.type);
    return row.name;
}

function ensureReadySetRowName(node, rowId) {
    if (!isJlcSet(node) || !rowId) return false;
    const rows = normalizeRows(node);
    const index = rows.findIndex((row) => row.id === rowId);
    if (index < 0 || node.inputs?.[index]?.link == null) return false;

    const row = rows[index];
    const before = row.name;
    commitSetRowName(node, row.id, row.name, !row.name);
    return row.name !== before;
}

function getChannelOptionValues(node, rowId) {
    const row = normalizeRows(node).find((candidate) => candidate.id === rowId);
    const values = [
        "",
        ...setChannelDescriptors(node.graph, true).map(
            (descriptor) => descriptor.key
        ),
    ];

    if (row?.sourceKey && !values.includes(row.sourceKey)) {
        values.push(row.sourceKey);
    }
    return values;
}

function getChannelOptionLabel(node, rowId, value) {
    if (!value) return "Select Set channel";
    const descriptor = channelDescriptorByKey(node.graph, value, false);
    if (descriptor) return descriptor.name;

    const row = normalizeRows(node).find((candidate) => candidate.id === rowId);
    return row?.name
        ? `Unavailable: ${row.name}`
        : "Unavailable Set channel";
}

function getComboOptions(node, rowId) {
    const options = {
        getOptionLabel: (value) => getChannelOptionLabel(node, rowId, value),
    };
    Object.defineProperty(options, "values", {
        get: () => getChannelOptionValues(node, rowId),
        enumerable: true,
        configurable: true,
    });
    return options;
}

function configureWidgetSize(node, widget) {
    const originalComputeSize = widget.computeSize?.bind(widget);
    widget.computeSize = function (width) {
        const available = widgetWidthForNode(node, node.size?.[0] ?? width);
        const base = originalComputeSize?.(available) ?? [available, ROW_HEIGHT];
        return [
            Math.max(
                MIN_WIDGET_WIDTH,
                Math.min(Number(base?.[0]) || available, available)
            ),
            ROW_HEIGHT,
        ];
    };
    widget.width = widgetWidthForNode(node);
}

function rebuildWidgets(node) {
    const wasRebuilding = node.__jlcRebuilding === true;
    node.__jlcRebuilding = true;

    try {
        removeRowWidgets(node);

        normalizeRows(node).forEach((row, index) => {
            const isGet = isJlcGet(node);
            if (isGet) bindLegacyGetRow(node, row);

            const widget = node.addWidget(
                isGet ? "combo" : "text",
                isGet
                    ? `Get ${index + 1} source`
                    : `Set ${index + 1} name`,
                isGet ? row.sourceKey || "" : row.name,
                (value) => {
                    if (node.__jlcRebuilding) return;
                    widget.value = value == null ? "" : value;
                    const current = normalizeRows(node).find(
                        (candidate) => candidate.id === row.id
                    );
                    if (!current) return;

                    if (isGet) {
                        current.sourceKey = value == null ? "" : String(value);
                        const descriptor = channelDescriptorByKey(
                            node.graph,
                            current.sourceKey,
                            false
                        );
                        current.name = descriptor?.name ?? "";
                        current.nameAutomatic = false;
                        refreshGetTypes(node);
                    } else {
                        commitSetRowName(node, current.id, value, false);
                        refreshAllGetWidgets(rootGraph(node.graph));
                        refreshAllGetTypes(rootGraph(node.graph));
                    }

                    current.automatic = false;
                    setRowType(
                        node,
                        normalizeRows(node).findIndex(
                            (candidate) => candidate.id === current.id
                        ),
                        current.type
                    );
                    node.setDirtyCanvas?.(true, true);
                },
                isGet ? getComboOptions(node, row.id) : { multiline: false }
            );

            widget[WIDGET_FLAG] = true;
            widget.__jlcRowId = row.id;
            widget.label = isGet
                ? `Get ${index + 1} source`
                : `Set ${index + 1} name`;
            widget.tooltip = isGet
                ? "Select a connected Set channel. Arbitrary Get names are disabled."
                : "Shared channel name. A unique default is assigned when the input is connected.";
            configureWidgetSize(node, widget);
        });

        normalizeRows(node).forEach((row, index) => {
            if (isJlcGet(node)) {
                const descriptor = channelDescriptorByKey(
                    node.graph,
                    row.sourceKey,
                    false
                );
                if (descriptor) row.name = descriptor.name;
            }
            setRowType(node, index, row.type);
        });
    } finally {
        node.__jlcRebuilding = wasRebuilding;
    }

    resizeNode(node);
}

function refreshGetWidgetOptions(node) {
    if (!isJlcGet(node)) return;

    for (const row of normalizeRows(node)) {
        bindLegacyGetRow(node, row);
        const descriptor = channelDescriptorByKey(
            node.graph,
            row.sourceKey,
            false
        );
        if (descriptor) row.name = descriptor.name;

        const widget = getRowWidget(node, row.id);
        if (!widget) continue;
        widget.options = getComboOptions(node, row.id);
        widget.value = row.sourceKey || "";
    }

    normalizeRows(node).forEach((row, index) => {
        setRowType(node, index, row.type);
    });
    applyResponsiveLayout(node, node.size, false);
}

function refreshAllGetWidgets(root) {
    for (const graph of allGraphs(root)) {
        for (const node of graph._nodes ?? graph.nodes ?? []) {
            if (isJlcGet(node)) refreshGetWidgetOptions(node);
        }
    }
}

function rowIsEmpty(node, index) {
    const row = normalizeRows(node)[index];
    if (!row || row.name !== "") return false;

    if (isJlcSet(node)) {
        return (
            node.inputs?.[index]?.link == null &&
            !hasOutputLinks(node.outputs?.[index])
        );
    }

    return !hasOutputLinks(node.outputs?.[index]);
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

function queueRowRemoval(node, rowId) {
    if (!rowId) return;
    node.__jlcPendingRowRemovals ??= new Set();
    node.__jlcPendingRowRemovals.add(rowId);
}

function rowCanBeRemoved(node, index) {
    if (!normalizeRows(node)[index]) return false;
    if (hasOutputLinks(node.outputs?.[index])) return false;
    return !isJlcSet(node) || node.inputs?.[index]?.link == null;
}

function removeSocketRowAt(node, index, rowId) {
    const rows = normalizeRows(node);
    if (rows[index]?.id !== rowId) return false;

    // Keep the row in place while LiteGraph removes the matching sockets.
    // LiteGraph then reindexes every surviving later link in the live graph.
    if (isJlcSet(node)) node.removeInput(index);
    node.removeOutput(index);

    const authoritativeRows = node.properties[ROWS_KEY];
    const currentIndex = authoritativeRows.findIndex((row) => row.id === rowId);
    if (currentIndex < 0) return false;
    authoritativeRows.splice(currentIndex, 1);
    return true;
}

function removeRowById(node, rowId) {
    const rows = normalizeRows(node);
    const index = rows.findIndex((row) => row.id === rowId);
    if (index < 0 || !rowCanBeRemoved(node, index)) return false;
    return removeSocketRowAt(node, index, rowId);
}

function appendTrailingRow(node) {
    const rows = normalizeRows(node);
    if (rows.length >= MAX_ROWS) return false;
    rows.push(newRow(true));
    return true;
}

function removeLastRow(node) {
    const rows = normalizeRows(node);
    if (rows.length <= MIN_ROWS) return false;

    const index = rows.length - 1;
    if (!rows[index].automatic || !rowIsEmpty(node, index)) return false;
    if (isJlcSet(node) && node.inputs?.[index]?.link != null) return false;
    if (hasOutputLinks(node.outputs?.[index])) return false;

    const before = rows.length;
    const removed = removeSocketRowAt(node, index, rows[index].id);
    return removed && node.properties[ROWS_KEY].length === before - 1;
}

function maintainTrailingRows(node) {
    let changed = false;
    let rows = normalizeRows(node);

    if (!rowIsEmpty(node, rows.length - 1) && rows.length < MAX_ROWS) {
        changed = appendTrailingRow(node) || changed;
    }

    rows = normalizeRows(node);
    while (
        rows.length > MIN_ROWS &&
        rows.at(-1)?.automatic === true &&
        rowIsEmpty(node, rows.length - 1) &&
        rowIsEmpty(node, rows.length - 2)
    ) {
        const before = rows.length;
        if (!removeLastRow(node)) break;
        rows = normalizeRows(node);
        if (rows.length >= before) break;
        changed = true;
    }

    return changed;
}

function sortedPendingRowIds(node) {
    const pending = node.__jlcPendingRowRemovals;
    if (!(pending instanceof Set) || pending.size === 0) return [];

    const positions = new Map(
        normalizeRows(node).map((row, index) => [row.id, index])
    );
    return [...pending].sort((left, right) => {
        const byPosition =
            (positions.get(right) ?? -1) - (positions.get(left) ?? -1);
        return byPosition || String(left).localeCompare(String(right));
    });
}

function maintainRows(node) {
    if (node.__jlcStructuralUpdate) return;
    node.__jlcStructuralUpdate = true;

    const graph = node.graph;
    let transactionStarted = false;
    let structureChanged = false;

    try {
        if (typeof graph?.beforeChange === "function") {
            graph.beforeChange();
            transactionStarted = true;
        }

        syncNames(node);
        for (const rowId of sortedPendingRowIds(node)) {
            if (removeRowById(node, rowId)) structureChanged = true;
            node.__jlcPendingRowRemovals.delete(rowId);
        }

        if (isJlcSet(node)) {
            for (const row of [...normalizeRows(node)]) {
                ensureReadySetRowName(node, row.id);
            }
        }

        structureChanged = maintainTrailingRows(node) || structureChanged;

        normalizeRows(node);
        ensureSlots(node);
        if (structureChanged) rebuildWidgets(node);
        else resizeNode(node);
    } finally {
        try {
            if (transactionStarted && typeof graph?.afterChange === "function") {
                graph.afterChange();
            }
        } finally {
            node.__jlcStructuralUpdate = false;
        }
    }

    if (!node.graph) return;
    const root = rootGraph(node.graph);
    if (isJlcSet(node)) refreshSetTypes(node);
    refreshAllGetWidgets(root);
    refreshAllGetTypes(root);

    if (node.__jlcPendingRowRemovals?.size) {
        scheduleMaintainRows(node);
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
                "[JLC Multi Set/Get] Structural maintenance failed",
                error
            );
        }
    }, 0);
}

function setCandidates(graph, channel) {
    if (!graph || !channel) return [];
    return setChannelDescriptors(graph, false).filter(
        (descriptor) => descriptor.name === channel
    );
}

function describeCandidate(candidate) {
    if (candidate.kind === "KJ") {
        return `KJ SetNode ${String(candidate.node.id)}`;
    }
    return (
        `JLC Multi Set ${String(candidate.node.id)}, ` +
        `row ${candidate.rowIndex + 1}`
    );
}

function candidateSource(candidate) {
    const input = candidate.node.inputs?.[candidate.slot];
    if (!input || input.link == null) return null;

    const link = linkById(candidate.graph, input.link);
    if (!link) return null;

    const sourceNode = nodeById(candidate.graph, link.origin_id);
    if (!sourceNode) return null;

    return {
        link,
        node: sourceNode,
        slot: Number(link.origin_slot),
        type:
            (input.type && input.type !== "*" && input.type) ||
            sourceNode.outputs?.[link.origin_slot]?.type ||
            "*",
    };
}

function setGetValidationError(node, hasError) {
    if (!isJlcGet(node)) return;

    const next = hasError === true;
    const stateChanged = node.__jlcGetValidationError !== next;
    const oldRenderState = node.has_errors;

    if (next) {
        if (node.__jlcGetValidationError !== true) {
            node.__jlcGetValidationOwnsErrorFlag = node.has_errors !== true;
        }
        node.__jlcGetValidationError = true;
        node.has_errors = true;
    } else {
        node.__jlcGetValidationError = false;
        if (node.__jlcGetValidationOwnsErrorFlag === true) {
            node.has_errors = false;
        }
        delete node.__jlcGetValidationOwnsErrorFlag;
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

function resolveChannel(node, outputIndex, throwOnError) {
    syncNames(node);
    const row = normalizeRows(node)[outputIndex];
    const rowNumber = outputIndex + 1;

    const fail = (message) => {
        if (throwOnError) {
            setGetValidationError(node, true);
            throw new Error(message);
        }
        return null;
    };

    if (!row) {
        return fail(`JLC Multi Get row ${rowNumber} does not exist.`);
    }

    let candidate = null;
    if (row.sourceKey) {
        candidate = channelDescriptorByKey(node.graph, row.sourceKey, false);
        if (!candidate) {
            return fail(
                `JLC Multi Get row ${rowNumber} selected channel ` +
                    `"${row.name || row.sourceKey}" is no longer available.`
            );
        }
    } else if (row.name) {
        const candidates = setCandidates(node.graph, row.name);
        if (candidates.length > 1) {
            return fail(
                `JLC Multi Get row ${rowNumber} legacy channel ` +
                    `"${row.name}" is ambiguous (${candidates
                        .map(describeCandidate)
                        .join("; ")}). Select a Set channel again.`
            );
        }
        candidate = candidates[0] ?? null;
        if (candidate) row.sourceKey = candidate.key;
    }

    if (!candidate) {
        return fail(
            `JLC Multi Get row ${rowNumber} has no Set channel selected.`
        );
    }

    row.name = candidate.name;
    const source = candidateSource(candidate);
    if (!source) {
        return fail(
            `JLC Multi Get row ${rowNumber} channel "${row.name}": ` +
                `matching ${describeCandidate(candidate)} has no connected source value.`
        );
    }

    return { candidate, source };
}

function refreshGetValidationError(node) {
    if (!isJlcGet(node) || node.__jlcGetValidationError !== true) return;

    const hasUnresolvedRelevantRow = normalizeRows(node).some((_row, index) => {
        return (
            hasOutputLinks(node.outputs?.[index]) &&
            resolveChannel(node, index, false) == null
        );
    });
    setGetValidationError(node, hasUnresolvedRelevantRow);
}

function linkedInputType(node, index) {
    const input = node.inputs?.[index];
    if (!input || input.link == null || !node.graph) return null;
    const link = linkById(node.graph, input.link);
    if (!link) return null;
    const source = nodeById(node.graph, link.origin_id);
    return normalizedType(source?.outputs?.[link.origin_slot]?.type ?? link.type);
}

function linkedOutputType(node, index) {
    const output = node.outputs?.[index];
    if (!hasOutputLinks(output) || !node.graph) return null;

    for (const linkId of output.links) {
        const link = linkById(node.graph, linkId);
        if (!link) continue;
        const target = nodeById(node.graph, link.target_id);
        const type = target?.inputs?.[link.target_slot]?.type ?? link.type;
        if (type && type !== "*") return normalizedType(type);
    }

    return null;
}

function refreshSetRowType(node, index) {
    if (!isJlcSet(node)) return;
    setRowType(
        node,
        index,
        linkedInputType(node, index) ?? linkedOutputType(node, index) ?? "*"
    );
}

function refreshSetTypes(node) {
    if (!isJlcSet(node)) return;
    normalizeRows(node).forEach((_row, index) => {
        refreshSetRowType(node, index);
    });
    node.__jlcPendingSetTypeRows?.clear();
}

function scheduleSetTypeRefresh(node, rowId) {
    if (!rowId || node.__jlcStructuralUpdate) return;

    node.__jlcPendingSetTypeRows ??= new Set();
    node.__jlcPendingSetTypeRows.add(rowId);
    if (node.__jlcSetTypeTimer != null) return;

    node.__jlcSetTypeTimer = setTimeout(() => {
        node.__jlcSetTypeTimer = null;
        if (!node.graph) return;
        if (node.__jlcStructuralUpdate) {
            scheduleSetTypeRefresh(node, rowId);
            return;
        }

        const pending = [...node.__jlcPendingSetTypeRows];
        node.__jlcPendingSetTypeRows.clear();
        let namesChanged = false;
        for (const pendingRowId of pending) {
            namesChanged =
                ensureReadySetRowName(node, pendingRowId) || namesChanged;
            const index = normalizeRows(node).findIndex(
                (row) => row.id === pendingRowId
            );
            if (index >= 0) refreshSetRowType(node, index);
        }

        const root = rootGraph(node.graph);
        refreshAllGetWidgets(root);
        refreshAllGetTypes(root);
        if (namesChanged) node.setDirtyCanvas?.(true, true);
    }, 0);
}

function typesOverlap(left, right) {
    if (!left || !right || left === "*" || right === "*") return true;
    const leftTypes = new Set(String(left).split(","));
    return String(right)
        .split(",")
        .some((type) => leftTypes.has(type));
}

function validateGetOutputLinks(node, index) {
    const output = node.outputs?.[index];
    if (!hasOutputLinks(output) || !node.graph) return;

    const outputType = output.type ?? "*";
    for (const linkId of [...output.links]) {
        const link = linkById(node.graph, linkId);
        if (!link) continue;
        const target = nodeById(node.graph, link.target_id);
        const targetType = target?.inputs?.[link.target_slot]?.type ?? link.type ?? "*";
        if (!typesOverlap(outputType, targetType)) {
            node.graph.removeLink?.(linkId);
        }
    }
}

function refreshGetTypes(node) {
    if (
        !isJlcGet(node) ||
        node.__jlcRefreshingTypes ||
        node.__jlcStructuralUpdate
    ) {
        return;
    }

    node.__jlcRefreshingTypes = true;

    try {
        normalizeRows(node).forEach((_row, index) => {
            const resolved = resolveChannel(node, index, false);
            setRowType(node, index, resolved?.source.type ?? "*");
            validateGetOutputLinks(node, index);
        });

        node.setDirtyCanvas?.(true, true);
    } finally {
        node.__jlcRefreshingTypes = false;
        if (node.__jlcPendingRowRemovals?.size) {
            scheduleMaintainRows(node);
        }
    }

    refreshGetValidationError(node);
}

function refreshAllGetTypes(root) {
    for (const graph of allGraphs(root)) {
        for (const node of graph._nodes ?? graph.nodes ?? []) {
            if (isJlcGet(node)) refreshGetTypes(node);
        }
    }
}

function sourceForSetRow(node, slot, throwOnError) {
    const row = normalizeRows(node)[slot];
    const fail = (message) => {
        if (throwOnError) throw new Error(message);
        return null;
    };

    if (!row) return fail(`JLC Multi Set row ${slot + 1} does not exist.`);
    const input = node.inputs?.[slot];
    if (!input || input.link == null) {
        return fail(
            `JLC Multi Set row ${slot + 1}${
                row.name ? ` channel "${row.name}"` : ""
            } has no connected source value.`
        );
    }

    const link = linkById(node.graph, input.link);
    if (!link) return fail(`JLC Multi Set row ${slot + 1} has a missing source link.`);
    const sourceNode = nodeById(node.graph, link.origin_id);
    if (!sourceNode) return fail(`JLC Multi Set row ${slot + 1} has a missing source node.`);

    return { link, node: sourceNode, slot: Number(link.origin_slot) };
}

function installNode(node, kind) {
    if (node[INSTALL_FLAG]) return;
    node[INSTALL_FLAG] = true;

    node.properties ??= {};
    node.properties[KIND_KEY] = kind;
    node.properties[FORMAT_KEY] = FORMAT_VERSION;
    node.properties[ROWS_KEY] ??= [newRow(false)];
    node.__jlcPendingRowRemovals ??= new Set();
    node.__jlcPendingSetTypeRows ??= new Set();

    node.isVirtualNode = true;
    node.serialize_widgets = false;
    node.widgets_start_y = WIDGET_TOP;

    const original = {
        onAdded: node.onAdded?.bind(node),
        onConfigure: node.onConfigure?.bind(node),
        onSerialize: node.onSerialize?.bind(node),
        onConnectionsChange: node.onConnectionsChange?.bind(node),
        onResize: node.onResize?.bind(node),
        computeSize: node.computeSize?.bind(node),
        getConnectionPos: node.getConnectionPos?.bind(node),
    };

    node.resizable = true;
    node.computeSize = function () {
        // computeSize() is used by LiteGraph as a natural/minimum-size hint
        // during interactive resizing. Returning the node's current width here
        // makes every widened width become the next minimum, so the node can
        // grow but can no longer shrink. Keep the width hint fixed instead;
        // onResize/applyResponsiveLayout handle the actual user-selected width.
        return [MIN_NODE_WIDTH, desiredNodeHeight(this)];
    };

    node.onResize = function (size) {
        const result = original.onResize?.(...arguments);
        applyResponsiveLayout(this, size ?? this.size, false);
        return result;
    };

    // LiteGraph uses getConnectionPos, not getInputPos/getOutputPos. Align each
    // socket with the channel widget in the same row.
    node.getConnectionPos = function (isInput, slot, out) {
        const row = normalizeRows(this)[slot];
        const supportedSide = !isInput || isJlcSet(this);
        if (!row || !supportedSide) {
            return (
                original.getConnectionPos?.(isInput, slot, out) ??
                new Float32Array([this.pos[0], this.pos[1]])
            );
        }

        const result = out ?? new Float32Array(2);
        const widget = getRowWidget(this, row.id);
        const localY = Number.isFinite(widget?.last_y)
            ? widget.last_y + ROW_HEIGHT / 2
            : WIDGET_TOP + slot * rowPitch() + ROW_HEIGHT / 2;

        result[0] = this.pos[0] + (isInput ? 0 : this.size[0]);
        result[1] = this.pos[1] + localY;
        return result;
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
        this.__jlcRebuilding = true;
        try {
            normalizeRows(this);
            ensureSlots(this);
            rebuildWidgets(this);
        } finally {
            this.__jlcRebuilding = false;
        }

        setTimeout(() => {
            if (!this.graph) return;
            maintainRows(this);
        }, 0);
        return result;
    };

    node.onSerialize = function (serialized) {
        const result = original.onSerialize?.(serialized);
        syncNames(this);

        serialized.properties ??= {};
        serialized.properties[KIND_KEY] = kind;
        serialized.properties[FORMAT_KEY] = FORMAT_VERSION;
        serialized.properties[ROWS_KEY] = normalizeRows(this).map((row) => ({
            id: row.id,
            name: row.name,
            type: row.type,
            automatic: row.automatic,
            nameAutomatic: row.nameAutomatic,
            sourceKey: row.sourceKey,
        }));

        return result;
    };

    node.onConnectionsChange = function (slotType, slot, isConnected, linkInfo) {
        const affectedRowId = this.__jlcStructuralUpdate
            ? null
            : rowIdFromLinkInfo(this, linkInfo);
        const result = original.onConnectionsChange?.(...arguments);
        if (app.configuringGraph) return result;
        if (this.__jlcStructuralUpdate) return result;

        if (!isConnected && affectedRowId) {
            queueRowRemoval(this, affectedRowId);
        }

        if (isJlcSet(this)) {
            // The callback slot is not authoritative for source callbacks.
            // Re-find the row by stable ID after LiteGraph finishes the change.
            scheduleSetTypeRefresh(this, affectedRowId);
        } else if (isJlcGet(this)) {
            if (!this.__jlcRefreshingTypes) refreshGetTypes(this);
        }

        // Type validation can synchronously remove links. Its outer refresh
        // schedules one structural pass after the graph becomes stable.
        if (this.__jlcRefreshingTypes) return result;

        // Always defer structural work until LiteGraph has completed the link
        // mutation. Pending removals retain every affected stable row ID.
        scheduleMaintainRows(this);

        return result;
    };

    if (kind === "set") {
        node.getInputLink = function (slot) {
            return sourceForSetRow(this, slot, true).link;
        };

        node.resolveVirtualOutput = function (slot) {
            const source = sourceForSetRow(this, slot, true);
            return { node: source.node, slot: source.slot };
        };
    } else {
        node.getInputLink = function (slot) {
            const resolved = resolveChannel(this, slot, true);
            if (resolved.candidate.graph !== this.graph) return null;
            return resolved.source.link;
        };

        node.resolveVirtualOutput = function (slot) {
            const resolved = resolveChannel(this, slot, true);
            setRowType(this, slot, resolved.source.type);
            if (resolved.candidate.graph === this.graph) return undefined;
            return { node: resolved.source.node, slot: resolved.source.slot };
        };
    }

    node.__jlcRebuilding = true;
    try {
        normalizeRows(node);
        ensureSlots(node);
        rebuildWidgets(node);
    } finally {
        node.__jlcRebuilding = false;
    }
}

app.registerExtension({
    name: "JLC.DynamicMultiSetGet",

    beforeRegisterNodeDef(nodeType, nodeData) {
        const kind = kindFromIdentifiers(
            nodeData?.name,
            nodeData?.display_name,
            nodeType?.comfyClass,
            nodeType?.title
        );
        if (!kind || nodeType.prototype.__jlcDynamicMultiPrototypeWrapped) {
            return;
        }

        nodeType.prototype.__jlcDynamicMultiPrototypeWrapped = true;
        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalOnNodeCreated?.apply(this, arguments);
            try {
                installNode(this, kind);
            } catch (error) {
                console.error(
                    `[JLC Multi Set/Get] Failed to initialize ${kind} node`,
                    error
                );
            }
            return result;
        };
    },

    // Fallback for frontend builds that expose the instance callback more
    // reliably than beforeRegisterNodeDef. INSTALL_FLAG prevents double setup.
    nodeCreated(node) {
        const kind = kindFromIdentifiers(
            node?.comfyClass,
            node?.constructor?.comfyClass,
            node?.type,
            node?.title
        );
        if (!kind) return;
        try {
            installNode(node, kind);
        } catch (error) {
            console.error(
                `[JLC Multi Set/Get] Failed to initialize ${kind} node`,
                error
            );
        }
    },
});

console.info("[JLC Multi Set/Get] frontend extension loaded");
