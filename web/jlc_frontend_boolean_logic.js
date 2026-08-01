/**
 * JLC ComfyUI Custom Nodes
 * ------------------------
 *
 * Component:
 *   JLC Boolean Logic (Frontend)
 *
 * Description:
 *   Pure client-side Boolean logic for controlling frontend/virtual workflow
 *   nodes such as ComfyUI-Switchboard Group Controller and Node Controller.
 *
 * Repository:
 *   https://github.com/Damkohler/jlc-comfyui-nodes
 *
 * Author:
 *   J. L. Córdova
 *
 * License:
 *   MIT
 *
 * Version:
 *   1.0.0
 *
 * Notes:
 *   - This is a virtual frontend node and is not submitted to the Python backend.
 *   - Both inputs must be connected and frontend-resolvable; otherwise the
 *     output fails closed to false.
 *   - The calculated result is exposed synchronously so frontend controllers
 *     can read the correct value immediately before queueing.
 */

import { app } from "../../scripts/app.js";

// ---------------------------------------------------------------------------
// JLC component metadata
// ---------------------------------------------------------------------------

const JLC_FRONTEND_BOOLEAN_LOGIC_VERSION = "1.0.0";

const NODE_TITLE = "\u2003JLC Boolean Logic (Frontend)";
const CATEGORY = "JLC/Utility";
const POLL_MS = 100;

const ICON_SIZE = 12;
const ICON_URL =
  "/extensions/JLC-ComfyUI-nodes/assets/icons/jlc-comfyui-nodes_Logo-Dark-0128.png";

const OPERATIONS = Object.freeze([
  "AND",
  "OR",
  "XOR",
  "NAND",
  "NOR",
  "XNOR",
  "A AND NOT B",
  "B AND NOT A",
]);

const DEFAULT_OPERATION = "AND";

const iconImage = new Image();
iconImage.src = ICON_URL;

// ---------------------------------------------------------------------------
// LiteGraph compatibility helpers
// ---------------------------------------------------------------------------

function linkById(graph, id) {
  const links = graph && graph.links;
  if (!links || id == null) return null;

  if (typeof links.get === "function") {
    return links.get(id) || null;
  }

  return links[id] || null;
}

function nodeById(graph, id) {
  if (!graph) return null;

  let node = graph.getNodeById ? graph.getNodeById(id) : null;

  if (!node && /^\d+$/.test(String(id)) && graph.getNodeById) {
    node = graph.getNodeById(Number(id));
  }

  if (node) return node;

  // Subgraph I/O proxy nodes are not always registered in getNodeById().
  const candidates = [];

  if (graph.inputNode) candidates.push(graph.inputNode);
  if (graph.outputNode) candidates.push(graph.outputNode);
  if (graph._inputNode) candidates.push(graph._inputNode);
  if (graph._outputNode) candidates.push(graph._outputNode);
  if (graph.input_node) candidates.push(graph.input_node);
  if (graph.output_node) candidates.push(graph.output_node);

  if (Array.isArray(graph._input_nodes)) {
    candidates.push(...graph._input_nodes);
  }

  if (Array.isArray(graph._output_nodes)) {
    candidates.push(...graph._output_nodes);
  }

  for (const candidate of candidates) {
    if (!candidate) continue;

    if (candidate.id === id || candidate.id === Number(id)) {
      return candidate;
    }
  }

  return null;
}

function subgraphOf(node) {
  if (!node) return null;

  if (node.subgraph) return node.subgraph;
  if (node._subgraph) return node._subgraph;

  const root = (node.graph && node.graph._rootGraph) || app.graph;
  const registry = root && (root._subgraphs || root.subgraphs);

  if (registry && typeof registry.get === "function") {
    const id =
      node.subgraphId ||
      node.properties?.subgraph ||
      node.type;

    const subgraph = id != null ? registry.get(id) : null;

    if (subgraph) {
      return subgraph._graph || subgraph.graph || subgraph;
    }
  }

  return null;
}

function subgraphIdOf(node) {
  if (!node) return null;

  if (node.subgraph && node.subgraph.id != null) {
    return node.subgraph.id;
  }

  return (
    node.subgraphId ||
    node.properties?.subgraph ||
    node.type ||
    null
  );
}

function findSubgraphHost(graph) {
  if (!graph) {
    return { host: null, parentGraph: null };
  }

  const direct =
    graph._subgraph_node ||
    graph.subgraphNode ||
    graph._node ||
    null;

  if (direct) {
    return {
      host: direct,
      parentGraph: direct.graph,
    };
  }

  const wantedId = graph.id;
  const roots = [];

  if (app.graph) {
    roots.push(app.graph);
  }

  if (graph._rootGraph && !roots.includes(graph._rootGraph)) {
    roots.push(graph._rootGraph);
  }

  const stack = [...roots];
  const seen = new Set();

  while (stack.length) {
    const current = stack.pop();

    if (!current || seen.has(current)) {
      continue;
    }

    seen.add(current);

    for (const node of current._nodes || current.nodes || []) {
      const subgraph = subgraphOf(node);

      if (
        subgraph === graph ||
        (
          wantedId != null &&
          subgraphIdOf(node) === wantedId
        )
      ) {
        return {
          host: node,
          parentGraph: current,
        };
      }

      if (subgraph && !seen.has(subgraph)) {
        stack.push(subgraph);
      }
    }
  }

  return {
    host: null,
    parentGraph: null,
  };
}

function crossSubgraphInput(graph, origin, originSlot) {
  try {
    const inputProxy =
      graph.inputNode ||
      graph._inputNode ||
      graph.input_node ||
      null;

    if (!inputProxy || origin !== inputProxy) {
      return null;
    }

    const { host, parentGraph } = findSubgraphHost(graph);

    if (!host || !parentGraph) {
      return null;
    }

    const parentInput = host.inputs?.[originSlot];

    if (!parentInput || parentInput.link == null) {
      return null;
    }

    return {
      graph: parentGraph,
      linkId: parentInput.link,
    };
  } catch (error) {
    console.debug(
      "[JLC Boolean Logic] Subgraph input crossing failed.",
      error,
    );

    return null;
  }
}

function crossSubgraphOutput(origin, originSlot) {
  try {
    const subgraph = subgraphOf(origin);

    if (!subgraph) {
      return null;
    }

    const output = subgraph.outputs?.[originSlot];

    if (!output) {
      return null;
    }

    let linkId = output.link;

    if (linkId == null && Array.isArray(output.linkIds)) {
      linkId = output.linkIds[0];
    }

    if (
      linkId == null &&
      Array.isArray(output._floatingLinks) &&
      output._floatingLinks[0]
    ) {
      const floatingLink = output._floatingLinks[0];

      linkId =
        floatingLink.id != null
          ? floatingLink.id
          : floatingLink;
    }

    if (linkId == null) {
      return null;
    }

    return {
      graph: subgraph,
      linkId,
    };
  } catch (error) {
    console.debug(
      "[JLC Boolean Logic] Subgraph output crossing failed.",
      error,
    );

    return null;
  }
}

// ---------------------------------------------------------------------------
// Frontend Boolean resolution
// ---------------------------------------------------------------------------

/**
 * Read a Boolean value that is already knowable in the browser.
 *
 * Resolution priority:
 *   1. Dedicated frontend resolver implemented by a virtual source node.
 *   2. Boolean-valued widget on the source node.
 *   3. Cached Boolean on the specific output slot.
 */
function readFrontendBoolean(node, originSlot, context) {
  if (!node) {
    return null;
  }

  if (typeof node.getFrontendBooleanOutput === "function") {
    try {
      const value = node.getFrontendBooleanOutput(
        originSlot,
        context,
      );

      if (typeof value === "boolean") {
        return value;
      }
    } catch (error) {
      console.debug(
        "[JLC Boolean Logic] Source frontend resolver failed.",
        error,
      );
    }
  }

  const widgets = node.widgets || [];

  let widget = widgets.find((item) => {
    try {
      return typeof item.value === "boolean";
    } catch {
      return false;
    }
  });

  if (!widget) {
    widget = widgets.find((item) =>
      /^(value|boolean|bool|result)$/i.test(item.name || ""),
    );
  }

  if (widget) {
    try {
      if (typeof widget.value === "boolean") {
        return !!widget.value;
      }
    } catch (error) {
      console.debug(
        "[JLC Boolean Logic] Boolean widget read failed.",
        error,
      );
    }
  }

  const output = node.outputs?.[originSlot];

  if (output && typeof output._data === "boolean") {
    return output._data;
  }

  return null;
}

/**
 * Follow a Boolean wire to a value resolvable in the browser.
 *
 * Returns:
 *   true / false  -> successfully resolved
 *   null          -> disconnected, backend-only, or otherwise unavailable
 */
function resolveBoolean(graph, linkId, context = null) {
  if (!graph || linkId == null) {
    return null;
  }

  const state =
    context ||
    {
      depth: 0,
      visited: new Set(),
    };

  if (state.depth > 24) {
    return null;
  }

  const graphKey =
    graph.id != null
      ? graph.id
      : "root";

  const visitKey = `${graphKey}:${String(linkId)}`;

  if (state.visited.has(visitKey)) {
    return null;
  }

  state.visited.add(visitKey);

  const link = linkById(graph, linkId);

  if (!link) {
    return null;
  }

  const origin = nodeById(graph, link.origin_id);

  if (!origin) {
    return null;
  }

  const nextContext = {
    depth: state.depth + 1,
    visited: state.visited,
  };

  const direct = readFrontendBoolean(
    origin,
    link.origin_slot,
    nextContext,
  );

  if (direct !== null) {
    return direct;
  }

  // Source is this subgraph's input proxy: walk outward to the parent graph.
  const up = crossSubgraphInput(
    graph,
    origin,
    link.origin_slot,
  );

  if (up) {
    return resolveBoolean(
      up.graph,
      up.linkId,
      nextContext,
    );
  }

  // Source is a subgraph instance: descend to what drives its output.
  const down = crossSubgraphOutput(
    origin,
    link.origin_slot,
  );

  if (down) {
    return resolveBoolean(
      down.graph,
      down.linkId,
      nextContext,
    );
  }

  return null;
}

// ---------------------------------------------------------------------------
// Boolean operation implementation
// ---------------------------------------------------------------------------

function normalizeOperation(value) {
  return OPERATIONS.includes(value)
    ? value
    : DEFAULT_OPERATION;
}

function evaluateOperation(operation, boolA, boolB) {
  switch (operation) {
    case "AND":
      return boolA && boolB;

    case "OR":
      return boolA || boolB;

    case "XOR":
      return boolA !== boolB;

    case "NAND":
      return !(boolA && boolB);

    case "NOR":
      return !(boolA || boolB);

    case "XNOR":
      return boolA === boolB;

    case "A AND NOT B":
      return boolA && !boolB;

    case "B AND NOT A":
      return boolB && !boolA;

    default:
      return false;
  }
}

// ---------------------------------------------------------------------------
// Node implementation
// ---------------------------------------------------------------------------

class JLCFrontendBooleanLogicNode extends LGraphNode {
  static nodeTitle = NODE_TITLE;
  static version = JLC_FRONTEND_BOOLEAN_LOGIC_VERSION;
  static isJLCFrontendBooleanLogic = true;

  constructor(title = NODE_TITLE) {
    super(title);

    // Virtual nodes are omitted from the Python/backend prompt.
    this.isVirtualNode = true;

    // Widget values are reconstructed from properties.
    this.serialize_widgets = false;

    if (!this.properties) {
      this.properties = {};
    }

    this.properties.operation = normalizeOperation(
      this.properties.operation,
    );

    // Serialized component-version marker for workflow diagnostics.
    this.properties.jlcVersion =
      JLC_FRONTEND_BOOLEAN_LOGIC_VERSION;

    this.addInput("bool_a", "BOOLEAN");
    this.addInput("bool_b", "BOOLEAN");
    this.addOutput("boolean", "BOOLEAN");

    this._cachedResult = false;

    this._buildWidgets();

    this.size = [250, 130];
    this._syncNow(false);
  }

  _graph() {
    return this.graph || app.graph;
  }

  _buildWidgets() {
    this.widgets = [];

    this._operationWidget = this.addWidget(
      "combo",
      "operation",
      this.properties.operation,
      (value) => {
        this.properties.operation = normalizeOperation(value);
        this._operationWidget.value = this.properties.operation;
        this._syncNow(true);
      },
      {
        values: OPERATIONS,
      },
    );

    this._resultWidget = this.addWidget(
      "toggle",
      "result",
      false,
      () => {
        // Derived display only. Any click is immediately replaced by the
        // calculated result.
        this._syncNow(true);
      },
      {
        on: "true",
        off: "false",
      },
    );

    /*
     * Switchboard discovers frontend Boolean values by reading a Boolean widget
     * on the source node. Make the result widget a synchronous getter so an
     * immediate queue action cannot observe a stale timer-cached value.
     */
    Object.defineProperty(
      this._resultWidget,
      "value",
      {
        configurable: true,
        enumerable: true,

        get: () => this._computeResult(),

        set: (value) => {
          this._cachedResult = !!value;
        },
      },
    );
  }

  _readInput(slot) {
    const input = this.inputs?.[slot];

    if (!input || input.link == null) {
      return {
        resolved: false,
        value: false,
      };
    }

    const resolvedValue = resolveBoolean(
      this._graph(),
      input.link,
    );

    if (resolvedValue === null) {
      return {
        resolved: false,
        value: false,
      };
    }

    return {
      resolved: true,
      value: resolvedValue,
    };
  }

  _computeResult() {
    const inputA = this._readInput(0);
    const inputB = this._readInput(1);

    /*
     * Fail closed:
     * Both inputs must be connected and frontend-resolvable. This prevents
     * NAND, NOR, or XNOR from unexpectedly returning true when an input is
     * disconnected or comes from a backend-only node.
     */
    if (!inputA.resolved || !inputB.resolved) {
      return false;
    }

    return evaluateOperation(
      this.properties.operation,
      inputA.value,
      inputB.value,
    );
  }

  /**
   * Public synchronous frontend resolver.
   *
   * Other JLC virtual logic nodes can call this directly instead of depending
   * on timer-cached output data.
   */
  getFrontendBooleanOutput(
    _originSlot = 0,
    _context = null,
  ) {
    return this._computeResult();
  }

  _syncNow(forceDirty = false) {
    const result = this._computeResult();
    const changed = this._cachedResult !== result;

    this._cachedResult = result;

    if (this.outputs?.[0]) {
      this.outputs[0]._data = result;
    }

    if (changed || forceDirty) {
      this.setDirtyCanvas(true, true);
    }

    return result;
  }

  /**
   * ComfyUI/Switchboard-compatible queue-time hook.
   *
   * Recompute before graph submission even though this node itself remains
   * entirely virtual and is never sent to Python.
   */
  applyToGraph() {
    this._syncNow(false);
  }

  onAdded() {
    this._syncNow(true);

    if (!this._pollTimer) {
      this._pollTimer = setInterval(
        () => this._syncNow(false),
        POLL_MS,
      );
    }
  }

  onRemoved() {
    if (this._pollTimer) {
      clearInterval(this._pollTimer);
      this._pollTimer = null;
    }
  }

  onConfigure() {
    if (!this.properties) {
      this.properties = {};
    }

    this.properties.operation = normalizeOperation(
      this.properties.operation,
    );

    this.properties.jlcVersion =
      JLC_FRONTEND_BOOLEAN_LOGIC_VERSION;

    if (this._operationWidget) {
      this._operationWidget.value =
        this.properties.operation;
    }

    setTimeout(
      () => this._syncNow(true),
      0,
    );
  }

  onConnectionsChange() {
    this._syncNow(true);
  }

  onDrawForeground(ctx) {
    if (
      !iconImage.complete ||
      iconImage.naturalWidth === 0
    ) {
      return;
    }

    ctx.save();

    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";

    const x = ICON_SIZE + 18;
    const y = -(ICON_SIZE + 9);

    ctx.drawImage(
      iconImage,
      x,
      y,
      ICON_SIZE,
      ICON_SIZE,
    );

    ctx.restore();
  }

  getExtraMenuOptions(_, options) {
    options.push({
      content: "Log JLC Boolean Logic diagnostics",

      callback: () => {
        const inputA = this._readInput(0);
        const inputB = this._readInput(1);

        const diagnostics = {
          node: NODE_TITLE,
          version: JLC_FRONTEND_BOOLEAN_LOGIC_VERSION,
          operation: this.properties.operation,
          graph:
            this._graph()?.constructor?.name ||
            null,
          inputAConnected:
            this.inputs?.[0]?.link != null,
          inputBConnected:
            this.inputs?.[1]?.link != null,
          inputAResolved: inputA.resolved,
          inputBResolved: inputB.resolved,
          inputAValue: inputA.value,
          inputBValue: inputB.value,
          result: this._computeResult(),
          outputData: this.outputs?.[0]?._data,
        };

        console.log(
          "[JLC Boolean Logic] DIAGNOSTICS - copy this:",
          diagnostics,
        );
      },
    });
  }
}

// ---------------------------------------------------------------------------
// ComfyUI frontend registration
// ---------------------------------------------------------------------------

app.registerExtension({
  name: "JLC.FrontendBooleanLogic",

  registerCustomNodes() {
    const alreadyRegistered =
      LiteGraph.registered_node_types?.[NODE_TITLE];

    if (alreadyRegistered) {
      return;
    }

    JLCFrontendBooleanLogicNode.title = NODE_TITLE;
    JLCFrontendBooleanLogicNode.collapsable = true;

    LiteGraph.registerNodeType(
      NODE_TITLE,
      JLCFrontendBooleanLogicNode,
    );

    JLCFrontendBooleanLogicNode.category = CATEGORY;

    console.info(
      `[JLC] ${NODE_TITLE.trim()} v${JLC_FRONTEND_BOOLEAN_LOGIC_VERSION} registered.`,
    );
  },
});
