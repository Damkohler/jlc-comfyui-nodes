/*
 * JLC Load/Resize Image Nodes - Dynamic Widget Visibility
 * --------------------------------------------------------
 * Shared frontend companion for:
 *   - JLC_LoadAndResizeImage
 *   - JLC_ResizeImage
 *
 * Both nodes predeclare every mode-specific numeric widget. This extension
 * exposes only the control used by the selected resize mode.
 *
 * Layout policy:
 *   - JLC_ResizeImage is compacted to the minimum visible-widget height.
 *   - JLC_LoadAndResizeImage preserves the user's current node height because
 *     that height also controls the embedded image-preview area.
 *
 * Important:
 *   - Keep each widget's real type intact. In current ComfyUI frontends,
 *     widget-backed input sockets coexist with their widgets.
 *   - Never replace node.widgets with a filtered list during computeSize().
 *     Current frontends expose node.widgets through a reactive setter, so
 *     filtering that array can permanently remove the hidden widgets.
 */

const { app } = window.comfyAPI.app;

const LOAD_NODE_NAME = "JLC_LoadAndResizeImage";
const RESIZE_NODE_NAME = "JLC_ResizeImage";

const NODE_NAMES = new Set([
    LOAD_NODE_NAME,
    RESIZE_NODE_NAME,
]);

const MODE_WIDGET = "resize_by";
const LAYOUT_KEY = "__jlc_resize_widget_layout";
const INSTALL_FLAG = "__jlc_resize_visibility_installed";
const SOCKET_VISIBILITY_FLAG = "__jlc_resize_socket_visibility_installed";
const ACTIVE_WIDGET_KEY = "__jlc_resize_active_widget";

const MODE_WIDGET_MAP = {
    "scale by multiplier": "multiplier",
    "scale longer dimension": "longer_size",
    "scale shorter dimension": "shorter_size",
    "scale width": "width",
    "scale height": "height",
    "scale total pixels": "megapixels",
};

const MODE_SPECIFIC_WIDGETS = new Set(Object.values(MODE_WIDGET_MAP));

/*
 * Current ComfyUI hit-testing checks every input socket in array order, even
 * when the widget associated with that socket is hidden. Because hidden
 * widget-backed sockets collapse onto the visible widget row, an earlier
 * hidden socket can intercept the click. In this node, the first such socket
 * is multiplier (FLOAT), which made the visible longer_size row behave like a
 * FLOAT socket and reject INT links.
 *
 * Keep all backend inputs intact, but move inactive mode-specific sockets out
 * of the hit-test/render area through getInputPos().
 */
function installSocketVisibility(node) {
    if (
        node[SOCKET_VISIBILITY_FLAG] ||
        typeof node.getInputPos !== "function"
    ) {
        return;
    }

    node[SOCKET_VISIBILITY_FLAG] = true;
    const originalGetInputPos = node.getInputPos;

    node.getInputPos = function (slot) {
        const input = this.inputs?.[slot];
        const widgetName = input?.widget?.name;

        if (
            MODE_SPECIFIC_WIDGETS.has(widgetName) &&
            widgetName !== this[ACTIVE_WIDGET_KEY]
        ) {
            const out = arguments[1] ?? [0, 0];

            out[0] = (this.pos?.[0] ?? 0) - 1000000;
            out[1] = (this.pos?.[1] ?? 0) - 1000000;
            return out;
        }

        return originalGetInputPos.apply(this, arguments);
    };
}

function rememberWidgetLayout(widget) {
    if (!widget[LAYOUT_KEY]) {
        widget[LAYOUT_KEY] = {
            hidden: widget.hidden,
            optionsHidden: widget.options?.hidden,
        };
    }
}

function hideWidget(widget) {
    rememberWidgetLayout(widget);

    // Preserve widget.type (for example, "number"). The associated socket
    // remains a valid INT/FLOAT widget-backed input while the control is hidden.
    widget.hidden = true;
    widget.options ??= {};
    widget.options.hidden = true;
}

function showWidget(widget) {
    rememberWidgetLayout(widget);
    const layout = widget[LAYOUT_KEY];

    widget.hidden = layout.hidden ?? false;
    widget.options ??= {};

    if (layout.optionsHidden === undefined) {
        delete widget.options.hidden;
    } else {
        widget.options.hidden = layout.optionsHidden;
    }
}

function resizeNodeAfterVisibility(node) {
    if (typeof node.computeSize !== "function" || !node.size) {
        return;
    }

    // Current ComfyUI computeSize() already ignores widget.hidden controls.
    const computed = node.computeSize();
    if (!computed) {
        return;
    }

    const currentWidth = node.size[0] ?? computed[0] ?? 260;
    const minimumWidth = computed[0] ?? currentWidth;
    const targetWidth = Math.max(currentWidth, minimumWidth);

    let targetHeight;

    if (node.comfyClass === RESIZE_NODE_NAME) {
        targetHeight = computed[1];
    } else {
        const currentHeight = node.size[1] ?? computed[1];
        targetHeight = Math.max(currentHeight, computed[1]);
    }

    if (typeof node.setSize === "function") {
        node.setSize([targetWidth, targetHeight]);
    } else {
        node.size[0] = targetWidth;
        node.size[1] = targetHeight;
        node.onResize?.(node.size);
    }
}

function applyModeVisibility(node) {
    const modeWidget = node.widgets?.find(
        (widget) => widget.name === MODE_WIDGET
    );

    const selectedMode = String(
        modeWidget?.value ?? "scale longer dimension"
    );

    const activeWidgetName =
        MODE_WIDGET_MAP[selectedMode] ?? "longer_size";

    node[ACTIVE_WIDGET_KEY] = activeWidgetName;

    for (const widget of node.widgets ?? []) {
        if (!MODE_SPECIFIC_WIDGETS.has(widget.name)) {
            continue;
        }

        if (widget.name === activeWidgetName) {
            showWidget(widget);
        } else {
            hideWidget(widget);
        }
    }

    // Visibility changes alter widget-backed socket placement.
    node._widgetSlotsDirty = true;

    resizeNodeAfterVisibility(node);
    node.setDirtyCanvas?.(true, true);
    node.graph?.setDirtyCanvas?.(true, true);
}

function scheduleModeVisibility(node) {
    requestAnimationFrame(() => {
        applyModeVisibility(node);
        requestAnimationFrame(() => applyModeVisibility(node));
    });
}

function installVisibility(node) {
    if (node[INSTALL_FLAG]) {
        return;
    }

    node[INSTALL_FLAG] = true;
    installSocketVisibility(node);

    const originalOnConfigure = node.onConfigure;
    node.onConfigure = function () {
        const result = originalOnConfigure?.apply(this, arguments);
        scheduleModeVisibility(this);
        return result;
    };

    const modeWidget = node.widgets?.find(
        (widget) => widget.name === MODE_WIDGET
    );

    if (modeWidget) {
        const originalCallback = modeWidget.callback;

        modeWidget.callback = function () {
            const result = originalCallback?.apply(this, arguments);
            scheduleModeVisibility(node);
            return result;
        };
    }

    scheduleModeVisibility(node);
}

app.registerExtension({
    name: "JLC.LoadAndResizeImage.Visibility",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!NODE_NAMES.has(nodeData?.name)) {
            return;
        }

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {
            const result = originalOnNodeCreated?.apply(this, arguments);
            installVisibility(this);
            return result;
        };
    },
});
