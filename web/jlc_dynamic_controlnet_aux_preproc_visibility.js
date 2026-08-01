/*
 * JLC Dynamic Aux Preprocessor Wrapper Visibility Helpers
 * ----------------------------------------------------------------
 *
 * JLC ComfyUI Nodes Collection
 *   Frontend companion for JLC_DynamicAuxPreprocessorWrapper.
 *
 * Purpose:
 *   The Python node predeclares ten preprocessor widgets and ten IMAGE
 *   outputs so workflow JSON can preserve slot values. This extension hides
 *   widgets and rebuilds visible IMAGE output sockets to match slot_count.
 *
 * Dynamic Slot Semantics:
 *   - slot_count is the visible/execution authority.
 *   - Slot widgets above slot_count are hidden, not deleted.
 *   - IMAGE output sockets above slot_count are removed visually because
 *     LiteGraph/ComfyUI does not reliably honor output.hidden for socket
 *     drawing.
 *   - Increasing slot_count recreates missing IMAGE output sockets by name.
 *   - Backend logic still ignores all slots above slot_count.
 *
 * Important Note:
 *   Removing output sockets is the only reliable way to visually collapse
 *   ComfyUI output rows. Links attached to outputs removed by reducing
 *   slot_count may be removed by LiteGraph. Leave a slot visible before saving
 *   or reducing slot_count if you need to keep its link.
 */

const { app } = window.comfyAPI.app;

const MAX_SLOTS = 10;
const SLOT_COUNT_WIDGET = "slot_count";
const UPDATE_BUTTON_LABEL = "Update Visible Slots";

const NODE_CONFIGS = {
    JLC_DynamicAuxPreprocessorWrapper: {
        extensionName: "JLC.DynamicAuxPreprocessorWrapper.Visibility",
        layoutKey: "__jlc_dynamic_aux_preproc_layout",
        installFlag: "__jlc_dynamic_aux_preproc_visibility_installed",
        outputType: "IMAGE",
        outputSlotOptions: {},
        slotWidgetNames(index) {
            return [`preprocessor_${slotSuffix(index)}`];
        },
        slotOutputName(index) {
            return `image_${slotSuffix(index)}`;
        },
    },
};

function slotSuffix(index) {
    return String(index).padStart(2, "0");
}

function getWidgetsByName(node, names) {
    return names
        .map((name) => node.widgets?.find((widget) => widget.name === name))
        .filter(Boolean);
}

function rememberWidgetLayout(widget, layoutKey) {
    if (!widget[layoutKey]) {
        widget[layoutKey] = {
            type: widget.type,
            computeSize: widget.computeSize,
            hidden: widget.hidden,
        };
    }
}

function hideWidget(widget, layoutKey) {
    rememberWidgetLayout(widget, layoutKey);
    widget.type = "hidden";
    widget.computeSize = () => [0, -4];
    widget.hidden = true;
}

function showWidget(widget, layoutKey) {
    const layout = widget[layoutKey];

    if (layout) {
        widget.type = layout.type;
        widget.computeSize = layout.computeSize;
        widget.hidden = layout.hidden ?? false;
    } else {
        widget.hidden = false;
    }
}

function getSlotCount(node) {
    const widget = node.widgets?.find((w) => w.name === SLOT_COUNT_WIDGET);
    const raw = Number.parseInt(widget?.value ?? 1, 10);

    if (!Number.isFinite(raw)) return 1;
    return Math.max(1, Math.min(MAX_SLOTS, raw));
}

function findOutputIndex(node, name) {
    return node.outputs?.findIndex((output) => output.name === name) ?? -1;
}

function hasOutput(node, name) {
    return findOutputIndex(node, name) >= 0;
}

function removeOutputByName(node, name) {
    const index = findOutputIndex(node, name);
    if (index < 0) return false;

    node.removeOutput(index);
    return true;
}

function syncOutputLinkSlots(node) {
    if (!node.outputs || !node.graph?.links) return;

    for (let slot = 0; slot < node.outputs.length; slot++) {
        for (const linkId of node.outputs[slot].links || []) {
            const link = node.graph.links[linkId];
            if (link) link.origin_slot = slot;
        }
    }
}

function ensureOutput(node, name, type, slotOptions) {
    if (hasOutput(node, name)) return;

    node.addOutput(name, type, slotOptions);
}

function rebuildImageOutputs(node, config, count) {
    if (!node.outputs) node.outputs = [];

    // Remove high-numbered outputs first so lower indices are not disturbed.
    for (let i = MAX_SLOTS; i > count; i--) {
        removeOutputByName(node, config.slotOutputName(i));
    }

    // Ensure visible outputs exist. Missing outputs are recreated by name.
    for (let i = 1; i <= count; i++) {
        ensureOutput(
            node,
            config.slotOutputName(i),
            config.outputType,
            config.outputSlotOptions,
        );
    }

    syncOutputLinkSlots(node);
}

function resizeNodeToVisibleWidgets(node) {
    if (!node.computeSize || !node.size) return;

    const currentWidth = node.size[0] ?? 200;
    const computed = node.computeSize();

    if (!computed) return;

    const newWidth = Math.max(currentWidth, computed[0]);
    const newHeight = computed[1];

    if (node.setSize) {
        node.setSize([newWidth, newHeight]);
    } else {
        node.size[0] = newWidth;
        node.size[1] = newHeight;
        node.onResize?.(node.size);
    }
}

const JLC_PRIMARY_BUTTON_BLUE = "#0B8CE9";
const JLC_PRIMARY_BUTTON_TEXT = "#FFFFFF";

function roundedRectPath(ctx, x, y, width, height, radius) {
    const r = Math.min(radius, width / 2, height / 2);

    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + width - r, y);
    ctx.quadraticCurveTo(x + width, y, x + width, y + r);
    ctx.lineTo(x + width, y + height - r);
    ctx.quadraticCurveTo(x + width, y + height, x + width - r, y + height);
    ctx.lineTo(x + r, y + height);
    ctx.quadraticCurveTo(x, y + height, x, y + height - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
}

function stylePrimaryButton(widget) {
    widget.draw = function (ctx, node, widgetWidth, y, widgetHeight) {
        const marginX = 10;
        const marginY = 2;
        const x = marginX;
        const h = Math.max(18, widgetHeight - marginY * 2);
        const w = Math.max(40, widgetWidth - marginX * 2);
        const buttonY = y + marginY;

        ctx.save();

        roundedRectPath(ctx, x, buttonY, w, h, 5);
        ctx.fillStyle = JLC_PRIMARY_BUTTON_BLUE;
        ctx.fill();

        ctx.fillStyle = JLC_PRIMARY_BUTTON_TEXT;
        ctx.font = "bold 12px sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(widget.name, x + w / 2, buttonY + h / 2);

        ctx.restore();
    };
}

function applyVisibleSlotCount(node, config) {
    const count = getSlotCount(node);

    rebuildImageOutputs(node, config, count);

    for (let i = 1; i <= MAX_SLOTS; i++) {
        const visible = i <= count;

        for (const widget of getWidgetsByName(node, config.slotWidgetNames(i))) {
            if (visible) showWidget(widget, config.layoutKey);
            else hideWidget(widget, config.layoutKey);
        }
    }

    const countWidget = node.widgets?.find((w) => w.name === SLOT_COUNT_WIDGET);
    if (countWidget && countWidget.value !== count) {
        countWidget.value = count;
    }

    resizeNodeToVisibleWidgets(node);
    node.setDirtyCanvas?.(true, true);
    node.graph?.setDirtyCanvas?.(true, true);
}

function installDynamicAuxPreprocessorVisibility(node, config) {
    if (node[config.installFlag]) return;
    node[config.installFlag] = true;

    const originalOnConfigure = node.onConfigure;
    node.onConfigure = function () {
        const result = originalOnConfigure?.apply(this, arguments);
        requestAnimationFrame(() => applyVisibleSlotCount(this, config));
        return result;
    };

    const countWidget = node.widgets?.find((w) => w.name === SLOT_COUNT_WIDGET);
    if (countWidget) {
        const originalCallback = countWidget.callback;
        countWidget.callback = function () {
            const result = originalCallback?.apply(this, arguments);
            // Keep explicit button as the stable user-facing operation.
            // The no-canvas callback path helps API-format reloads settle.
            if (!arguments[1]) {
                requestAnimationFrame(() => applyVisibleSlotCount(node, config));
            }
            return result;
        };
    }

    const updateButton = node.addWidget("button", UPDATE_BUTTON_LABEL, null, () => {
        applyVisibleSlotCount(node, config);
    });

    stylePrimaryButton(updateButton);

    requestAnimationFrame(() => applyVisibleSlotCount(node, config));
}

app.registerExtension({
    name: "JLC.DynamicAuxPreprocessorWrapper.Visibility",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const config = NODE_CONFIGS[nodeData?.name];
        if (!config) return;

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalOnNodeCreated?.apply(this, arguments);
            installDynamicAuxPreprocessorVisibility(this, config);
            return result;
        };
    },
});
