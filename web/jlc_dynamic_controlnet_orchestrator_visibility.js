/*
 * JLC Dynamic ControlNet Orchestrator Visibility Helpers
 * ------------------------------------------------------
 * Frontend companion for the wired JLC ControlNet Orchestrator.
 */

const { app } = window.comfyAPI.app;

const MAX_SLOTS = 5;
const SLOT_COUNT_WIDGET = "slot_count";
const UPDATE_BUTTON_LABEL = "Update Visible Slots";
const NODE_NAMES = new Set([
    "JLC_ControlNetOrchestrator",
    "JLC_DynamicControlNetOrchestrator",
]);

const JLC_PRIMARY_BUTTON_BLUE = "#0B8CE9";
const JLC_PRIMARY_BUTTON_TEXT = "#FFFFFF";
const LAYOUT_KEY = "__jlc_dynamic_controlnet_orchestrator_layout";
const INSTALL_FLAG = "__jlc_dynamic_controlnet_orchestrator_installed";

function slotSuffix(index) {
    return String(index).padStart(2, "0");
}

function slotWidgetNames(index) {
    const suffix = slotSuffix(index);
    return [
        `enabled_${suffix}`,
        `strength_${suffix}`,
        `start_${suffix}`,
        `end_${suffix}`,
        `weight_${suffix}`,
    ];
}

function slotInputNames(index) {
    const suffix = slotSuffix(index);
    return [
        { name: `control_net_${suffix}`, type: "CONTROL_NET", options: {} },
        { name: `image_${suffix}`, type: "IMAGE", options: { shape: 7 } },
    ];
}

function getWidgetsByName(node, names) {
    return names
        .map((name) => node.widgets?.find((widget) => widget.name === name))
        .filter(Boolean);
}

function rememberWidgetLayout(widget) {
    if (!widget[LAYOUT_KEY]) {
        widget[LAYOUT_KEY] = {
            type: widget.type,
            computeSize: widget.computeSize,
            hidden: widget.hidden,
        };
    }
}

function hideWidget(widget) {
    rememberWidgetLayout(widget);
    widget.type = "hidden";
    widget.computeSize = () => [0, -4];
    widget.hidden = true;
}

function showWidget(widget) {
    const layout = widget[LAYOUT_KEY];
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

function findInputIndex(node, name) {
    return node.inputs?.findIndex((input) => input.name === name) ?? -1;
}

function hasInput(node, name) {
    return findInputIndex(node, name) >= 0;
}

function removeInputByName(node, name) {
    const index = findInputIndex(node, name);
    if (index < 0) return false;
    node.removeInput(index);
    return true;
}

function ensureInput(node, name, type, options) {
    if (hasInput(node, name)) return;
    node.addInput(name, type, options);
}

function rebuildSlotInputs(node, count) {
    if (!node.inputs) node.inputs = [];

    for (let i = MAX_SLOTS; i > count; i--) {
        for (const input of slotInputNames(i)) removeInputByName(node, input.name);
    }

    for (let i = 1; i <= count; i++) {
        for (const input of slotInputNames(i)) {
            ensureInput(node, input.name, input.type, input.options);
        }
    }
}

function resizeNodeToVisibleWidgets(node) {
    if (!node.computeSize || !node.size) return;
    const currentWidth = node.size[0] ?? 200;
    const computed = node.computeSize();
    if (!computed) return;
    const newWidth = Math.max(currentWidth, computed[0]);
    const newHeight = computed[1];
    if (node.setSize) node.setSize([newWidth, newHeight]);
    else {
        node.size[0] = newWidth;
        node.size[1] = newHeight;
        node.onResize?.(node.size);
    }
}

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

function applyVisibleSlotCount(node) {
    const count = getSlotCount(node);

    rebuildSlotInputs(node, count);

    for (let i = 1; i <= MAX_SLOTS; i++) {
        const visible = i <= count;
        for (const widget of getWidgetsByName(node, slotWidgetNames(i))) {
            if (visible) showWidget(widget);
            else hideWidget(widget);
        }
    }

    const countWidget = node.widgets?.find((w) => w.name === SLOT_COUNT_WIDGET);
    if (countWidget && countWidget.value !== count) countWidget.value = count;

    resizeNodeToVisibleWidgets(node);
    node.setDirtyCanvas?.(true, true);
    node.graph?.setDirtyCanvas?.(true, true);
}

function install(node) {
    if (node[INSTALL_FLAG]) return;
    node[INSTALL_FLAG] = true;

    const originalOnConfigure = node.onConfigure;
    node.onConfigure = function () {
        const result = originalOnConfigure?.apply(this, arguments);
        requestAnimationFrame(() => applyVisibleSlotCount(this));
        return result;
    };

    const countWidget = node.widgets?.find((w) => w.name === SLOT_COUNT_WIDGET);
    if (countWidget) {
        const originalCallback = countWidget.callback;
        countWidget.callback = function () {
            const result = originalCallback?.apply(this, arguments);
            if (!arguments[1]) requestAnimationFrame(() => applyVisibleSlotCount(node));
            return result;
        };
    }

    const updateButton = node.addWidget("button", UPDATE_BUTTON_LABEL, null, () => {
        applyVisibleSlotCount(node);
    });
    stylePrimaryButton(updateButton);

    requestAnimationFrame(() => applyVisibleSlotCount(node));
}

app.registerExtension({
    name: "JLC.DynamicControlNetOrchestrator.Visibility",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!NODE_NAMES.has(nodeData?.name)) return;
        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalOnNodeCreated?.apply(this, arguments);
            install(this);
            return result;
        };
    },
});
