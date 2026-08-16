/*
 * JLC Resize Images - Dynamic Slots and Resize Widget Visibility
 * ----------------------------------------------------------------
 *
 * Python predeclares five IMAGE inputs and six fixed IMAGE outputs. This
 * extension:
 *   - exposes only image_1..image_n;
 *   - visually hides inactive individual outputs without removing them;
 *   - draws the fixed sixth batch output immediately after image_n;
 *   - changes the socket layout only when Update Visible Slots is pressed;
 *   - exposes only the numeric widget used by the selected resize mode.
 *
 * Output safety contract:
 *   - node.outputs is never filtered, reordered, spliced, or rebuilt;
 *   - every backend output index remains stable;
 *   - existing links remain attached when slot_count changes;
 *   - inactive output positions are moved outside render/hit-test space.
 *
 * Backend slot_count remains authoritative. Inactive individual outputs return
 * None even though their fixed sockets remain present in the graph schema.
 */

const { app } = window.comfyAPI.app;

const NODE_NAME = "JLC_ResizeImages";
const MAX_SLOTS = 5;
const SLOT_COUNT_WIDGET = "slot_count";
const MODE_WIDGET = "resize_by";
const BATCH_OUTPUT = "batch";
const UPDATE_BUTTON_LABEL = "Update Visible Slots";

const INSTALL_FLAG = "__jlc_resize_images_visibility_installed";
const SOCKET_VISIBILITY_FLAG =
    "__jlc_resize_images_socket_visibility_installed";
const OUTPUT_VISIBILITY_FLAG =
    "__jlc_resize_images_output_visibility_installed";
const ACTIVE_WIDGET_KEY = "__jlc_resize_images_active_widget";
const ACTIVE_SLOT_COUNT_KEY = "__jlc_resize_images_active_slot_count";
const WIDGET_LAYOUT_KEY = "__jlc_resize_images_widget_layout";

const MODE_WIDGET_MAP = {
    "scale by multiplier": "multiplier",
    "scale longer dimension": "longer_size",
    "scale shorter dimension": "shorter_size",
    "scale width": "width",
    "scale height": "height",
    "scale total pixels": "megapixels",
};

const MODE_SPECIFIC_WIDGETS = new Set(Object.values(MODE_WIDGET_MAP));

const JLC_PRIMARY_BUTTON_BLUE = "#0B8CE9";
const JLC_PRIMARY_BUTTON_TEXT = "#FFFFFF";

function imageSlotName(index) {
    return `image_${index}`;
}

function getRequestedSlotCount(node) {
    const widget = node.widgets?.find(
        (candidate) => candidate.name === SLOT_COUNT_WIDGET
    );
    const raw = Number.parseInt(widget?.value ?? 1, 10);
    if (!Number.isFinite(raw)) return 1;
    return Math.max(1, Math.min(MAX_SLOTS, raw));
}

function findInputIndex(node, name) {
    return node.inputs?.findIndex((input) => input.name === name) ?? -1;
}

function ensureImageInput(node, name) {
    if (findInputIndex(node, name) >= 0) return;
    node.addInput(name, "IMAGE", { shape: 7 });
}

function removeInputByName(node, name) {
    const index = findInputIndex(node, name);
    if (index >= 0) node.removeInput(index);
}

function rebuildVisibleSlots(node, count) {
    if (!node.inputs) node.inputs = [];

    for (let index = MAX_SLOTS; index > count; index--) {
        removeInputByName(node, imageSlotName(index));
    }

    for (let index = 1; index <= count; index++) {
        const name = imageSlotName(index);
        ensureImageInput(node, name);
    }
}

function rememberWidgetLayout(widget) {
    if (!widget[WIDGET_LAYOUT_KEY]) {
        widget[WIDGET_LAYOUT_KEY] = {
            hidden: widget.hidden,
            optionsHidden: widget.options?.hidden,
        };
    }
}

function hideWidget(widget) {
    rememberWidgetLayout(widget);
    widget.hidden = true;
    widget.options ??= {};
    widget.options.hidden = true;
}

function showWidget(widget) {
    rememberWidgetLayout(widget);
    const layout = widget[WIDGET_LAYOUT_KEY];

    widget.hidden = layout.hidden ?? false;
    widget.options ??= {};
    if (layout.optionsHidden === undefined) {
        delete widget.options.hidden;
    } else {
        widget.options.hidden = layout.optionsHidden;
    }
}

function installModeSocketVisibility(node) {
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

/*
 * Keep the six backend output objects permanently in their declared order.
 * Only their canvas positions change:
 *   - inactive image outputs are moved far outside render/hit-test space;
 *   - batch keeps backend slot 5 but borrows the visual row immediately after
 *     the final active individual output.
 *
 * This mirrors the defensive getInputPos strategy used for hidden resize-mode
 * sockets and avoids every link-destructive output-array operation.
 */
function installOutputVisibility(node) {
    if (
        node[OUTPUT_VISIBILITY_FLAG] ||
        typeof node.getOutputPos !== "function"
    ) {
        return;
    }

    node[OUTPUT_VISIBILITY_FLAG] = true;
    node[ACTIVE_SLOT_COUNT_KEY] = getRequestedSlotCount(node);

    const originalGetOutputPos = node.getOutputPos;

    node.getOutputPos = function (slot) {
        const output = this.outputs?.[slot];
        const outputName = output?.name;
        const count = Math.max(
            1,
            Math.min(
                MAX_SLOTS,
                Number.parseInt(this[ACTIVE_SLOT_COUNT_KEY] ?? 1, 10) || 1
            )
        );

        const imageMatch = /^image_(\d+)$/.exec(String(outputName ?? ""));
        if (imageMatch && Number.parseInt(imageMatch[1], 10) > count) {
            const out = arguments[1] ?? [0, 0];
            out[0] = (this.pos?.[0] ?? 0) - 1000000;
            out[1] = (this.pos?.[1] ?? 0) - 1000000;
            return out;
        }

        if (outputName === BATCH_OUTPUT && slot !== count) {
            const args = Array.from(arguments);
            args[0] = count;
            return originalGetOutputPos.apply(this, args);
        }

        return originalGetOutputPos.apply(this, arguments);
    };
}

function applyModeVisibility(node) {
    const modeWidget = node.widgets?.find(
        (candidate) => candidate.name === MODE_WIDGET
    );
    const selectedMode = String(
        modeWidget?.value ?? "scale longer dimension"
    );
    const activeWidgetName =
        MODE_WIDGET_MAP[selectedMode] ?? "longer_size";

    node[ACTIVE_WIDGET_KEY] = activeWidgetName;

    for (const widget of node.widgets ?? []) {
        if (!MODE_SPECIFIC_WIDGETS.has(widget.name)) continue;
        if (widget.name === activeWidgetName) showWidget(widget);
        else hideWidget(widget);
    }

    node._widgetSlotsDirty = true;
}

function getOutputRowHeight(node, count) {
    if (typeof node.getOutputPos !== "function") return 20;

    const firstOutputIndex = node.outputs?.findIndex(
        (output) => output?.name === imageSlotName(1)
    ) ?? -1;
    const secondVisibleIndex =
        count >= 2
            ? node.outputs?.findIndex(
                  (output) => output?.name === imageSlotName(2)
              ) ?? -1
            : node.outputs?.findIndex(
                  (output) => output?.name === BATCH_OUTPUT
              ) ?? -1;

    if (firstOutputIndex < 0 || secondVisibleIndex < 0) return 20;

    const firstPos = node.getOutputPos(firstOutputIndex, [0, 0]);
    const secondPos = node.getOutputPos(secondVisibleIndex, [0, 0]);
    const rowHeight = Math.abs((secondPos?.[1] ?? 0) - (firstPos?.[1] ?? 0));

    return Number.isFinite(rowHeight) && rowHeight > 0 ? rowHeight : 20;
}

function getHiddenOutputHeight(node) {
    const count = Math.max(
        1,
        Math.min(
            MAX_SLOTS,
            Number.parseInt(node[ACTIVE_SLOT_COUNT_KEY] ?? 1, 10) || 1
        )
    );

    /*
     * The backend always declares image_1..image_5 plus batch, but the
     * frontend visually exposes only image_1..image_n plus batch.
     *
     * Do NOT derive this correction from node.inputs.length. ComfyUI keeps
     * widget-backed/convertible sockets in node.inputs even when this
     * extension moves those sockets out of render space. Counting them made
     * the previous compact-height calculation conclude that there were no
     * hidden rows at all.
     *
     * Exactly MAX_SLOTS - count individual output rows are hidden.
     */
    const hiddenOutputRows = Math.max(0, MAX_SLOTS - count);
    return hiddenOutputRows * getOutputRowHeight(node, count);
}

function resizeNodeToVisibleContent(node) {
    if (typeof node.computeSize !== "function" || !node.size) return;

    const computed = node.computeSize();
    if (!computed) return;

    const currentWidth = node.size[0] ?? computed[0] ?? 260;
    const targetWidth = Math.max(currentWidth, computed[0] ?? currentWidth);
    const targetHeight = Math.max(
        1,
        (computed[1] ?? node.size[1] ?? 1) - getHiddenOutputHeight(node)
    );

    if (typeof node.setSize === "function") {
        node.setSize([targetWidth, targetHeight]);
    } else {
        node.size[0] = targetWidth;
        node.size[1] = targetHeight;
        node.onResize?.(node.size);
    }
}

function dirtyNode(node) {
    node.setDirtyCanvas?.(true, true);
    node.graph?.setDirtyCanvas?.(true, true);
}

function applyVisibleSlotCount(node) {
    const count = getRequestedSlotCount(node);
    node[ACTIVE_SLOT_COUNT_KEY] = count;
    rebuildVisibleSlots(node, count);

    const countWidget = node.widgets?.find(
        (candidate) => candidate.name === SLOT_COUNT_WIDGET
    );
    if (countWidget && countWidget.value !== count) {
        countWidget.value = count;
    }

    applyModeVisibility(node);
    resizeNodeToVisibleContent(node);
    dirtyNode(node);
}

function scheduleInitialLayout(node) {
    requestAnimationFrame(() => {
        applyVisibleSlotCount(node);
        requestAnimationFrame(() => applyVisibleSlotCount(node));
    });
}

function roundedRectPath(ctx, x, y, width, height, radius) {
    const r = Math.min(radius, width / 2, height / 2);
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + width - r, y);
    ctx.quadraticCurveTo(x + width, y, x + width, y + r);
    ctx.lineTo(x + width, y + height - r);
    ctx.quadraticCurveTo(
        x + width,
        y + height,
        x + width - r,
        y + height
    );
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

function install(node) {
    if (node[INSTALL_FLAG]) return;
    node[INSTALL_FLAG] = true;

    installModeSocketVisibility(node);
    installOutputVisibility(node);

    const originalOnConfigure = node.onConfigure;
    node.onConfigure = function () {
        const result = originalOnConfigure?.apply(this, arguments);
        scheduleInitialLayout(this);
        return result;
    };

    const modeWidget = node.widgets?.find(
        (candidate) => candidate.name === MODE_WIDGET
    );
    if (modeWidget) {
        const originalCallback = modeWidget.callback;
        modeWidget.callback = function () {
            const result = originalCallback?.apply(this, arguments);
            requestAnimationFrame(() => {
                applyModeVisibility(node);
                resizeNodeToVisibleContent(node);
                dirtyNode(node);
            });
            return result;
        };
    }

    // Deliberately do not install a slot_count callback. The button is the
    // only live-layout commit action for this node.
    const updateButton = node.addWidget(
        "button",
        UPDATE_BUTTON_LABEL,
        null,
        () => applyVisibleSlotCount(node)
    );
    stylePrimaryButton(updateButton);

    scheduleInitialLayout(node);
}

app.registerExtension({
    name: "JLC.ResizeImages.Visibility",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_NAME) return;

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalOnNodeCreated?.apply(this, arguments);
            install(this);
            return result;
        };
    },
});
