/*
 * JLC Text Concatenate — draft 2
 *
 * One combined output (slot 0) and 1–32 independent STRING output slots.
 * The fixed backend declaration supplies all 32 slots; this frontend hides
 * unused trailing ones. Physical slot index remains the section's identity.
 *
 * The serialized `sections` STRING widget accepts legacy string arrays and
 * now saves objects: {id, text, enabled, order}.
 */

import { app } from "../../scripts/app.js";

const NODE_CLASS = "JLC_TextConcatenate";
const MAX_SECTIONS = 32;
const MIN_WIDTH = 420;
const MIN_TEXT_HEIGHT = 30;
const MAX_TEXT_HEIGHT = 210;
const ROW_HEADER_HEIGHT = 26;
const ROW_GAP = 8;
const ROW_INNER_GAP = 4;

function slotName(index) {
    return `section_${String(index + 1).padStart(2, "0")}`;
}

function freshSection(index) {
    return { id: index + 1, text: "", enabled: true, order: index + 1 };
}

function parseSections(json) {
    const raw = JSON.parse(json);
    if (!Array.isArray(raw)) {
        throw new Error("The sections widget must contain a JSON array.");
    }
    if (raw.length > MAX_SECTIONS) {
        throw new Error(`A maximum of ${MAX_SECTIONS} sections is supported.`);
    }

    const sections = raw.map((value, index) => {
        if (typeof value === "string") {
            return { ...freshSection(index), text: value };
        }
        if (!value || typeof value !== "object" || Array.isArray(value)) {
            throw new Error(`Invalid section ${index + 1}.`);
        }
        const order = Number(value.order ?? index + 1);
        return {
            id: index + 1, // Output identity is the physical position.
            text: String(value.text ?? ""),
            enabled: value.enabled !== false,
            order: Number.isSafeInteger(order) && order !== 0
                ? order
                : index + 1,
        };
    });
    if (!sections.length) sections.push(freshSection(0));
    return sections;
}

function hasLinks(slot) {
    return Array.isArray(slot?.links) && slot.links.length > 0;
}

app.registerExtension({
    name: "JLC.TextConcatenate",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_CLASS) return;

        const originalCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            originalCreated?.apply(this, arguments);
            const node = this;

            const storage = node.widgets?.find(widget => widget.name === "sections");
            const countWidget = node.widgets?.find(
                widget => widget.name === "number_of_sections"
            );
            if (!storage || !countWidget) {
                console.error("[JLC Text Concatenate] Missing required backend widgets.");
                return;
            }

            // Store all persistent row data in one widget, for workflow saving.
            storage.type = "hidden";
            storage.computeSize = () => [0, -4];
            storage.draw = () => {};

            let sections = [];
            let editorHeight = 42;
            let removed = false;
            let observer;
            let measuredWidth = 0;

            const container = document.createElement("div");
            Object.assign(container.style, {
                display: "flex",
                flexDirection: "column",
                gap: `${ROW_GAP}px`,
                padding: "6px",
                boxSizing: "border-box",
                width: "100%",
            });

            // LiteGraph's standard numeric widget remains visible. This button
            // applies its proposed count to sections and output sockets.
            node.addWidget("button", "Update Slots", null, () => updateSlots(), {
                serialize: false,
            });

            const editor = node.addDOMWidget("section_editor", "jlc_sections", container, {
                serialize: false,
                getMinHeight: () => editorHeight,
                getMaxHeight: () => editorHeight,
                hideOnZoom: false,
            });
            editor.computeSize = width => [width, editorHeight];

            function dirty() {
                node.graph?.setDirtyCanvas?.(true, true);
            }

            function save() {
                storage.value = JSON.stringify(sections);
                dirty();
            }

            function resize() {
                // Preserve the original fix: never measure a hidden,
                // disconnected, or zero-width editor.
                if (
                    removed ||
                    !container.isConnected ||
                    !container.getClientRects().length ||
                    container.clientWidth <= 0
                ) return;

                let measuredHeight = 12; // container's vertical padding
                for (const row of container.children) {
                    const textarea = row.querySelector("textarea");
                    if (!textarea) continue;
                    textarea.style.height = `${MIN_TEXT_HEIGHT}px`;
                    const textHeight = Math.min(
                        MAX_TEXT_HEIGHT,
                        Math.max(MIN_TEXT_HEIGHT, textarea.scrollHeight + 2)
                    );
                    textarea.style.height = `${textHeight}px`;
                    measuredHeight += ROW_HEADER_HEIGHT + ROW_INNER_GAP + textHeight;
                }
                measuredHeight += Math.max(0, sections.length - 1) * ROW_GAP;

                editorHeight = Math.ceil(measuredHeight);
                const computedHeight = node.computeSize?.()[1];
                if (
                    Number.isFinite(computedHeight) &&
                    Math.abs((node.size?.[1] ?? 0) - computedHeight) > 1
                ) {
                    node.setSize([Math.max(MIN_WIDTH, node.size[0]), computedHeight]);
                }
                dirty();
            }

            function scheduleResize() {
                requestAnimationFrame(() => {
                    if (!removed) resize();
                });
            }

            function ensureOutputs(count) {
                node.outputs ??= [];

                // Output zero always keeps its old name and index.
                if (!node.outputs.length) node.addOutput("text", "STRING");
                node.outputs[0].name = "text";
                node.outputs[0].type = "STRING";

                const expected = count + 1;

                // Remove only trailing, disconnected outputs. No reindexing
                // of any surviving output is ever performed.
                while (node.outputs.length > expected) {
                    const last = node.outputs.at(-1);
                    if (hasLinks(last)) {
                        console.warn(
                            "[JLC Text Concatenate] Retaining connected trailing output."
                        );
                        break;
                    }
                    node.removeOutput(node.outputs.length - 1);
                }
                while (node.outputs.length < expected) {
                    node.addOutput(slotName(node.outputs.length - 1), "STRING");
                }
                for (let index = 0; index < node.outputs.length - 1; index++) {
                    const output = node.outputs[index + 1];
                    output.name = slotName(index);
                    output.type = "STRING";
                }
            }

            function buildRow(section, index) {
                const row = document.createElement("div");
                Object.assign(row.style, {
                    display: "flex",
                    flexDirection: "column",
                    gap: `${ROW_INNER_GAP}px`,
                    minWidth: "0",
                });

                const header = document.createElement("div");
                Object.assign(header.style, {
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    gap: "8px",
                    minHeight: `${ROW_HEADER_HEIGHT}px`,
                    fontSize: "12px",
                    color: "var(--input-text, #ddd)",
                });

                const sectionLabel = document.createElement("label");
                Object.assign(sectionLabel.style, {
                    display: "inline-flex",
                    alignItems: "center",
                    gap: "6px",
                    whiteSpace: "nowrap",
                });
                const enabled = document.createElement("input");
                enabled.type = "checkbox";
                enabled.checked = section.enabled;
                enabled.setAttribute("aria-label", `Include section ${index + 1} in combined text`);
                enabled.addEventListener("change", () => {
                    section.enabled = enabled.checked;
                    save();
                });
                sectionLabel.append(
                    enabled,
                    document.createTextNode(`Section ${String(index + 1).padStart(2, "0")}`)
                );

                const orderLabel = document.createElement("label");
                Object.assign(orderLabel.style, {
                    display: "inline-flex",
                    alignItems: "center",
                    gap: "4px",
                    whiteSpace: "nowrap",
                });
                orderLabel.append(document.createTextNode("Order"));
                const orderInput = document.createElement("input");
                orderInput.type = "number";
                orderInput.step = "1";
                orderInput.value = String(section.order);
                orderInput.title = "Positive: earlier first. -1: last, -2: before -1. No zero.";
                orderInput.setAttribute("aria-label", `Concatenation order for section ${index + 1}`);
                Object.assign(orderInput.style, {
                    width: "54px",
                    height: "24px",
                    padding: "2px 4px",
                    boxSizing: "border-box",
                    fontSize: "12px",
                    color: "var(--input-text, #ddd)",
                    background: "var(--comfy-input-bg, #222)",
                    border: "1px solid var(--border-color, #555)",
                    borderRadius: "4px",
                });
                function commitOrder() {
                    const numeric = Number(orderInput.value);
                    if (Number.isSafeInteger(numeric) && numeric !== 0) {
                        section.order = numeric;
                        save();
                    }
                    orderInput.value = String(section.order);
                }
                orderInput.addEventListener("change", commitOrder);
                orderInput.addEventListener("blur", commitOrder);
                orderLabel.appendChild(orderInput);

                const outputLabel = document.createElement("span");
                outputLabel.textContent = slotName(index);
                outputLabel.style.opacity = "0.7";
                outputLabel.style.whiteSpace = "nowrap";

                const right = document.createElement("div");
                Object.assign(right.style, {
                    display: "flex",
                    alignItems: "center",
                    gap: "9px",
                });
                right.append(orderLabel, outputLabel);
                header.append(sectionLabel, right);

                const textarea = document.createElement("textarea");
                textarea.value = section.text;
                textarea.rows = 1;
                textarea.placeholder = "Enter text (Enter inserts a line break)";
                textarea.setAttribute("aria-label", `Text section ${index + 1}`);
                Object.assign(textarea.style, {
                    width: "100%",
                    minHeight: `${MIN_TEXT_HEIGHT}px`,
                    flexShrink: "0",
                    boxSizing: "border-box",
                    resize: "none",
                    overflowY: "auto",
                    padding: "4px 6px",
                    lineHeight: "20px",
                    fontSize: "13px",
                    fontFamily: "monospace",
                    color: "var(--input-text, #ddd)",
                    background: "var(--comfy-input-bg, #222)",
                    border: "1px solid var(--border-color, #555)",
                    borderRadius: "4px",
                });
                textarea.addEventListener("input", () => {
                    section.text = textarea.value;
                    save();
                    resize();
                });

                // Keep editor keystrokes out of ComfyUI canvas shortcuts, but
                // allow native Enter, Shift+Enter, selection and undo behavior.
                for (const input of [textarea, orderInput]) {
                    input.addEventListener("keydown", event => event.stopPropagation());
                }
                row.append(header, textarea);
                return row;
            }

            function renderRows() {
                const fragment = document.createDocumentFragment();
                sections.forEach((section, index) => {
                    fragment.appendChild(buildRow(section, index));
                });
                container.replaceChildren(fragment);
                scheduleResize();
            }

            function restore() {
                try {
                    sections = parseSections(storage.value);
                } catch (error) {
                    console.error("[JLC Text Concatenate] Invalid saved sections:", error);
                    // Preserve unreadable raw data rather than overwrite it.
                    return;
                }

                // If an imported workflow contains a connected output beyond
                // the stored row array, retain that physical slot too. Never
                // silently discard its link while recovering the node.
                const highestConnected = (node.outputs ?? []).reduce(
                    (highest, output, slot) =>
                        slot > 0 && hasLinks(output) ? Math.max(highest, slot) : highest,
                    0
                );
                while (sections.length < Math.min(highestConnected, MAX_SECTIONS)) {
                    sections.push(freshSection(sections.length));
                }

                // An applied section count comes from the serialized sections.
                // An unapplied numeric widget change is intentionally discarded.
                countWidget.value = sections.length;
                ensureOutputs(sections.length);
                renderRows();
                save(); // also upgrades legacy arrays on first load
            }

            function updateSlots() {
                const requested = Number(countWidget.value);
                if (!Number.isSafeInteger(requested) ||
                    requested < 1 || requested > MAX_SECTIONS) {
                    countWidget.value = sections.length;
                    window.alert(`Choose 1–${MAX_SECTIONS} sections.`);
                    return;
                }
                if (requested < sections.length) {
                    for (let index = requested; index < sections.length; index++) {
                        if (sections[index].text !== "" || hasLinks(node.outputs?.[index + 1])) {
                            countWidget.value = sections.length;
                            window.alert(
                                `Cannot remove section ${index + 1}: clear its text and ` +
                                "disconnect its output first."
                            );
                            return;
                        }
                    }
                }

                sections = sections.slice(0, requested);
                while (sections.length < requested) {
                    sections.push(freshSection(sections.length));
                }
                ensureOutputs(sections.length);
                renderRows();
                save();
                scheduleResize();
            }

            const originalConfigure = node.onConfigure;
            node.onConfigure = function () {
                originalConfigure?.apply(this, arguments);
                restore();
            };

            observer = new ResizeObserver(() => {
                const width = container.clientWidth;
                if (width <= 0) {
                    measuredWidth = 0;
                    return;
                }
                if (width !== measuredWidth) {
                    measuredWidth = width;
                    resize();
                }
            });
            observer.observe(container);

            const originalRemoved = node.onRemoved;
            node.onRemoved = function () {
                removed = true;
                observer.disconnect();
                return originalRemoved?.apply(this, arguments);
            };

            node.size[0] = Math.max(node.size[0], MIN_WIDTH);
            restore();
        };
    },
});
