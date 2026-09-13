/*
 * JLC Seed Display + Randomize Replay
 * -----------------------------------
 *
 * JLC ComfyUI Nodes Collection
 *   Frontend companion for JLC_SeedGenerator.
 *
 * Responsibilities
 * ----------------
 * 1. Preserve the user's visible base seed after queued submissions.
 * 2. Display the seed actually reported by backend execution.
 * 3. Treat -1 as an intentional "unseeded" sentinel and prevent ComfyUI's
 *    native control-after-generate logic from mutating it.
 * 4. Keep ordinary randomization away from the -1 sentinel even though the
 *    Python widget must expose min=-1 so the user can enter it deliberately.
 * 5. Record and replay the actual seed sequence used by native Randomize mode.
 *
 * Randomize replay is deliberately frontend-owned because ComfyUI applies
 * control-after-generate behavior while prompts are being queued, before the
 * backend node executes. The backend remains a simple, explicit seed source.
 *
 * Copyright (c) 2026 J. L. Córdova
 * Released under the MIT License.
 */

const { app } = window.comfyAPI.app;

const NODE_NAME = "JLC_SeedGenerator";

const SEED_WIDGET = "seed";
const REPLAY_WIDGET = "randomize_replay";
const SPACER_WIDGET = "spacer";
const CONTROL_WIDGET = "control_after_generate";

const DISPLAY_PROPERTY = "jlc_seed";
const LAST_SOURCE_PROPERTY = "jlc_seed_last_source";
const REPLAY_SEQUENCE_PROPERTY = "jlc_seed_replay_sequence";
const STATUS_PROPERTY = "jlc_seed_status";

const UNSEEDED_TEXT = "-1";
const COMFY_SAFE_RANDOM_MAX = 1125899906842624;
const VALID_REPLAY_MODES = new Set(["off", "record", "replay"]);
const VALID_CONTROL_MODES = new Set(["fixed", "increment", "decrement", "randomize"]);

const PANEL_HEIGHT = 58;
const PANEL_MARGIN_X = 10;
const PANEL_MARGIN_Y = 4;
const PANEL_RADIUS = 7;

const COLORS = {
    panel: "#202327",
    panelBorder: "#3a4047",
    primary: "#e8edf2",
    secondary: "#9da7b1",
    seeded: "#62d394",
    unseeded: "#f4bf69",
    record: "#57c7ff",
    replay: "#bc8cff",
    warning: "#ff9f62",
    pillText: "#111418",
};

let queueSessionCounter = 0;

function firstPayloadValue(value) {
    if (value === undefined || value === null) return undefined;

    if (Array.isArray(value)) {
        return firstPayloadValue(value[0]);
    }

    if (typeof value === "object") {
        if (value.seed !== undefined) return firstPayloadValue(value.seed);
        if (value.value !== undefined) return firstPayloadValue(value.value);
        return undefined;
    }

    return value;
}

function normalizeText(value) {
    const raw = firstPayloadValue(value);
    if (raw === undefined || raw === null || raw === "") return undefined;
    return String(raw).trim();
}

function isNonnegativeSeedText(value) {
    const text = normalizeText(value);
    return text !== undefined && /^(0|[1-9]\d*)$/.test(text);
}

function readExecutionValue(message, key) {
    return normalizeText(message?.[key]) ?? normalizeText(message?.ui?.[key]);
}

function readSeedFromExecutionMessage(message) {
    return (
        readExecutionValue(message, "jlc_seed") ??
        readExecutionValue(message, "seed")
    );
}

function isJLCSeedNode(node) {
    return (
        node?.comfyClass === NODE_NAME ||
        node?.type === NODE_NAME ||
        node?.constructor?.comfyClass === NODE_NAME
    );
}

function getSeedNodes() {
    return (app.graph?._nodes ?? []).filter(isJLCSeedNode);
}

function getWidget(node, name) {
    return node.widgets?.find((widget) => widget.name === name);
}

function getSeedWidget(node) {
    return getWidget(node, SEED_WIDGET);
}

function ensureSeedWidgetSentinelRange(node) {
    const widget = getSeedWidget(node);
    if (!widget) return;

    // Do not rely exclusively on the backend node-definition schema here.
    // Some ComfyUI frontend builds normalize seed widgets back to min=0
    // during widget construction/configuration.  -1 is a deliberate JLC
    // sentinel, so enforce the live widget range after the widget exists.
    widget.options = widget.options || {};
    widget.options.min = -1;
    widget.options.step2 = 1;
    widget.options.precision = 0;

    // Legacy LiteGraph numeric widgets also consult `step` (historically 10x
    // step2 for INT widgets). Preserve that convention when absent.
    if (widget.options.step === undefined || widget.options.step === null) {
        widget.options.step = 10;
    }
}

function getReplayWidget(node) {
    return getWidget(node, REPLAY_WIDGET);
}

function getControlWidget(node) {
    const direct = getWidget(node, CONTROL_WIDGET);
    if (direct) return direct;

    // Fallback for frontend variants that rename/display the control widget
    // differently but still link it to the seed widget.
    const seedWidget = getSeedWidget(node);
    return seedWidget?.linkedWidgets?.find((widget) => {
        const values = widget?.options?.values;
        return (
            Array.isArray(values) &&
            values.includes("fixed") &&
            values.includes("randomize") &&
            typeof widget.beforeQueued === "function" &&
            typeof widget.afterQueued === "function"
        );
    });
}

function readSeedWidgetRaw(node) {
    return getSeedWidget(node)?.value;
}

function readSeedWidgetText(node) {
    return normalizeText(readSeedWidgetRaw(node));
}

function sanitizeReplayWidget(node) {
    const widget = getReplayWidget(node);
    if (!widget) return;

    const value = String(widget.value ?? "off").toLowerCase();
    if (VALID_REPLAY_MODES.has(value)) return;

    // Compatibility with workflows saved before randomize_replay existed.
    // ComfyUI serializes widget values positionally, so an old spacer string
    // can land in this newly inserted combo on first load. It carried no user
    // state, so normalize the new selector to its safe default.
    widget.value = "off";
}

function readReplayMode(node) {
    sanitizeReplayWidget(node);
    const value = String(getReplayWidget(node)?.value ?? "off").toLowerCase();
    return VALID_REPLAY_MODES.has(value) ? value : "off";
}

function readControlMode(node) {
    const value = String(getControlWidget(node)?.value ?? "fixed").toLowerCase();
    return VALID_CONTROL_MODES.has(value) ? value : "fixed";
}

function seedTextToWidgetValue(seedText) {
    const text = normalizeText(seedText);
    if (text === undefined) return undefined;

    // Recorded native-randomized seeds are within ComfyUI frontend's safe
    // integer range. Prefer a Number for numeric widgets. Fall back to the
    // exact string only for unusually large values so we do not silently alter
    // their decimal representation here.
    const numeric = Number(text);
    if (Number.isSafeInteger(numeric)) return numeric;
    return text;
}

function setSeedWidget(node, value, { callCallback = false } = {}) {
    const widget = getSeedWidget(node);
    if (!widget || value === undefined || value === null) return;

    const currentText = normalizeText(widget.value);
    const nextText = normalizeText(value);
    if (currentText === nextText) return;

    widget.value = value;

    if (callCallback) {
        widget.callback?.(widget.value, app.canvas, node, node.pos, undefined);
    }

    node.setDirtyCanvas?.(true, true);
}

function setSeedWidgetFromText(node, seedText, options = {}) {
    const value = seedTextToWidgetValue(seedText);
    if (value === undefined) return;
    setSeedWidget(node, value, options);
}

function ensureNodeProperties(node) {
    node.properties = node.properties || {};
    return node.properties;
}

function getStoredReplaySequence(node) {
    const raw = ensureNodeProperties(node)[REPLAY_SEQUENCE_PROPERTY];
    if (!Array.isArray(raw)) return [];

    return raw
        .map((value) => normalizeText(value))
        .filter((value) => value !== undefined && isNonnegativeSeedText(value));
}

function setStoredReplaySequence(node, sequence) {
    const clean = (sequence ?? [])
        .map((value) => normalizeText(value))
        .filter((value) => value !== undefined && isNonnegativeSeedText(value));

    ensureNodeProperties(node)[REPLAY_SEQUENCE_PROPERTY] = clean;
    node.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
    return clean;
}

function setStatus(node, status) {
    ensureNodeProperties(node)[STATUS_PROPERTY] = status || "";
    node.setDirtyCanvas?.(true, true);
}

function setDisplayedSeed(node, seedText, source = "manual") {
    const text = normalizeText(seedText);
    if (text === undefined) return;

    const properties = ensureNodeProperties(node);
    properties[DISPLAY_PROPERTY] = text;
    properties[LAST_SOURCE_PROPERTY] = source;
    node.setDirtyCanvas?.(true, true);
}

function updateDisplayedPreview(node, { force = false } = {}) {
    const seedText = readSeedWidgetText(node);
    if (seedText === undefined) return;

    const properties = ensureNodeProperties(node);
    if (force || properties[DISPLAY_PROPERTY] === undefined) {
        setDisplayedSeed(node, seedText, force ? "preview" : "initial");
    } else {
        node.setDirtyCanvas?.(true, true);
    }
}

function updateDisplayedExecution(node, message) {
    const seedText = readSeedFromExecutionMessage(message);
    if (seedText === undefined) return;

    // Backend execution is authoritative for the display. Recording itself is
    // performed at queue time because JLC owns the randomized seed before the
    // prompt is serialized; this avoids depending on execution callbacks,
    // caching, or post-queue widget mutation.
    setDisplayedSeed(node, seedText, "execution");
}

function generateRandomSeed(node) {
    const widget = getSeedWidget(node);
    const options = widget?.options ?? {};

    const rawMax = Number(options.max);
    const max = Number.isFinite(rawMax)
        ? Math.min(COMFY_SAFE_RANDOM_MAX, Math.max(0, rawMax))
        : COMFY_SAFE_RANDOM_MAX;

    const rawStep = Number(options.step2);
    const step = Number.isFinite(rawStep) && rawStep > 0 ? rawStep : 1;

    const slots = Math.max(1, Math.floor(max / step));
    return Math.floor(Math.random() * slots) * step;
}

function appendRecordedSeed(node, seedText) {
    const text = normalizeText(seedText);
    if (!isNonnegativeSeedText(text)) return getStoredReplaySequence(node);

    const sequence = getStoredReplaySequence(node);
    sequence.push(String(text));
    return setStoredReplaySequence(node, sequence);
}

function normalizeBatchCount(value) {
    const count = Number(value);
    if (!Number.isInteger(count) || count <= 0) return null;
    return count;
}

function beginQueueSession(node, batchCount) {
    const baseRaw = readSeedWidgetRaw(node);
    const baseText = normalizeText(baseRaw);
    if (baseText === undefined) return null;

    const controlMode = readControlMode(node);
    const replayMode = readReplayMode(node);
    const sequence = getStoredReplaySequence(node);
    const sessionId = ++queueSessionCounter;

    const context = {
        id: sessionId,
        baseRaw,
        baseText,
        controlMode,
        replayMode,
        replaySequence: sequence,
        replayCursor: Number.isInteger(node.__jlc_seed_replay_cursor)
            ? node.__jlc_seed_replay_cursor
            : 0,
        replayActive: false,
        replayBlocked: false,
        recordActive: false,
    };

    node.__jlc_seed_queue_context = context;
    node.__jlc_seed_replay_state = null;

    if (baseText === UNSEEDED_TEXT) {
        node.__jlc_seed_record_session = null;
        setStatus(node, "unseeded sentinel locked");
    } else if (controlMode === "randomize" && replayMode === "record") {
        context.recordActive = true;
        node.__jlc_seed_record_session = {
            id: sessionId,
            active: true,
            count: 0,
            expected: normalizeBatchCount(batchCount),
        };
        setStatus(node, `recording · ${sequence.length} already stored`);
    } else if (controlMode === "randomize" && replayMode === "replay") {
        node.__jlc_seed_record_session = null;

        if (sequence.length > 0) {
            context.replayActive = true;
            const startIndex = context.replayCursor % sequence.length;
            node.__jlc_seed_replay_state = {
                cursor: startIndex,
                total: sequence.length,
            };
            setStatus(node, `replay ready · ${sequence.length} stored`);
        } else {
            // Fail deterministic: do not silently fall back to fresh random
            // values when the user explicitly asked for replay.
            context.replayBlocked = true;
            setStatus(node, "replay requested · no recorded sequence");
            console.warn(
                "[JLC Seed Generator] Replay requested, but no recorded random seed sequence exists. " +
                "This queue will hold the visible base seed instead of generating fresh random seeds."
            );
        }
    } else {
        node.__jlc_seed_record_session = null;

        if (replayMode !== "off" && controlMode !== "randomize") {
            setStatus(node, `${replayMode} idle · set control to randomize`);
        } else {
            setStatus(node, `${controlMode}`);
        }
    }

    setDisplayedSeed(node, baseText, "queue_capture");
    return { node, baseRaw, baseText, context };
}

function finishQueueSession(record, succeeded) {
    const node = record?.node;
    if (!node) return;

    const context = node.__jlc_seed_queue_context;
    if (context?.id === record.context?.id) {
        node.__jlc_seed_queue_context = null;
    }

    const recording = node.__jlc_seed_record_session;
    if (recording?.id === record.context?.id) {
        recording.active = false;
        if (succeeded) {
            const total = getStoredReplaySequence(node).length;
            setStatus(node, `recorded · ${total} stored`);
        }
    }

    if (!succeeded) {
        setStatus(node, "queue failed");
    }
}

function restoreSeedBaseValues(records) {
    for (const record of records ?? []) {
        if (!record?.node) continue;

        // Restore the exact pre-queue widget value silently. Do not invoke the
        // callback here: execution results may already have updated the panel.
        setSeedWidget(record.node, record.baseRaw, { callCallback: false });
        updateDisplayedPreview(record.node, { force: false });
    }

    app.graph?.setDirtyCanvas?.(true, true);
    app.canvas?.setDirty?.(true, true);
}

function withNativeMinGuard(node, callback, thisArg, args) {
    if (typeof callback !== "function") return undefined;

    const seedWidget = getSeedWidget(node);
    if (!seedWidget) return callback.apply(thisArg, args);

    const options = seedWidget.options || (seedWidget.options = {});
    const originalMin = options.min;

    // Python must expose min=-1 so the sentinel is editable. Native ComfyUI
    // increment/decrement/randomize should nevertheless operate only on
    // ordinary nonnegative seeds; otherwise decrement(0) can manufacture -1.
    options.min = 0;

    try {
        return callback.apply(thisArg, args);
    } finally {
        options.min = originalMin;
    }
}

function applyOwnedRandomSeed(node, context) {
    const randomSeed = generateRandomSeed(node);
    const seedText = String(randomSeed);

    setSeedWidget(node, randomSeed, { callCallback: false });
    setDisplayedSeed(node, seedText, "queue_randomize");

    // Record immediately at the exact pre-serialization point where JLC
    // assigns the seed that will enter the prompt. Do not defer recording to
    // afterQueued; frontend callback timing differs across ComfyUI builds.
    if (context?.recordActive) {
        const sequence = appendRecordedSeed(node, seedText);

        const recording = node.__jlc_seed_record_session;
        if (recording?.active) {
            recording.count += 1;
        }

        setStatus(node, `recording · ${sequence.length} captured`);
    }

    return seedText;
}

function applyReplaySeed(node, context) {
    const sequence = context?.replaySequence ?? [];
    if (!sequence.length) return;

    // Cycling keeps replay deterministic even if the user submits more prompts
    // than were recorded. Switching into Replay resets the persistent cursor.
    const index = context.replayCursor % sequence.length;
    const seedText = sequence[index];
    setSeedWidgetFromText(node, seedText, { callCallback: false });
    setDisplayedSeed(node, seedText, "queue_replay");

    node.__jlc_seed_replay_state = {
        cursor: index + 1,
        total: sequence.length,
    };
    setStatus(node, `replay ${index + 1}/${sequence.length}`);
}

function installControlHooks(node) {
    const control = getControlWidget(node);
    if (!control || control.__jlc_seed_control_wrapped) return;

    control.__jlc_seed_control_wrapped = true;

    const originalBeforeQueued = control.beforeQueued;
    const originalAfterQueued = control.afterQueued;

    control.beforeQueued = function () {
        const context = node.__jlc_seed_queue_context;

        if (context?.baseText === UNSEEDED_TEXT) {
            setSeedWidgetFromText(node, UNSEEDED_TEXT, { callCallback: false });
            setStatus(node, "unseeded sentinel locked");
            return undefined;
        }

        if (context?.replayActive) {
            applyReplaySeed(node, context);
            return undefined;
        }

        if (context?.replayBlocked) {
            setSeedWidget(node, context.baseRaw, { callCallback: false });
            return undefined;
        }

        // JLC owns randomization whenever Randomize is selected. This is
        // necessary because the visible base seed is deliberately restored
        // after every submission; relying on ComfyUI's post-queue mutation
        // would otherwise submit the base seed again on the next single run.
        if (context?.controlMode === "randomize") {
            applyOwnedRandomSeed(node, context);
            return undefined;
        }

        return withNativeMinGuard(node, originalBeforeQueued, this, arguments);
    };

    control.afterQueued = function () {
        const context = node.__jlc_seed_queue_context;

        if (context?.baseText === UNSEEDED_TEXT) {
            setSeedWidgetFromText(node, UNSEEDED_TEXT, { callCallback: false });
            return undefined;
        }

        if (context?.replayActive) {
            context.replayCursor += 1;
            node.__jlc_seed_replay_cursor = context.replayCursor;
            return undefined;
        }

        if (context?.replayBlocked) {
            setSeedWidget(node, context.baseRaw, { callCallback: false });
            return undefined;
        }

        if (context?.controlMode === "randomize") {
            // JLC-owned randomization and optional recording already happened
            // in beforeQueued, before prompt serialization.
            return undefined;
        }

        return withNativeMinGuard(node, originalAfterQueued, this, arguments);
    };
}

function installWidgetWatcher(node, widgetName) {
    const widget = getWidget(node, widgetName);
    if (!widget || widget.__jlc_seed_display_watched) return;

    widget.__jlc_seed_display_watched = true;
    const originalCallback = widget.callback;

    if (widgetName === REPLAY_WIDGET && node.__jlc_seed_last_replay_mode === undefined) {
        node.__jlc_seed_last_replay_mode = readReplayMode(node);
    }

    widget.callback = function () {
        const previousReplayMode = node.__jlc_seed_last_replay_mode;
        const result = originalCallback?.apply(this, arguments);

        if (widgetName === SEED_WIDGET) {
            updateDisplayedPreview(node, { force: true });
        } else {
            const controlMode = readControlMode(node);
            const replayMode = readReplayMode(node);

            if (widgetName === REPLAY_WIDGET && replayMode !== previousReplayMode) {
                if (replayMode === "record") {
                    setStoredReplaySequence(node, []);
                    node.__jlc_seed_replay_cursor = 0;
                    node.__jlc_seed_replay_state = null;
                    setStatus(node, "record ready · sequence cleared");
                } else if (replayMode === "replay") {
                    node.__jlc_seed_replay_cursor = 0;
                    node.__jlc_seed_replay_state = null;
                    const count = getStoredReplaySequence(node).length;
                    setStatus(
                        node,
                        count > 0
                            ? `replay ready · ${count} stored`
                            : "replay requested · no recorded sequence"
                    );
                } else {
                    node.__jlc_seed_replay_cursor = 0;
                    node.__jlc_seed_replay_state = null;
                }
            }

            if (!(widgetName === REPLAY_WIDGET && replayMode !== previousReplayMode)) {
                if (replayMode !== "off" && controlMode !== "randomize") {
                    setStatus(node, `${replayMode} idle · set control to randomize`);
                } else {
                    setStatus(node, controlMode === "randomize" ? replayMode : controlMode);
                }
            }

            node.__jlc_seed_last_replay_mode = replayMode;
            node.setDirtyCanvas?.(true, true);
        }

        return result;
    };
}

function roundedRectPath(ctx, x, y, width, height, radius) {
    const r = Math.max(0, Math.min(radius, width / 2, height / 2));
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

function getPanelState(node) {
    const baseSeed = readSeedWidgetText(node) ?? "?";
    const executedSeed = ensureNodeProperties(node)[DISPLAY_PROPERTY] ?? baseSeed;
    const controlMode = readControlMode(node);
    const replayMode = readReplayMode(node);
    const sequenceLength = getStoredReplaySequence(node).length;
    const status = ensureNodeProperties(node)[STATUS_PROPERTY] ?? "";
    const replayState = node.__jlc_seed_replay_state;

    if (baseSeed === UNSEEDED_TEXT) {
        return {
            accent: COLORS.unseeded,
            badge: "UNSEEDED",
            seedLabel: "seed  -1",
            detail: "sentinel locked · no seed mutation",
        };
    }

    if (status.includes("no recorded sequence")) {
        return {
            accent: COLORS.warning,
            badge: "REPLAY EMPTY",
            seedLabel: `seed  ${executedSeed}`,
            detail: "record a random sequence first",
        };
    }

    const recording = node.__jlc_seed_record_session;
    if (recording?.active) {
        const total = getStoredReplaySequence(node).length;
        const expectedText = recording.expected
            ? ` · queue ${recording.count}/${recording.expected}`
            : "";
        return {
            accent: COLORS.record,
            badge: "RECORD",
            seedLabel: `seed  ${executedSeed}`,
            detail: `randomize · ${total} captured${expectedText}`,
        };
    }

    if (controlMode === "randomize" && replayMode === "replay" && sequenceLength > 0) {
        const position = replayState?.cursor ?? 0;
        return {
            accent: COLORS.replay,
            badge: "REPLAY",
            seedLabel: `seed  ${executedSeed}`,
            detail: `randomize · ${position}/${sequenceLength} · ${sequenceLength} stored`,
        };
    }

    if (controlMode === "randomize" && replayMode === "record") {
        return {
            accent: COLORS.record,
            badge: "RECORD READY",
            seedLabel: `seed  ${executedSeed}`,
            detail: `randomize · ${sequenceLength} stored`,
        };
    }

    const replaySuffix = replayMode !== "off" ? ` · ${replayMode} idle` : "";
    return {
        accent: COLORS.seeded,
        badge: "SEEDED",
        seedLabel: `seed  ${executedSeed}`,
        detail: `${controlMode}${replaySuffix}`,
    };
}

function installDisplayHost(node) {
    const spacer = getWidget(node, SPACER_WIDGET);
    if (!spacer) return;

    if (spacer.__jlc_seed_display_host_installed) return;
    spacer.__jlc_seed_display_host_installed = true;

    // Convert the backend spacer row into a non-editable canvas widget while
    // preserving real LiteGraph layout height.
    spacer.type = "button";
    spacer.value = "";
    spacer.callback = () => true;
    spacer.computeSize = () => [0, PANEL_HEIGHT + PANEL_MARGIN_Y * 2];

    spacer.draw = function (ctx, node, widgetWidth, y, widgetHeight) {
        const state = getPanelState(node);
        const x = PANEL_MARGIN_X;
        const panelY = y + PANEL_MARGIN_Y;
        const panelW = Math.max(80, widgetWidth - PANEL_MARGIN_X * 2);
        const panelH = Math.max(44, Math.min(PANEL_HEIGHT, widgetHeight - PANEL_MARGIN_Y * 2));

        ctx.save();

        roundedRectPath(ctx, x, panelY, panelW, panelH, PANEL_RADIUS);
        ctx.fillStyle = COLORS.panel;
        ctx.fill();
        ctx.strokeStyle = COLORS.panelBorder;
        ctx.lineWidth = 1;
        ctx.stroke();

        // Accent strip.
        roundedRectPath(ctx, x, panelY, 4, panelH, 2);
        ctx.fillStyle = state.accent;
        ctx.fill();

        // Badge.
        ctx.font = "bold 11px sans-serif";
        const badgePaddingX = 7;
        const badgeHeight = 18;
        const badgeWidth = Math.ceil(ctx.measureText(state.badge).width + badgePaddingX * 2);
        const badgeX = x + 12;
        const badgeY = panelY + 8;

        roundedRectPath(ctx, badgeX, badgeY, badgeWidth, badgeHeight, 5);
        ctx.fillStyle = state.accent;
        ctx.fill();

        ctx.fillStyle = COLORS.pillText;
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText(state.badge, badgeX + badgePaddingX, badgeY + badgeHeight / 2 + 0.5);

        // Executed seed, right aligned. Clip so huge decimal values cannot
        // overdraw the badge area.
        ctx.save();
        ctx.beginPath();
        ctx.rect(badgeX + badgeWidth + 8, panelY + 4, panelW - badgeWidth - 32, 26);
        ctx.clip();
        ctx.fillStyle = COLORS.primary;
        ctx.font = "13px monospace";
        ctx.textAlign = "right";
        ctx.textBaseline = "middle";
        ctx.fillText(state.seedLabel, x + panelW - 13, badgeY + badgeHeight / 2 + 0.5);
        ctx.restore();

        // Detail line. Keep it comfortably inside the panel rather than
        // riding the lower clip edge on newer LiteGraph renderers.
        ctx.save();
        const detailTop = badgeY + badgeHeight + 5;
        const detailHeight = Math.max(16, panelY + panelH - detailTop - 7);
        ctx.beginPath();
        ctx.rect(x + 12, detailTop, panelW - 24, detailHeight);
        ctx.clip();
        ctx.fillStyle = COLORS.secondary;
        ctx.font = "11px sans-serif";
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText(state.detail, x + 12, detailTop + detailHeight / 2 - 1);
        ctx.restore();

        ctx.restore();
    };
}

function fitNodeHeight(node) {
    // This node has a fixed vertical widget stack. Older panel revisions could
    // leave the LiteGraph node taller than its current computed content size.
    // Re-fit height only; preserve the user's chosen width.
    requestAnimationFrame(() => {
        const computed = node.computeSize?.();
        if (!computed || !Array.isArray(computed) || computed.length < 2) return;

        const targetHeight = Math.ceil(computed[1]);
        const currentWidth = node.size?.[0];
        const currentHeight = node.size?.[1];

        if (!Number.isFinite(targetHeight) || !Number.isFinite(currentWidth)) return;

        // Avoid churn for sub-pixel / one-pixel differences.
        if (!Number.isFinite(currentHeight) || Math.abs(currentHeight - targetHeight) > 2) {
            node.setSize?.([currentWidth, targetHeight]);
            node.setDirtyCanvas?.(true, true);
        }
    });
}

function installSeedUI(node) {
    sanitizeReplayWidget(node);
    ensureSeedWidgetSentinelRange(node);
    installDisplayHost(node);
    installControlHooks(node);
    installWidgetWatcher(node, SEED_WIDGET);
    installWidgetWatcher(node, REPLAY_WIDGET);
    installWidgetWatcher(node, CONTROL_WIDGET);
    updateDisplayedPreview(node, { force: false });
    fitNodeHeight(node);
}

function patchQueuePrompt() {
    if (app.__jlc_seed_queue_prompt_patched_v7) return;
    if (typeof app.queuePrompt !== "function") return;

    app.__jlc_seed_queue_prompt_patched_v7 = true;

    const originalQueuePrompt = app.queuePrompt;
    app.queuePrompt = async function () {
        const batchCount = arguments[1];
        const records = getSeedNodes()
            .map((node) => beginQueueSession(node, batchCount))
            .filter(Boolean);

        let succeeded = false;

        try {
            const result = await originalQueuePrompt.apply(this, arguments);
            succeeded = true;
            return result;
        } finally {
            for (const record of records) {
                finishQueueSession(record, succeeded);
            }

            // ComfyUI's control widgets mutate target values while queueing.
            // Restore the user's visible base value after submission. Multiple
            // delayed passes make this tolerant of frontend scheduling changes.
            setTimeout(() => restoreSeedBaseValues(records), 0);
            setTimeout(() => restoreSeedBaseValues(records), 100);
            setTimeout(() => restoreSeedBaseValues(records), 500);
        }
    };
}

app.registerExtension({
    name: "JLC.SeedDisplayReplayCosmetic",

    async setup() {
        patchQueuePrompt();
    },

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_NAME) return;

        const originalOnExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const result = originalOnExecuted?.apply(this, arguments);
            updateDisplayedExecution(this, message);
            return result;
        };

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalOnNodeCreated?.apply(this, arguments);
            installSeedUI(this);

            requestAnimationFrame(() => {
                installSeedUI(this);
                this.setDirtyCanvas?.(true, true);
            });

            return result;
        };

        const originalOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = originalOnConfigure?.apply(this, arguments);

            // Sanitize any replay sequence loaded from workflow JSON, including
            // older numeric arrays or hand-edited values.
            setStoredReplaySequence(this, getStoredReplaySequence(this));

            requestAnimationFrame(() => {
                installSeedUI(this);
                this.setDirtyCanvas?.(true, true);
            });

            return result;
        };
    },
});
