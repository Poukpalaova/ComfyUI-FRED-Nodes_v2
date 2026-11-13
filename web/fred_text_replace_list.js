// fred_text_replace_list.js — refresh-safe dynamic UI
// Based on the FRED_WildcardConcat_Dynamic frontend pattern.
// State is kept in the hidden rows_json widget and rebuilt in configure().

import { app } from "/scripts/app.js";

app.registerExtension({
    name: "fred.text_replace_list.dynamic.canvas",

    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== "FRED_TextReplaceList_Dynamic") return;

        const superCreate = nodeType.prototype.onNodeCreated;
        const superConfigure = nodeType.prototype.configure;
        const cache = { files: null };

        async function listFiles(node) {
            if (cache.files) return cache.files;
            try {
                const dir = encodeURIComponent(currentDir(node));
                const response = await fetch(`/fred/text_replace/files?dir=${dir}`);
                const data = await response.json();
                cache.files = Array.isArray(data.files) ? data.files : [];
            } catch {
                cache.files = [];
            }
            return cache.files;
        }

        function currentDir(node) {
            const widget = (node.widgets || []).find(w => w.name === "replacement_dir");
            return typeof widget?.value === "string" ? widget.value.trim() : "";
        }

        function insertAt(node, widget, index) {
            if (!widget) return;
            const oldIndex = node.widgets.indexOf(widget);
            if (oldIndex >= 0) node.widgets.splice(oldIndex, 1);
            node.widgets.splice(Math.min(index, node.widgets.length), 0, widget);
        }

        function orderTopWidgets(node) {
            const find = name => (node.widgets || []).find(w => w?.name === name);
            const path = find("replacement_dir");
            const add = find("➕ Add Replacement");
            const clear = find("🧹 Clear All");
            const toggle = find("fred_tr_header");
            const text = find("text");
            let index = 0;
            insertAt(node, path, index++);
            insertAt(node, add, index++);
            insertAt(node, clear, index++);
            insertAt(node, toggle, index++);
            insertAt(node, text, index++);
        }

        function sync(node) {
            const rowsWidget = (node.widgets || []).find(w => w.name === "rows_json");
            if (!rowsWidget) return;
            rowsWidget.value = JSON.stringify((node._fred_rows || []).map(row => ({
                on: !!row.data.on,
                file: row.data.file || "",
                replacement: row.data.replacement ?? "",
                mode: row.data.mode || "exact",
                case_sensitive: !!row.data.case_sensitive,
            })));
            node.graph?.setDirtyCanvas(true, true);
            node.graph?.change();
        }

        function relabel(node) {
            let number = 1;
            for (const row of node._fred_rows || []) {
                row.widgets.sep.name = `— Replacement ${number} —`;
                row.widgets.onW.name = `On ${number}`;
                row.widgets.fileW.name = `List File ${number}`;
                row.widgets.replW.name = `Replacement ${number}`;
                row.widgets.modeW.name = `Match Mode ${number}`;
                row.widgets.caseW.name = `Case Sensitive ${number}`;
                row.widgets.rmW.name = `❌ Remove ${number}`;
                number++;
            }
        }

        function removeWidget(node, widget) {
            const index = node.widgets.indexOf(widget);
            if (index >= 0) node.widgets.splice(index, 1);
        }

        function clearRows(node) {
            for (const row of node._fred_rows || []) {
                for (const widget of Object.values(row.widgets)) removeWidget(node, widget);
            }
            node._fred_rows = [];
            sync(node);
            node.setSize(node.computeSize());
            node.setDirtyCanvas(true, true);
        }

        async function addRow(node, preset = {}) {
            node._fred_rows ??= [];
            const data = {
                on: preset.on !== false,
                file: String(preset.file || "").trim(),
                replacement: String(preset.replacement ?? ""),
                mode: preset.mode === "whole_word" ? "whole_word" : "exact",
                case_sensitive: preset.case_sensitive === true,
            };

            const index = node._fred_rows.length + 1;
            const sep = node.addWidget("separator", `— Replacement ${index} —`, null, () => {}, { serialize: false });
            const onW = node.addWidget("toggle", `On ${index}`, data.on, value => {
                data.on = !!value;
                sync(node);
            }, { serialize: false });
            const fileW = node.addWidget("combo", `List File ${index}`, data.file, async value => {
                data.file = value || "";
                sync(node);
            }, { values: [], serialize: false });
            const replW = node.addWidget("string", `Replacement ${index}`, data.replacement, value => {
                data.replacement = value ?? "";
                sync(node);
            }, { serialize: false, multiline: false });
            const modeW = node.addWidget("combo", `Match Mode ${index}`, data.mode, value => {
                data.mode = value === "whole_word" ? "whole_word" : "exact";
                sync(node);
            }, { values: ["exact", "whole_word"], serialize: false });
            const caseW = node.addWidget("toggle", `Case Sensitive ${index}`, data.case_sensitive, value => {
                data.case_sensitive = !!value;
                sync(node);
            }, { serialize: false });
            const rmW = node.addWidget("button", `❌ Remove ${index}`, null, () => {
                for (const widget of [sep, onW, fileW, replW, modeW, caseW, rmW]) removeWidget(node, widget);
                node._fred_rows = (node._fred_rows || []).filter(row => row.widgets.rmW !== rmW);
                relabel(node);
                sync(node);
                node.setSize(node.computeSize());
                node.setDirtyCanvas(true, true);
            }, { serialize: false });

            const row = { data, widgets: { sep, onW, fileW, replW, modeW, caseW, rmW } };
            node._fred_rows.push(row);

            const files = await listFiles(node);
            fileW.options.values = files;
            if (!files.includes(data.file)) {
                data.file = files[0] || "";
                fileW.value = data.file;
            }
            relabel(node);
            sync(node);
            node.setSize(node.computeSize());
            node.setDirtyCanvas(true, true);
        }

        function addToggleHeader(node) {
            const header = {
                name: "fred_tr_header",
                type: "custom",
                draw(ctx, n, width, y, height) {
                    const rows = n._fred_rows || [];
                    const allOn = rows.length > 0 && rows.every(row => !!row.data.on);
                    const allOff = rows.length > 0 && rows.every(row => !row.data.on);
                    const mixed = rows.length > 0 && !allOn && !allOff;
                    const x = 10, pillW = 28, pillH = 18, y0 = y + 2;
                    ctx.save();
                    ctx.globalAlpha = app.canvas.editor_alpha;
                    ctx.fillStyle = "#aaa";
                    ctx.beginPath();
                    ctx.roundRect?.(x, y0, pillW, pillH, pillH / 2);
                    ctx.fill();
                    ctx.fillStyle = allOn ? "#76c06b" : mixed ? "#d0a846" : "#888";
                    const knobX = allOn ? x + pillW - 8 : mixed ? x + pillW / 2 : x + 8;
                    ctx.beginPath();
                    ctx.arc(knobX, y0 + pillH / 2, 6, 0, Math.PI * 2);
                    ctx.fill();
                    ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR || "#ddd";
                    ctx.textAlign = "left";
                    ctx.textBaseline = "middle";
                    ctx.fillText("Toggle All", x + pillW + 8, y + height / 2);
                    ctx.restore();
                    this._hit = [x, pillW];
                },
                mouse(event, pos, n) {
                    const type = event?.type || event?.event?.type || "";
                    if (!["pointerdown", "mousedown", "down"].includes(type)) return false;
                    const [x, width] = this._hit || [0, 0];
                    if (pos[0] < x || pos[0] > x + width) return false;
                    const rows = n._fred_rows || [];
                    const allOn = rows.length > 0 && rows.every(row => !!row.data.on);
                    for (const row of rows) row.data.on = !allOn;
                    sync(n);
                    n.setDirtyCanvas(true, true);
                    return true;
                },
                serializeValue() { return {}; },
            };
            return node.addCustomWidget(header);
        }

        nodeType.prototype.onNodeCreated = function () {
            superCreate && superCreate.apply(this, arguments);
            const node = this;
            node._fred_rows = [];
            node._fredAddRow = preset => addRow(node, preset);
            node._fredClearAll = () => clearRows(node);

            const rowsWidget = (node.widgets || []).find(w => w.name === "rows_json");
            if (!rowsWidget) {
                console.warn("[FRED_TextReplaceList_Dynamic] rows_json not found");
                return;
            }
            rowsWidget.hidden = true;
            rowsWidget.computeSize = () => [0, 0];

            let addButton = (node.widgets || []).find(w => w.name === "➕ Add Replacement");
            if (!addButton) addButton = node.addWidget("button", "➕ Add Replacement", null, () => node._fredAddRow(), { serialize: false });
            let clearButton = (node.widgets || []).find(w => w.name === "🧹 Clear All");
            if (!clearButton) clearButton = node.addWidget("button", "🧹 Clear All", null, () => node._fredClearAll(), { serialize: false });
            if (!(node.widgets || []).find(w => w.name === "fred_tr_header")) addToggleHeader(node);
            orderTopWidgets(node);
        };

        nodeType.prototype.configure = function (info) {
            superConfigure && superConfigure.apply(this, arguments);
            const node = this;
            const rowsWidget = (node.widgets || []).find(w => w.name === "rows_json");
            if (!rowsWidget || typeof node._fredAddRow !== "function") return;
            rowsWidget.hidden = true;
            rowsWidget.computeSize = () => [0, 0];

            const savedValue = rowsWidget.value || "[]";
            let saved = [];
            try { saved = JSON.parse(savedValue); } catch { saved = []; }
            clearRows(node);
            (async () => {
                if (Array.isArray(saved) && saved.length) {
                    for (const row of saved) await node._fredAddRow(row);
                } else {
                    await node._fredAddRow();
                }
                orderTopWidgets(node);
                node.setSize(node.computeSize());
                node.setDirtyCanvas(true, true);
            })();
        };
    },
});