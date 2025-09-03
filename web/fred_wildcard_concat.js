// fred_wildcard_concat.js — refresh-safe + exact ordering
// Keeps all existing widgets on reload, rebuilds rows only when absent,
// and reorders the top controls to: path, Add, Clear, Toggle, prefix.
// Also respects "rows_json" as backing state (hidden).

import { app } from "/scripts/app.js";

app.registerExtension({
  name: "fred.wildcards.dynamic.canvas.order_and_refresh_fix",

  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== "FRED_WildcardConcat_Dynamic") return;

    const superCreate    = nodeType.prototype.onNodeCreated;
    const superConfigure = nodeType.prototype.configure;

    // ------- tiny helpers -------
    const RANDOM_VALUE = "random";
    const RANDOM_LABEL = "🎲 random";

    function _insertAt(node, widget, index) {
      if (!widget) return;
      const i = node.widgets.indexOf(widget);
      if (i !== -1) node.widgets.splice(i, 1);
      const safeIndex = Math.max(0, Math.min(index, node.widgets.length));
      node.widgets.splice(safeIndex, 0, widget);
    }

    function placeTopOrdering(node) {
      if (!node.widgets || !node.widgets.length) return;
      const find = (name) => (node.widgets || []).find(w => w && w.name === name);

      const dirW   = find("wildcards_dir");       // 1) path
      const delimW = find("string_delimiter");    // 2)
      const addW   = find("➕ Add Wildcard");     // 3)
      const clrW   = find("🧹 Clear All");        // 4)
      const hdrW   = find("fred_wc_header");      // 5) toggle-all header
      const prefW  = find("prefix");              // 6)

      let i = 0;
      _insertAt(node, dirW,  i++);  // path
      _insertAt(node, delimW,i++);  // delimiter
      _insertAt(node, addW,  i++);  // add
      _insertAt(node, clrW,  i++);  // clear
      _insertAt(node, hdrW,  i++);  // toggle all
      _insertAt(node, prefW, i++);  // prefix
      node.setDirtyCanvas(true, true);
    }

    function hasRowWidgets(node) {
      const ws = node.widgets || [];
      // Head widgets we create below get names "fred_wc_row_#"
      return ws.some(w => typeof w?.name === "string" && w.name.startsWith("fred_wc_row_"));
    }

    // ------- row factory (header + details) -------
    function makeRowFactory(node) {
      // simple cache for listing wildcards
      const cache = { files: null, lines: {} };

      function currentDir() {
        const w = (node.widgets||[]).find(w => w.name === "wildcards_dir");
        return (w && typeof w.value === "string") ? w.value.trim() : "";
      }
      async function listFiles() {
        if (cache.files) return cache.files;
        try {
          const dir = encodeURIComponent(currentDir());
          const r = await fetch(`/fred/wildcards/files?dir=${dir}`);
          const j = await r.json();
          cache.files = Array.isArray(j.files) ? j.files : [];
        } catch { cache.files = []; }
        return cache.files;
      }
      async function listLines(file) {
        if (!file) return [RANDOM_VALUE];
        if (cache.lines[file]) return cache.lines[file];
        try {
          const dir = encodeURIComponent(currentDir());
          const r = await fetch(`/fred/wildcards/lines?file=${encodeURIComponent(file)}&dir=${dir}`);
          const j = await r.json();
          cache.lines[file] = Array.isArray(j.lines) ? j.lines : [RANDOM_VALUE];
        } catch { cache.lines[file] = [RANDOM_VALUE]; }
        return cache.lines[file];
      }

      // expose a hook so changing the path clears caches & refreshes combos
      node._fredRefreshLinesFromPath = async () => {
        cache.files = null; cache.lines = {};
        for (const r of (node._fred_rows||[])) {
          const files = await listFiles();
          if (!files.includes(r.data.file)) r.data.file = files[0] || "";
          const raw  = await listLines(r.data.file);
          const vals = raw.map(x => x===RANDOM_VALUE ? RANDOM_LABEL : x);
          r.widgets.lineW.options.values = vals;
          if (!vals.includes(r.data.line)) {
            r.data.line = RANDOM_LABEL;
            r.widgets.lineW.value = RANDOM_LABEL;
          }
        }
        node._fredSync?.(); node.setDirtyCanvas(true,true);
      };

      // Header custom widget (toggle/file/weight strip)
      function headerWidgetForRow(rowRef) {
        return {
          name: `fred_wc_row`,
          type: "custom",
          draw(ctx, n, width, y, h) {
            const data = rowRef.data;
            const W = n.size?.[0] ?? width;
            const rowH = 20;
            const y0 = y;
            // bg
            ctx.save();
            ctx.globalAlpha = app.canvas.editor_alpha;
            ctx.fillStyle = LiteGraph.WIDGET_BGCOLOR || "#3a3a3a";
            ctx.strokeStyle = LiteGraph.WIDGET_OUTLINE_COLOR || "#555";
            ctx.beginPath();
            // set the new widget rounded rectangle (left margin, y, width, height, radius)
            ctx.roundRect?.(15, y0, W - 30, rowH, 10);
            ctx.fill();
            if (ctx.roundRect) ctx.stroke();
            ctx.restore();

            // toggle pill
            const pillH = 14;
            const pillW = 22;
            const tX = 18, tY = y0 + (rowH - pillH)/2;
            ctx.save();
            ctx.globalAlpha = app.canvas.editor_alpha;
            ctx.fillStyle = data.on ? "#76c06b" : "#888";
            ctx.beginPath(); ctx.roundRect?.(tX, tY, pillW, pillH, pillH/2); ctx.fill();
            ctx.fillStyle = "#fff";
            const knobX = data.on ? (tX + pillW - 7) : (tX + 7);
            ctx.beginPath(); ctx.arc(knobX, tY + pillH/2, 5, 0, Math.PI*2); ctx.fill();
            ctx.restore();

            // file chooser label
            const fileLeft = tX + pillW + 16;
            const fileRight = W - 110;
            const midY = y0 + rowH/2 + 0.5;
            ctx.save();
            ctx.globalAlpha = app.canvas.editor_alpha;
            ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR || "#ddd";
            ctx.textAlign = "left"; ctx.textBaseline = "middle";
            const label = data.file || "Choose file";
            const fit = (txt, maxW) => {
              if (ctx.measureText(txt).width <= maxW) return txt;
              const ell = "…", ellW = ctx.measureText(ell).width;
              let lo=0, hi=txt.length;
              while (lo<=hi) {
                const mid=(lo+hi)>>1;
                const w = ctx.measureText(txt.slice(0, mid)).width;
                if (w <= maxW - ellW) lo = mid + 1; else hi = mid - 1;
              }
              return txt.slice(0, hi) + ell;
            };
            ctx.fillText(fit(label, Math.max(40, fileRight - fileLeft - 12)), fileLeft, midY);
            ctx.textAlign = "right"; ctx.fillText("▾", fileRight, midY);
            ctx.restore();

            // strength ◀ value ▶
            const numW = 70;
            const rightStart = W - 18 - numW;
            ctx.save();
            ctx.globalAlpha = app.canvas.editor_alpha;
            ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR || "#ddd";
            ctx.textAlign = "center"; ctx.textBaseline = "middle";
            const numCenter = rightStart + numW/2;
            ctx.fillText(String((Number(data.weight)||1).toFixed(2)), numCenter, midY);
            ctx.restore();

            // hit map
            this._hits = {
              toggle: [tX, pillW],
              file:   [fileLeft, Math.max(40, fileRight - fileLeft)],
              dec:    [rightStart, 20],
              inc:    [rightStart + numW - 20, 20],
              val:    [rightStart + 20, numW - 40],
            };
          },
          mouse(e, pos, n) {
            const t = e?.type || e?.event?.type || "";
            if (t !== "pointerdown" && t !== "mousedown" && t !== "down") return false;
            const x = pos[0];
            const h = this._hits || {};
            function inside(b){ if (!b) return false; const [bx,bw]=b; return x>=bx && x<=bx+bw; }

            if (inside(h.toggle)) { rowRef.data.on = !rowRef.data.on; n._fredSync(); n.setDirtyCanvas(true,true); return true; }
            if (inside(h.file)) {
              (async ()=>{
                const files = await rowRef.listFiles();
                new LiteGraph.ContextMenu(files.length?files:["(no files)"], {
                  event: e, callback: async (v)=>{
                    if (typeof v === "string" && v!=="(no files)") {
                      rowRef.data.file = v;
                      const raw = await rowRef.listLines(v);
                      const vals = raw.map(x => x===RANDOM_VALUE ? RANDOM_LABEL : x);
                      rowRef.widgets.lineW.options.values = vals;
                      if (!vals.includes(rowRef.data.line)) {
                        rowRef.data.line = RANDOM_LABEL; rowRef.widgets.lineW.value = RANDOM_LABEL;
                      }
                      n._fredSync(); n.setDirtyCanvas(true,true);
                    }
                  }
                });
              })();
              return true;
            }
            if (inside(h.dec)) { rowRef.data.weight = Math.round(((rowRef.data.weight||1)-0.05)*100)/100; n._fredSync(); n.setDirtyCanvas(true,true); return true; }
            if (inside(h.inc)) { rowRef.data.weight = Math.round(((rowRef.data.weight||1)+0.05)*100)/100; n._fredSync(); n.setDirtyCanvas(true,true); return true; }
            if (inside(h.val)) {
              app.canvas.prompt("Strength", (rowRef.data.weight||1).toFixed(2), v=>{
                const num = Number(v); rowRef.data.weight = Number.isFinite(num)? Math.round(num*100)/100:1.00;
                n._fredSync(); n.setDirtyCanvas(true,true);
              }, e);
              return true;
            }
            return false;
          },
          serializeValue(){ return { ...rowRef.data }; }
        };
      }

      return async function addRow(preset) {
        node._fred_rows ??= [];

        const data = {
          on:     preset?.on ?? true,
          file:   (preset?.file||"").trim(),
          line:   preset?.line ? (preset.line===RANDOM_VALUE?RANDOM_LABEL:preset.line) : RANDOM_LABEL,
          weight: Number.isFinite(preset?.weight) ? Math.round(preset.weight*100)/100 : 1.0,
          suffix: typeof preset?.suffix === "string" ? preset.suffix : "",
        };

        // build rowRef with delegates to listFiles/listLines
        const rowRef = { data, widgets: {}, listFiles, listLines };
        const head = node.addCustomWidget( headerWidgetForRow(rowRef) );
        rowRef.head = head;

        const lineW = node.addWidget("combo",  "Line",   data.line, (v)=>{ data.line = v||RANDOM_LABEL; node._fredSync(); }, { values:[RANDOM_LABEL], serialize:false });
        const sfxW  = node.addWidget("string", "Suffix", data.suffix, (v)=>{ data.suffix = v ?? ""; node._fredSync(); }, { serialize:false, multiline:false });
        const rmW   = node.addWidget("button", "❌ Remove", null, ()=>{
          const ih = node.widgets.indexOf(head); if (ih!==-1) node.widgets.splice(ih,1);
          for (const w of [lineW,sfxW,rmW]) { const i=node.widgets.indexOf(w); if (i!==-1) node.widgets.splice(i,1); }
          node._fred_rows = (node._fred_rows||[]).filter(r => r.widgets.rmW !== rmW);
          relabel(); node._fredSync(); return true;
        }, { serialize:false });
        rowRef.widgets = { lineW, sfxW, rmW };

        // initialize combos
        const files = await listFiles();
        if (!files.includes(data.file)) data.file = files[0] || "";
        const raw  = await listLines(data.file);
        const vals = raw.map(x => x===RANDOM_VALUE ? RANDOM_LABEL : x);
        lineW.options.values = vals;
        if (!vals.includes(data.line)) { data.line = RANDOM_LABEL; lineW.value = RANDOM_LABEL; }

        node._fred_rows.push(rowRef);
        relabel(); node._fredSync(); node.graph?.setDirtyCanvas(true,true);

        function relabel() {
          let i=1;
          for (const r of (node._fred_rows||[])) {
            r.head.name = `fred_wc_row_${i}`;
            r.widgets.lineW.name = `Line ${i}`;
            r.widgets.sfxW.name  = `Suffix ${i}`;
            r.widgets.rmW.name   = `❌ Remove ${i}`;
            i++;
          }
        }
      };
    }

    // ------- state sync to rows_json -------
    function installStateSync(node) {
      const rowsJson = (node.widgets||[]).find(w => w.name === "rows_json");
      if (!rowsJson) return;
      rowsJson.hidden = true; rowsJson.computeSize = () => [0,0];

      node._fredSync = () => {
        const payload = (node._fred_rows||[]).map(r => ({
          on: !!r.data.on,
          file: (r.data.file||"").trim(),
          line: (r.data.line===RANDOM_LABEL?RANDOM_VALUE:(r.data.line||"")).trim(),
          weight: Number.isFinite(r.data.weight) ? Math.round(r.data.weight*100)/100 : 1.0,
          suffix: r.data.suffix ?? ""
        }));
        rowsJson.value = JSON.stringify(payload);
        node.graph?.setDirtyCanvas(true,true);
      };
    }

    // ------- toggle-all header -------
    function addToggleAllHeader(node) {
      const header = {
        name: "fred_wc_header",
        type: "custom",
        draw(ctx, n, W, y, h) {
          const rows = n._fred_rows||[];
          const allOn  = rows.length && rows.every(r=>!!r.data.on);
          const allOff = rows.length && rows.every(r=>!r.data.on);
          const tri    = (!rows.length) ? false : (allOn ? true : (allOff ? false : null));

          // pill + label
          const pillH=18, pillW=28, x=10, y0=y + 2;
          ctx.save();
          ctx.globalAlpha = app.canvas.editor_alpha;
          ctx.fillStyle = "#aaa";
          ctx.beginPath(); ctx.roundRect?.(x, y0, pillW, pillH, pillH/2); ctx.fill();
          ctx.fillStyle = (tri===true?"#76c06b":tri===null?"#d0a846":"#888");
          const knobX = (tri===true) ? (x + pillW - 8) : (tri===null ? (x + pillW/2) : (x + 8));
          ctx.beginPath(); ctx.arc(knobX, y0 + pillH/2, 6, 0, Math.PI*2); ctx.fill();
          ctx.restore();

          ctx.save();
          ctx.globalAlpha = app.canvas.editor_alpha;
          ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR || "#ddd";
          ctx.textAlign = "left"; ctx.textBaseline = "middle";
          ctx.fillText("Toggle All", x + pillW + 8, y + h/2 + 0.5);
          ctx.restore();

          this._hit = [x, pillW];
        },
        mouse(e, pos, n) {
          const t = e?.type || e?.event?.type || "";
          if (t !== "pointerdown" && t !== "mousedown" && t !== "down") return false;
          const x = pos[0];
          const [bx,bw] = this._hit || [0,0];
          if (x>=bx && x<=bx+bw) {
            const rows = n._fred_rows||[];
            const allOn  = rows.length && rows.every(r=>!!r.data.on);
            const flip = !(allOn === true);
            for (const r of rows) r.data.on = flip;
            n._fredSync?.(); n.setDirtyCanvas(true,true);
            return true;
          }
          return false;
        },
        serializeValue(){ return {}; }
      };
      return node.addCustomWidget(header);
    }

    // ====== Hook in ======
    nodeType.prototype.onNodeCreated = function () {
      superCreate && superCreate.apply(this, arguments);

      // (1) rows_json sync
      installStateSync(this);

      // (2) row factory (provides this._fredAddRow)
      this._fredAddRow = makeRowFactory(this);

      // (3) top buttons (assumes your Python exposed these buttons/strings already)
      // If they already exist, we simply reuse; otherwise add them.
      let addBtn = (this.widgets||[]).find(w => w.name === "➕ Add Wildcard");
      if (!addBtn) addBtn = this.addWidget("button","➕ Add Wildcard",null,()=>this._fredAddRow(),{serialize:false});

      let clearBtn = (this.widgets||[]).find(w => w.name === "🧹 Clear All");
      if (!clearBtn) clearBtn = this.addWidget("button","🧹 Clear All",null,()=>{
        for (const r of (this._fred_rows||[])) {
          const idxH = this.widgets.indexOf(r.head); if (idxH!==-1) this.widgets.splice(idxH,1);
          for (const w of Object.values(r.widgets)) {
            const i = this.widgets.indexOf(w); if (i!==-1) this.widgets.splice(i,1);
          }
        }
        this._fred_rows = [];
        const rj = (this.widgets||[]).find(w=>w.name==="rows_json"); if (rj) rj.value="[]";
        this.graph?.setDirtyCanvas(true,true);
      },{serialize:false});

      // (4) toggle-all header
      let hdr = (this.widgets||[]).find(w => w.name === "fred_wc_header");
      if (!hdr) hdr = addToggleAllHeader(this);

      // (5) when path changes, refresh combos
      const pathW = (this.widgets||[]).find(w => w.name === "wildcards_dir");
      if (pathW) {
        const orig = pathW.callback;
        pathW.callback = (v)=>{ orig && orig(v); this._fredRefreshLinesFromPath?.(); };
      }

      // Place top ordering once at create
      placeTopOrdering(this);
    };

    nodeType.prototype.configure = function (info) {
      superConfigure && superConfigure.apply(this, arguments);

      // do NOT wipe node.widgets here — that was the bug causing losses

      // re-install sync if missing (defensive)
      installStateSync(this);

      // rebuild rows from rows_json ONLY if there are no visible rows yet
      try {
        if (!hasRowWidgets(this) && typeof this._fredAddRow === "function") {
          const rowsJson = (this.widgets||[]).find(w => w.name === "rows_json");
          const saved = rowsJson ? JSON.parse(rowsJson.value || "[]") : [];
          if (Array.isArray(saved) && saved.length) {
            (async () => {
              for (const r of saved) await this._fredAddRow(r);
              placeTopOrdering(this);
              this.setDirtyCanvas(true, true);
            })();
          }
        }
      } catch {}

      // Re-enforce ordering (path, add, clear, toggle, prefix)
      placeTopOrdering(this);
    };
  },
});



// // FRED Wildcard Concat — canvas UI (no rgthree deps)
// // - Header order: [➕ Add Wildcard] [🧹 Clear All] [Toggle All] [prefix]
// // - Row header:   [On toggle] [File (rounded input)] [Strength (◀ value ▶ pill)]
// // - Row details:  Line (combo), Suffix (string), ❌ Remove (button)
// // - State is persisted in hidden `rows_json`

// // FRED Wildcard Concat — compact canvas UI (Power-Lora-like), no rgthree deps
// // One header per row: [toggle] [file label] [◀ value ▶] inside a single rounded bar
// // Header order: [➕ Add Wildcard] [🧹 Clear All] [Toggle All] [prefix]
// // State persists in hidden `rows_json`

// import { app } from "/scripts/app.js";

// /* ----------------------- VISUAL TUNING (Power-Lora-ish) ---------------------- */
// const MARGIN           = 15;    // outer horizontal padding of the header bar
// const INNER            = 3;     // spacing between inline parts
// const ROW_H            = 20;    // header bar height (Power-Lora ~20)
// const HEADER_Y_OFFSET  = 0;     // move header up/down (negative = up)
// const TOGGLE_H         = 16;    // toggle visual height (<= ROW_H-6)
// const ARROW_W          = 10;    // width of little triangles
// const ARROW_H          = 10;    // height of little triangles
// const NUM_W            = 32;    // width of numeric text
// const RANDOM_VALUE     = "random";
// const RANDOM_LABEL     = "🎲 random";

// /* ------------------------------ tiny draw utils ------------------------------ */
// function isLowQuality() {
  // const ds = app?.canvas?.ds?.scale ?? 1;
  // return ds <= 0.55;
// }
// function roundRectPath(ctx, x, y, w, h, r=4) {
  // if (w < 0) { x += w; w = -w; }
  // if (h < 0) { y += h; h = -h; }
  // const rr = {tl:r,tr:r,br:r,bl:r};
  // ctx.beginPath();
  // ctx.moveTo(x+rr.tl, y);
  // ctx.lineTo(x+w-rr.tr, y);
  // ctx.quadraticCurveTo(x+w, y, x+w, y+rr.tr);
  // ctx.lineTo(x+w, y+h-rr.br);
  // ctx.quadraticCurveTo(x+w, y+h, x+w-rr.br, y+h);
  // ctx.lineTo(x+rr.bl, y+h);
  // ctx.quadraticCurveTo(x, y+h, x, y+h-rr.bl);
  // ctx.lineTo(x, y+rr.tl);
  // ctx.quadraticCurveTo(x, y, x+rr.tl, y);
  // ctx.closePath();
// }
// function fitText(ctx, s, maxW){
  // if (!s) return "";
  // if (ctx.measureText(s).width <= maxW) return s;
  // const ell = "…", ellW = ctx.measureText(ell).width;
  // let lo=0, hi=s.length;
  // while (lo<=hi) {
    // const mid=(lo+hi)>>1;
    // const w = ctx.measureText(s.slice(0, mid)).width;
    // if (w <= maxW - ellW) lo = mid + 1; else hi = mid - 1;
  // }
  // return s.slice(0, hi) + ell;
// }
// // returns [x,width] hit range (X-only; Y is handled by widget row)
// function drawTogglePart(ctx, { posX, posY, height, value }) {
  // const pillW = Math.round(height * 1.6);
  // const knobR = Math.floor((height-8)/2);
  // const y = posY + Math.floor((height - (knobR*2+4))/2);
  // ctx.save();
  // // pill bg (subtle)
  // ctx.globalAlpha = app.canvas.editor_alpha * 0.15;
  // ctx.fillStyle = "#aaa";
  // roundRectPath(ctx, posX, y, pillW, knobR*2+4, knobR+2);
  // ctx.fill();
  // // knob
  // const on = value === true;
  // const mix = value === null;
  // ctx.globalAlpha = app.canvas.editor_alpha * 0.95;
  // ctx.fillStyle = on ? "#76c06b" : (mix ? "#d0a846" : "#888");
  // const knobX = on ? (posX + pillW - (knobR+2)) : (mix ? (posX + pillW/2) : (posX + (knobR+2)));
  // ctx.beginPath();
  // ctx.arc(knobX, posY + height/2, knobR, 0, Math.PI*2);
  // ctx.fill();
  // ctx.restore();
  // return [posX, pillW];
// }
// function drawTinyTriangle(ctx, cx, cy, dir /* -1 left, +1 right */) {
  // const hw = ARROW_W, hh = ARROW_H / 2;
  // ctx.beginPath();
  // if (dir < 0) { // ◀
    // ctx.moveTo(cx + hw/2, cy - hh);
    // ctx.lineTo(cx - hw/2, cy);
    // ctx.lineTo(cx + hw/2, cy + hh);
  // } else {      // ▶
    // ctx.moveTo(cx - hw/2, cy - hh);
    // ctx.lineTo(cx + hw/2, cy);
    // ctx.lineTo(cx - hw/2, cy + hh);
  // }
  // ctx.closePath();
  // ctx.fill();
// }
// // inline ◀ value ▶ on the header (no extra boxes). Returns 3 hit boxes.
// function drawNumberInlinePart(ctx, { posX, posY, height, value }) {
  // const midY = posY + height / 2 + 0.5;
  // ctx.save();
  // ctx.globalAlpha = app.canvas.editor_alpha;
  // ctx.fillStyle   = LiteGraph.WIDGET_TEXT_COLOR || "#ddd";
  // ctx.textAlign   = "center";
  // ctx.textBaseline= "middle";
  // // left ◀
  // drawTinyTriangle(ctx, posX + ARROW_W/2, midY, -1);
  // const left = [posX, ARROW_W];
  // // value
  // const numX = posX + ARROW_W + INNER;
  // ctx.fillText((Number(value)||0).toFixed(2), numX + NUM_W/2, midY);
  // const txt  = [numX, NUM_W];
  // // right ▶
  // const rightX = numX + NUM_W + INNER;
  // drawTinyTriangle(ctx, rightX + ARROW_W/2, midY, +1);
  // const right = [rightX, ARROW_W];
  // ctx.restore();
  // return [left, txt, right];
// }

// /* ------------------------ minimal base canvas widget ------------------------ */
// function baseCanvasWidget(name, impl) {
  // const w = {
    // name, type: "custom", last_y: 0, hitAreas: {}, _downKey: null,
    // draw(ctx, node, width, posY, height){ impl.draw?.call(this, ctx, node, width, posY, height); },
    // mouse(e, pos, node){
      // const t = e?.type || e?.event?.type || "";
      // const isDown = (t === "pointerdown" || t === "mousedown" || t === "down");
      // const isUp   = (t === "pointerup"   || t === "mouseup"   || t === "up");
      // const isMove = (t === "pointermove" || t === "mousemove" || t === "move");
      // const x = pos[0]; // local X
      // const pickKey = () => {
        // for (const k in this.hitAreas) {
          // const b = this.hitAreas[k]?.bounds; if (!b) continue;
          // const [bx,bw] = b;
          // if (x >= bx && x <= bx + bw) return k;
        // }
        // return null;
      // };
      // if (isDown) {
        // const k = pickKey();
        // if (k) { this._downKey = k; return !!this.hitAreas[k]?.onDown?.(e,pos,node) || true; }
      // } else if (isMove && this._downKey) {
        // this.hitAreas[this._downKey]?.onMove?.(e,pos,node); return true;
      // } else if (isUp && this._downKey) {
        // this.hitAreas[this._downKey]?.onUp?.(e,pos,node); this._downKey = null; return true;
      // }
      // return false;
    // },
    // serializeValue(node, index){ return impl.serializeValue?.call(this, node, index) ?? this.value; },
  // };
  // return w;
// }

// /* --------------------------------- extension -------------------------------- */
// app.registerExtension({
  // name: "fred.wildcards.dynamic.canvas",

  // beforeRegisterNodeDef(nodeType, nodeData) {
    // if (nodeData?.name !== "FRED_WildcardConcat_Dynamic") return;

    // const superCreate    = nodeType.prototype.onNodeCreated;
    // const superConfigure = nodeType.prototype.configure;

    // nodeType.prototype.onNodeCreated = function () {
      // superCreate && superCreate.apply(this, arguments);
      // const node = this;

      // const rowsJson = (node.widgets||[]).find(w => w.name === "rows_json");
      // const prefixW  = (node.widgets||[]).find(w => w.name === "prefix");
      // if (!rowsJson) { console.warn("[FRED] rows_json not found"); return; }

      // // hide rows_json
      // rowsJson.hidden = true; rowsJson.computeSize = () => [0,0];

      // // ---- tiny API
      // const cache = { files: null, lines:{} };
      // async function listFiles() {
        // if (cache.files) return cache.files;
        // try {
          // const r = await fetch("/fred/wildcards/files");
          // const j = await r.json();
          // cache.files = Array.isArray(j.files) ? j.files : [];
        // } catch { cache.files = []; }
        // return cache.files;
      // }
      // async function listLines(file) {
        // if (!file) return [RANDOM_VALUE];
        // if (cache.lines[file]) return cache.lines[file];
        // try {
          // const r = await fetch(`/fred/wildcards/lines?file=${encodeURIComponent(file)}`);
          // const j = await r.json();
          // cache.lines[file] = Array.isArray(j.lines) ? j.lines : [RANDOM_VALUE];
        // } catch { cache.lines[file] = [RANDOM_VALUE]; }
        // return cache.lines[file];
      // }

      // // ---- persistence
      // function syncRowsJson() {
        // const payload = (node._fred_rows||[]).map(r => ({
          // on: !!r.data.on,
          // file: (r.data.file||"").trim(),
          // line: (r.data.line===RANDOM_LABEL?RANDOM_VALUE:(r.data.line||"")).trim(),
          // weight: Number.isFinite(r.data.weight) ? Math.round(r.data.weight*100)/100 : 1.0,
          // suffix: r.data.suffix ?? ""
        // }));
        // rowsJson.value = JSON.stringify(payload);
        // node.graph?.setDirtyCanvas(true,true);
      // }
      // node._fredSync = syncRowsJson;

      // function allRowsState() {
        // const rows = node._fred_rows||[];
        // if (!rows.length) return false;
        // const allOn  = rows.every(r=>!!r.data.on);
        // const allOff = rows.every(r=>!r.data.on);
        // return allOn ? true : (allOff ? false : null);
      // }
      // function relabel() {
        // let i=1;
        // for (const r of (node._fred_rows||[])) {
          // r.head.name = `fred_wc_row_${i}`;
          // r.widgets.lineW.name = `Line ${i}`;
          // r.widgets.sfxW.name  = `Suffix ${i}`;
          // r.widgets.rmW.name   = `❌ Remove ${i}`;
          // i++;
        // }
      // }
      // function purgeLegacy() {
        // const legacy = (node.widgets||[]).filter(w =>
          // /^(On|File|Strength) \d+$/.test(w.name||"") || (w.name||"").startsWith("— Row "));
        // for (const w of legacy) {
          // const i = node.widgets.indexOf(w);
          // if (i!==-1) node.widgets.splice(i,1);
        // }
      // }

      // // ---- header: Toggle All (insert AFTER Clear All)
      // const header = baseCanvasWidget("fred_wc_header", {
        // draw(ctx, n, w, y, h) {
          // this.hitAreas = {};
          // const margin = 10, inner = 8;
          // const toggleRange = drawTogglePart(ctx, { posX: margin, posY: y, height: 20, value: allRowsState() });
          // this.hitAreas.toggle = {
            // bounds: [toggleRange[0], toggleRange[1]],
            // onDown: () => {
              // const st = allRowsState();
              // const flip = !(st === true);
              // for (const r of (n._fred_rows||[])) r.data.on = flip;
              // n._fredSync(); n.setDirtyCanvas(true,true);
              // return true;
            // }
          // };
          // // label
          // ctx.save();
          // ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
          // ctx.textAlign="left"; ctx.textBaseline="middle";
          // ctx.fillText("Toggle All", margin + toggleRange[1] + inner, y + h/2);
          // ctx.restore();
        // },
        // serializeValue(){ return {}; }
      // });

      // // ---- add one row
      // node._fredAddRow = async (preset)=>{
        // node._fred_rows ??= [];
        // const data = {
          // on:    preset?.on ?? true,
          // file:  (preset?.file||"").trim(),
          // line:  preset?.line ? (preset.line===RANDOM_VALUE?RANDOM_LABEL:preset.line) : RANDOM_LABEL,
          // weight: Number.isFinite(preset?.weight) ? Math.round(preset.weight*100)/100 : 1.0,
          // suffix: typeof preset?.suffix === "string" ? preset.suffix : "",
        // };

        // const head = baseCanvasWidget("fred_wc_row", {
          // draw(ctx, n, w, y, h) {
            // this.hitAreas = {};
            // const width = n.size?.[0] ?? w;

            // // geometry
            // const rowH = ROW_H;
            // const y0   = y + HEADER_Y_OFFSET;
            // const midY = y0 + rowH / 2;

            // // single rounded header bar
            // ctx.save();
            // ctx.globalAlpha = app.canvas.editor_alpha;
            // ctx.fillStyle   = LiteGraph.WIDGET_BGCOLOR   || "#3a3a3a";
            // ctx.strokeStyle = LiteGraph.WIDGET_OUTLINE_COLOR || "#555";
            // roundRectPath(ctx, MARGIN, y0, width - MARGIN*2, rowH, Math.round(rowH/1.9));
            // ctx.fill();
            // if (!isLowQuality()) ctx.stroke();
            // ctx.restore();

            // // left: toggle
            // const tX   = MARGIN + 8;
            // const tH   = Math.min(TOGGLE_H, rowH - 6);
            // const tBox = drawTogglePart(ctx, { posX: tX, posY: y0 + (rowH - tH)/2, height: tH, value: !!data.on });
            // this.hitAreas.toggle = {
              // bounds: [tBox[0], tBox[1]],
              // onDown: ()=>{ data.on=!data.on; n._fredSync(); n.setDirtyCanvas(true,true); return true; }
            // };

            // // right: inline number ◀ value ▶
            // const numTotalW = ARROW_W + INNER + NUM_W + INNER + ARROW_W;
            // const numX = width - MARGIN - 6 - numTotalW; // 8px inner right padding
            // const [less, txt, more] = drawNumberInlinePart(ctx, { posX: numX, posY: y0, height: rowH, value: data.weight });
            // this.hitAreas.dec = { bounds: less, onDown:()=>{ data.weight = Math.round(((data.weight||1)-0.05)*100)/100; n._fredSync(); n.setDirtyCanvas(true,true); return true; } };
            // this.hitAreas.inc = { bounds: more, onDown:()=>{ data.weight = Math.round(((data.weight||1)+0.05)*100)/100; n._fredSync(); n.setDirtyCanvas(true,true); return true; } };
            // this.hitAreas.val = { bounds: txt,  onDown:(ev)=> {
              // app.canvas.prompt("Strength", (data.weight||1).toFixed(2), v=>{
                // const num = Number(v); data.weight = Number.isFinite(num)? Math.round(num*100)/100:1.00;
                // n._fredSync(); n.setDirtyCanvas(true,true);
              // }, ev);
              // return true;
            // }};

            // // middle: file label (click to choose)
            // const fileLeft  = tX + tBox[1] + INNER + 24;
            // const fileRight = numX - INNER - 16;
            // const fileW     = Math.max(60, fileRight - fileLeft);

            // ctx.save();
            // ctx.globalAlpha = app.canvas.editor_alpha;
            // ctx.fillStyle   = LiteGraph.WIDGET_TEXT_COLOR || "#ddd";
            // ctx.textAlign   = "left"; ctx.textBaseline = "middle";
            // ctx.fillText(fitText(ctx, data.file || "Choose file", fileW - 12), fileLeft, midY + 0.5);
            // ctx.textAlign = "right";
            // ctx.fillText("▾", fileLeft + fileW - 4, midY + 0.5);
            // ctx.restore();

            // this.hitAreas.file = {
              // bounds: [fileLeft, fileW],
              // onDown: (ev)=> {
                // listFiles().then(files=>{
                  // new LiteGraph.ContextMenu(files.length?files:["(no files)"], {
                    // event: ev, callback: (v)=>{
                      // if (typeof v === "string" && v!=="(no files)") {
                        // data.file = v;
                        // listLines(data.file).then(lines=>{
                          // const vals = lines.map(x => x===RANDOM_VALUE ? RANDOM_LABEL : x);
                          // const lineW = (n._fred_rows||[]).find(r => r.head === rowRef)?.widgets?.lineW;
                          // if (lineW) {
                            // lineW.options.values = vals;
                            // if (!vals.includes(data.line)) { data.line = RANDOM_LABEL; lineW.value = RANDOM_LABEL; }
                          // }
                          // n._fredSync(); n.setDirtyCanvas(true,true);
                        // });
                      // }
                    // }
                  // });
                // });
                // return true;
              // }
            // };
          // },
          // serializeValue(){ return { ...data }; }
        // });
        // const rowRef = head; // used in draw() to update the right Line widget
        // const headWidget = node.addCustomWidget(head);

        // // details as standard widgets
        // const lineW = node.addWidget("combo",  "Line",   data.line, (v)=>{ data.line = v||RANDOM_LABEL; node._fredSync(); }, { values:[RANDOM_LABEL], serialize:false });
        // const sfxW  = node.addWidget("string", "Suffix", data.suffix, (v)=>{ data.suffix = v ?? ""; node._fredSync(); }, { serialize:false, multiline:false });
        // const rmW   = node.addWidget("button", "❌ Remove", null, ()=>{
          // const ih = node.widgets.indexOf(headWidget); if (ih!==-1) node.widgets.splice(ih,1);
          // for (const w of [lineW,sfxW,rmW]) { const i=node.widgets.indexOf(w); if (i!==-1) node.widgets.splice(i,1); }
          // node._fred_rows = (node._fred_rows||[]).filter(r => r.widgets.rmW !== rmW);
          // relabel(); node._fredSync(); return true;
        // }, { serialize:false });

        // // init options
        // const files = await listFiles();
        // if (!files.includes(data.file)) data.file = files[0] || "";
        // const raw  = await listLines(data.file);
        // const vals = raw.map(x => x===RANDOM_VALUE ? RANDOM_LABEL : x);
        // lineW.options.values = vals;
        // if (!vals.includes(data.line)) { data.line = RANDOM_LABEL; lineW.value = RANDOM_LABEL; }

        // node._fred_rows.push({ data, head: headWidget, widgets:{ lineW, sfxW, rmW } });
        // relabel(); node._fredSync(); node.graph?.setDirtyCanvas(true,true);
      // };

      // // top controls
      // const addBtn   = node.addWidget("button","➕ Add Wildcard",null,()=>node._fredAddRow(),{serialize:false});
      // const clearBtn = node.addWidget("button","🧹 Clear All",   null,()=>{
        // for (const r of (node._fred_rows||[])) {
          // const iH = node.widgets.indexOf(r.head); if (iH!==-1) node.widgets.splice(iH,1);
          // for (const w of Object.values(r.widgets)) { const i=node.widgets.indexOf(w); if (i!==-1) node.widgets.splice(i,1); }
        // }
        // node._fred_rows = [];
        // const rj = (node.widgets||[]).find(w=>w.name==="rows_json"); if (rj) rj.value="[]";
        // node.graph?.setDirtyCanvas(true,true);
      // },{serialize:false});

      // // create & insert header RIGHT AFTER Clear All
      // const headerWidget = node.addCustomWidget(header);
      // {
        // const iHdr  = node.widgets.indexOf(headerWidget);
        // const iClr  = node.widgets.indexOf(clearBtn);
        // if (iHdr > -1 && iClr > -1 && iHdr !== iClr + 1) {
          // node.widgets.splice(iHdr,1);
          // node.widgets.splice(iClr+1,0,headerWidget);
        // }
      // }

      // // Prefix AFTER header
      // if (prefixW) {
        // const iPref = node.widgets.indexOf(prefixW);
        // const iHdr  = node.widgets.indexOf(headerWidget);
        // if (iPref !== -1 && iHdr !== -1) {
          // node.widgets.splice(iPref, 1);
          // node.widgets.splice(iHdr + 1, 0, prefixW);
        // }
      // }

      // purgeLegacy();
    // };

    // nodeType.prototype.configure = function(info){
      // superConfigure && superConfigure.apply(this, arguments);
      // const node = this;
      // // purge legacy again (in case super re-added)
      // (function purge(){
        // const legacy = (node.widgets||[]).filter(w =>
          // /^(On|File|Strength) \d+$/.test(w.name||"") || (w.name||"").startsWith("— Row "));
        // for (const w of legacy) {
          // const i=node.widgets.indexOf(w); if (i!==-1) node.widgets.splice(i,1);
        // }
      // })();

      // const rowsJson = (node.widgets||[]).find(w => w.name === "rows_json");
      // if (!rowsJson) return;
      // let saved=[]; try{ saved = JSON.parse(rowsJson.value||"[]"); } catch {}

      // // keep only top controls, header, prefix, rows_json
      // const keep = new Set(["wildcards_dir","➕ Add Wildcard","🧹 Clear All","fred_wc_header","prefix","rows_json"]);
      // node.widgets = (node.widgets||[]).filter(w => keep.has(w.name));
      // node._fred_rows = [];

      // (async ()=>{
        // if (Array.isArray(saved) && saved.length) {
          // for (const r of saved) await node._fredAddRow(r);
        // } else {
          // await node._fredAddRow();
        // }
        // node.graph?.setDirtyCanvas(true,true);
      // })();
    // };
  // },
// });




// // FRED_WildcardConcat_Dynamic UI (stable with optional compact header)
// // - Persists via hidden rows_json
// // - Rehydrates in configure(info)
// // - Header buttons: ➕ Add • 🧹 Clear All • Toggle All
// // - Per-row Remove uses a red ❌ emoji (canvas text can't be per-widget colored)
// // - Optional one-line header (On + File + Strength) using a guarded DOM widget:
// //     set COMPACT_HEADER = true to try it; falls back to stable if DOM fails.

// import { app } from "/scripts/app.js";

// const RANDOM_VALUE = "random";
// const RANDOM_LABEL = "🎲 random";
// const COMPACT_HEADER = false; // <-- set to true to try the one-line header

// app.registerExtension({
  // name: "fred.wildcards.dynamic",

  // beforeRegisterNodeDef(nodeType, nodeData) {
    // if (nodeData?.name !== "FRED_WildcardConcat_Dynamic") return;

    // nodeType.prototype.serialize_widgets = true;

    // const superCreate = nodeType.prototype.onNodeCreated;
    // const superConfigure = nodeType.prototype.configure;

    // nodeType.prototype.onNodeCreated = function () {
      // if (superCreate) superCreate.apply(this, arguments);
      // const node = this;

      // const rowsJson = (node.widgets || []).find(w => w.name === "rows_json");
      // const prefixW  = (node.widgets || []).find(w => w.name === "prefix");
      // if (!rowsJson) { console.warn("[FRED] rows_json not found."); return; }

      // // hide the storage widget
      // rowsJson.hidden = true;
      // rowsJson.computeSize = () => [0, 0];

      // // simple backend caches
      // const cache = { files: null, lines: {} };
      // async function listFiles() {
        // if (cache.files) return cache.files;
        // const r = await fetch("/fred/wildcards/files");
        // const j = await r.json();
        // cache.files = Array.isArray(j.files) ? j.files : [];
        // return cache.files;
      // }
      // async function listLines(file) {
        // if (!file) return [RANDOM_VALUE];
        // if (cache.lines[file]) return cache.lines[file];
        // const r = await fetch(`/fred/wildcards/lines?file=${encodeURIComponent(file)}`);
        // const j = await r.json();
        // const arr = Array.isArray(j.lines) ? j.lines : [RANDOM_VALUE];
        // cache.lines[file] = arr;
        // return arr;
      // }

      // function syncRowsJson() {
        // const payload = (node._fred_rows || []).map(r => ({
          // on: !!r.data.on,
          // file: (r.data.file || "").trim(),
          // line: (r.data.line === RANDOM_LABEL ? RANDOM_VALUE : (r.data.line || "")).trim(),
          // weight: Number.isFinite(r.data.weight) ? Math.round(r.data.weight * 100) / 100 : 1.0,
          // suffix: r.data.suffix ?? ""
        // }));
        // rowsJson.value = JSON.stringify(payload);
        // app.graph.setDirtyCanvas(true, true);
      // }
      // node._fredSync = syncRowsJson;

      // function relabel() {
        // let n = 1;
        // for (const r of (node._fred_rows || [])) {
          // r.widgets.sep.name     = `— Row ${n} —`;
          // r.widgets.onW?.name    && (r.widgets.onW.name     = `On ${n}`);
          // r.widgets.fileW?.name  && (r.widgets.fileW.name   = `File ${n}`);
          // r.widgets.lineW.name   = `Line ${n}`;
          // r.widgets.weightW?.name&& (r.widgets.weightW.name = `Strength ${n}`);
          // r.widgets.sfxW.name    = `Suffix ${n}`;
          // r.widgets.rmW.name     = `❌ Remove ${n}`; // red icon
          // n++;
        // }
      // }

      // // remove all UI rows cleanly
      // node._fredClearAll = () => {
        // if (!node._fred_rows) node._fred_rows = [];
        // for (const r of node._fred_rows) {
          // // remove optional DOM header
          // if (r.domW) {
            // const iDom = node.widgets.indexOf(r.domW);
            // if (iDom !== -1) node.widgets.splice(iDom, 1);
          // }
          // // remove standard widgets
          // const ws = r.widgets;
          // [ws.sep, ws.onW, ws.fileW, ws.lineW, ws.weightW, ws.sfxW, ws.rmW]
            // .filter(Boolean)
            // .forEach(w => {
              // const i = node.widgets.indexOf(w);
              // if (i !== -1) node.widgets.splice(i, 1);
            // });
        // }
        // node._fred_rows = [];
        // rowsJson.value = "[]";
        // app.graph.setDirtyCanvas(true, true);
      // };

      // // ----------- two ways to add a row -------------
      // const addRowStandard = async (preset) => {
        // node._fred_rows ??= [];
        // const data = {
          // on:    preset?.on ?? true,
          // file:  (preset?.file || "").trim(),
          // line:  preset?.line ? (preset.line === RANDOM_VALUE ? RANDOM_LABEL : preset.line) : RANDOM_LABEL,
          // weight: Number.isFinite(preset?.weight) ? Math.round(preset.weight * 100) / 100 : 1.0,
          // suffix: (typeof preset?.suffix === "string") ? preset.suffix : "",
        // };

        // const idx = (node._fred_rows?.length || 0) + 1;

        // const sep = node.addWidget("separator", `— Row ${idx} —`, null, ()=>{}, { serialize:false });

        // const onW = node.addWidget("toggle", `On ${idx}`, data.on, (v)=>{
          // data.on = !!v; node._fredSync();
        // }, { serialize:false });

        // const fileW = node.addWidget("combo", `File ${idx}`, data.file, async (v)=>{
          // data.file = v || "";
          // const rawLines = await listLines(data.file);
          // const display  = rawLines.map(x => x === RANDOM_VALUE ? RANDOM_LABEL : x);
          // lineW.options.values = display;
          // if (!display.includes(data.line)) {
            // data.line = RANDOM_LABEL;
            // lineW.value = RANDOM_LABEL;
          // }
          // node._fredSync();
        // }, { values: [], serialize:false });

        // const lineW = node.addWidget("combo", `Line ${idx}`, data.line, (v)=>{
          // data.line = v || RANDOM_LABEL;
          // node._fredSync();
        // }, { values: [RANDOM_LABEL], serialize:false });

        // const weightW = node.addWidget("number", `Strength ${idx}`, data.weight, (v)=>{
          // const n = Number.isFinite(v) ? Math.round(v * 100) / 100 : 1.0;
          // data.weight = n; weightW.value = n;
          // node._fredSync();
        // }, { min:-10, max:10, step:0.01, precision: 2, serialize:false });

        // const sfxW = node.addWidget("string", `Suffix ${idx}`, data.suffix, (v)=>{
          // data.suffix = (v ?? "");
          // node._fredSync();
        // }, { serialize:false, multiline:false });

        // const rmW = node.addWidget("button", `❌ Remove ${idx}`, null, ()=>{
          // data.on = false;
          // [sep, onW, fileW, lineW, weightW, sfxW, rmW].forEach(w=>{
            // const i = node.widgets.indexOf(w);
            // if (i !== -1) node.widgets.splice(i,1);
          // });
          // node._fred_rows = (node._fred_rows || []).filter(r => r.widgets.rmW !== rmW);
          // relabel();
          // node._fredSync();
          // return true;
        // }, { serialize:false });

        // // populate options
        // const files = await listFiles();
        // fileW.options.values = files;
        // if (!files.includes(data.file)) {
          // data.file = files[0] || "";
          // fileW.value = data.file;
        // }
        // const rawLines = await listLines(data.file);
        // const display  = rawLines.map(x => x === RANDOM_VALUE ? RANDOM_LABEL : x);
        // lineW.options.values = display;
        // if (!display.includes(data.line)) {
          // data.line = RANDOM_LABEL;
          // lineW.value = RANDOM_LABEL;
        // }

        // node._fred_rows.push({ data, widgets: { sep, onW, fileW, lineW, weightW, sfxW, rmW } });
        // relabel();
        // node._fredSync();
        // node.graph.setDirtyCanvas(true, true);
      // };

      // const addRowCompact = async (preset) => {
        // // Guarded DOM header (falls back if anything fails)
        // try {
          // node._fred_rows ??= [];
          // const data = {
            // on:    preset?.on ?? true,
            // file:  (preset?.file || "").trim(),
            // line:  preset?.line ? (preset.line === RANDOM_VALUE ? RANDOM_LABEL : preset.line) : RANDOM_LABEL,
            // weight: Number.isFinite(preset?.weight) ? Math.round(preset.weight * 100) / 100 : 1.0,
            // suffix: (typeof preset?.suffix === "string") ? preset.suffix : "",
          // };

          // const sep = node.addWidget("separator", "— Row —", null, ()=>{}, { serialize:false });

          // // DOM: On + File + Strength in one line
          // const dom = document.createElement("div");
          // dom.style.display = "flex";
          // dom.style.alignItems = "center";
          // dom.style.gap = "8px";
          // dom.style.width = "100%";

          // const onBox = document.createElement("input");
          // onBox.type = "checkbox";
          // onBox.checked = !!data.on;
          // dom.append(onBox);

          // const fileSel = document.createElement("select");
          // fileSel.style.flex = "1 1 auto";
          // dom.append(fileSel);

          // const wInput = document.createElement("input");
          // wInput.type = "number";
          // wInput.step = "0.01";
          // wInput.value = String(data.weight);
          // wInput.title = "Strength";
          // wInput.style.width = "96px";
          // dom.append(wInput);

          // onBox.addEventListener("input", () => { data.on = onBox.checked; node._fredSync(); });
          // wInput.addEventListener("input", () => {
            // const n = Number(wInput.value);
            // data.weight = Number.isFinite(n) ? Math.round(n * 100) / 100 : 1.0;
            // wInput.value = String(data.weight);
            // node._fredSync();
          // });
          // fileSel.addEventListener("change", async () => {
            // data.file = fileSel.value || "";
            // const rawLines = await listLines(data.file);
            // const display  = rawLines.map(x => x === RANDOM_VALUE ? RANDOM_LABEL : x);
            // lineW.options.values = display;
            // if (!display.includes(data.line)) {
              // data.line = RANDOM_LABEL;
              // lineW.value = RANDOM_LABEL;
            // }
            // node._fredSync();
          // });

          // const domW = node.addDOMWidget(`fred_row_hdr_${Date.now()}_${Math.random().toString(36).slice(2)}`, dom, {
            // serialize: false,
            // computeSize: () => [node.size?.[0] ?? 300, 28],
          // });

          // // second line widgets
          // const lineW = node.addWidget("combo", "Line", data.line, (v)=>{
            // data.line = v || RANDOM_LABEL;
            // node._fredSync();
          // }, { values: [RANDOM_LABEL], serialize:false });

          // const sfxW  = node.addWidget("string", "Suffix", data.suffix, (v)=>{
            // data.suffix = (v ?? "");
            // node._fredSync();
          // }, { serialize:false, multiline:false });

          // const rmW   = node.addWidget("button", "❌ Remove", null, ()=>{
            // const iDom = node.widgets.indexOf(domW);
            // if (iDom !== -1) node.widgets.splice(iDom,1);
            // [sep, lineW, sfxW, rmW].forEach(w=>{
              // const i = node.widgets.indexOf(w);
              // if (i !== -1) node.widgets.splice(i,1);
            // });
            // node._fred_rows = (node._fred_rows || []).filter(r => r.widgets.rmW !== rmW);
            // relabel();
            // node._fredSync();
            // return true;
          // }, { serialize:false });

          // // populate file + lines
          // const files = await listFiles();
          // fileSel.replaceChildren(...files.map(f => {
            // const o = document.createElement("option");
            // o.value = f; o.textContent = f; return o;
          // }));
          // if (!files.includes(data.file)) data.file = files[0] || "";
          // fileSel.value = data.file;

          // const rawLines = await listLines(data.file);
          // const display  = rawLines.map(x => x === RANDOM_VALUE ? RANDOM_LABEL : x);
          // lineW.options.values = display;
          // if (!display.includes(data.line)) {
            // data.line = RANDOM_LABEL;
            // lineW.value = RANDOM_LABEL;
          // }

          // node._fred_rows.push({
            // data,
            // domW,
            // dom: { on: onBox, file: fileSel, weight: wInput },
            // widgets: { sep, lineW, sfxW, rmW }
          // });
          // relabel();
          // node._fredSync();
          // node.graph.setDirtyCanvas(true, true);
        // } catch (e) {
          // console.warn("[FRED] Compact header failed, falling back:", e);
          // await addRowStandard(preset);
        // }
      // };
      // // ------------------------------------------------

      // // expose entry point selecting mode
      // node._fredAddRow = (preset) => COMPACT_HEADER ? addRowCompact(preset) : addRowStandard(preset);

      // // Header controls
      // const addBtn   = node.addWidget("button", "➕ Add Wildcard", null, ()=> node._fredAddRow(), { serialize:false });
      // const clearBtn = node.addWidget("button", "🧹 Clear All", null, ()=> node._fredClearAll(), { serialize:false });
      // const togBtn   = node.addWidget("button", "Toggle All", null, ()=>{
        // const rows = node._fred_rows || [];
        // const allOn = rows.length > 0 && rows.every(r => !!r.data.on);
        // const newVal = !allOn; // invert all
        // for (const r of rows) {
          // r.data.on = newVal;
          // // reflect on either kind of header
          // if (r.widgets?.onW) r.widgets.onW.value = newVal;   // standard
          // if (r.dom?.on)      r.dom.on.checked   = newVal;    // compact
        // }
        // node._fredSync();
        // return true;
      // }, { serialize:false });

      // // Move prefix AFTER Toggle All
      // if (prefixW) {
        // const iPref = node.widgets.indexOf(prefixW);
        // const iTog  = node.widgets.indexOf(togBtn);
        // if (iPref > -1 && iTog > -1) {
          // node.widgets.splice(iPref, 1);
          // node.widgets.splice(iTog + 1, 0, prefixW);
        // }
      // }

      // // Initial build is handled in configure()
    // };

    // // Rehydrate after saved widgets_values are applied
    // nodeType.prototype.configure = function (info) {
      // if (superConfigure) superConfigure.apply(this, arguments);
      // const node = this;
      // const rowsJson = (node.widgets || []).find(w => w.name === "rows_json");
      // if (!rowsJson) return;

      // let saved = [];
      // try { saved = JSON.parse(rowsJson.value || "[]"); } catch { saved = []; }

      // node._fredClearAll?.();

      // (async () => {
        // if (Array.isArray(saved) && saved.length) {
          // for (const r of saved) await node._fredAddRow?.(r);
        // } else {
          // await node._fredAddRow?.();
        // }
      // })();
    // };
  // },
// });



// // Minimal dynamic UI for FRED_WildcardConcat_Dynamic
// // - File: dropdown (from /fred/wildcards/files)
// // - Line: dropdown with "🎲 random" (value stored as "random")
// // - Weight: number (step 0.01)
// // - Rows are serialized into the declared "rows_json" widget.

// import { app } from "/scripts/app.js";

// const RANDOM_VALUE = "random";
// const RANDOM_LABEL = "🎲 random";

// app.registerExtension({
  // name: "fred.wildcards.dynamic",

  // beforeRegisterNodeDef(nodeType, nodeData) {
    // if (nodeData?.name !== "FRED_WildcardConcat_Dynamic") return;

    // // optional but safe with dynamic widgets:
    // nodeType.prototype.serialize_widgets = true;

    // const superCreate = nodeType.prototype.onNodeCreated;
    // const superConfigure = nodeType.prototype.configure;

    // nodeType.prototype.onNodeCreated = function () {
      // if (superCreate) superCreate.apply(this, arguments);
      // const node = this;

      // const rowsJson = (node.widgets || []).find(w => w.name === "rows_json");
      // const prefixW  = (node.widgets || []).find(w => w.name === "prefix");
      // if (!rowsJson) { console.warn("[FRED] rows_json not found."); return; }

      // // hide rows_json
      // rowsJson.hidden = true;
      // rowsJson.computeSize = () => [0, 0];

      // // simple caches
      // const cache = { files: null, lines: {} };
      // async function listFiles() {
        // if (cache.files) return cache.files;
        // const r = await fetch("/fred/wildcards/files");
        // const j = await r.json();
        // cache.files = Array.isArray(j.files) ? j.files : [];
        // return cache.files;
      // }
      // async function listLines(file) {
        // if (!file) return [RANDOM_VALUE];
        // if (cache.lines[file]) return cache.lines[file];
        // const r = await fetch(`/fred/wildcards/lines?file=${encodeURIComponent(file)}`);
        // const j = await r.json();
        // const arr = Array.isArray(j.lines) ? j.lines : [RANDOM_VALUE];
        // cache.lines[file] = arr;
        // return arr;
      // }

      // // move prefix visually to be after Clear All (we'll recreate Clear later)
      // // (we’ll add Clear after we define helpers)

      // function syncRowsJson() {
        // const payload = (node._fred_rows || []).map(r => ({
          // on: r.data.on,
          // file: r.data.file,
          // line: r.data.line === RANDOM_LABEL ? RANDOM_VALUE : r.data.line,
          // weight: Number.isFinite(r.data.weight) ? Math.round(r.data.weight * 100) / 100 : 1.0,
          // suffix: r.data.suffix ?? ""
        // }));
        // rowsJson.value = JSON.stringify(payload);
        // // important: mark the graph dirty so value is persisted server-side
        // app.graph.setDirtyCanvas(true, true);
        // app.graph.change(); // <- ensures ComfyUI knows widget value changed
      // }

      // // expose for configure()
      // node._fredSync = syncRowsJson;

      // // helper to add one row (exposed for configure)
      // node._fredAddRow = async (preset) => {
        // node._fred_rows ??= [];
        // const data = {
          // on: true,
          // file: (preset && preset.file) || "",
          // line: (preset && preset.line)
                  // ? (preset.line === RANDOM_VALUE ? RANDOM_LABEL : preset.line)
                  // : RANDOM_LABEL,
          // weight: (preset && preset.weight != null)
                    // ? Math.round(preset.weight * 100) / 100
                    // : 1.0,
          // suffix: (preset && typeof preset.suffix === "string") ? preset.suffix : "",
        // };

        // const idx = (node._fred_rows?.length || 0) + 1;

        // const sep = node.addWidget("separator", `— Row ${idx} —`, null, ()=>{}, { serialize:false });

        // const onW = node.addWidget("toggle", `On ${idx}`, data.on, (v)=>{
          // data.on = !!v; node._fredSync();
        // }, { serialize:false });

        // const fileW = node.addWidget("combo", `File ${idx}`, data.file, async (v)=>{
          // data.file = v || "";
          // const rawLines = await listLines(data.file);
          // const display = rawLines.map(x => x === RANDOM_VALUE ? RANDOM_LABEL : x);
          // lineW.options.values = display;
          // if (!display.includes(data.line)) {
            // data.line = RANDOM_LABEL;
            // lineW.value = RANDOM_LABEL;
          // }
          // node._fredSync();
        // }, { values: [], serialize:false });

        // const lineW = node.addWidget("combo", `Line ${idx}`, data.line, (v)=>{
          // data.line = v || RANDOM_LABEL;
          // node._fredSync();
        // }, { values: [RANDOM_LABEL], serialize:false });

        // const weightW = node.addWidget("number", `Weight ${idx}`, data.weight, (v)=>{
          // const n = Number.isFinite(v) ? Math.round(v * 100) / 100 : 1.0;
          // data.weight = n; weightW.value = n;
          // node._fredSync();
        // }, { min:-10, max:10, step:0.01, precision: 2, serialize:false });

        // const sfxW = node.addWidget("string", `Suffix ${idx}`, data.suffix, (v)=>{
          // data.suffix = (v ?? "");
          // node._fredSync();
        // }, { serialize:false, multiline: false });

        // const rmW = node.addWidget("button", `Remove ${idx}`, null, ()=>{
          // data.on = false;
          // [sep,onW,fileW,lineW,weightW,sfxW,rmW].forEach(w=>{
            // const i = node.widgets.indexOf(w);
            // if (i !== -1) node.widgets.splice(i,1);
          // });
          // node._fred_rows = node._fred_rows.filter(r => r.widgets.rmW !== rmW);
          // relabel();
          // node._fredSync();
          // return true;
        // }, { serialize:false });

        // // populate file + lines
        // const files = await listFiles();
        // fileW.options.values = files;
        // if (!files.includes(data.file)) {
          // data.file = files[0] || "";
          // fileW.value = data.file;
        // }
        // const rawLines = await listLines(data.file);
        // const display = rawLines.map(x => x === RANDOM_VALUE ? RANDOM_LABEL : x);
        // lineW.options.values = display;
        // if (!display.includes(data.line)) {
          // data.line = RANDOM_LABEL;
          // lineW.value = RANDOM_LABEL;
        // }

        // node._fred_rows.push({ data, widgets:{sep,onW,fileW,lineW,weightW,sfxW,rmW} });
        // node._fredSync();
        // node.graph.setDirtyCanvas(true, true);
      // };

      // function relabel() {
        // let n = 1;
        // for (const r of (node._fred_rows || [])) {
          // r.widgets.sep.name     = `— Row ${n} —`;
          // r.widgets.onW.name     = `On ${n}`;
          // r.widgets.fileW.name   = `File ${n}`;
          // r.widgets.lineW.name   = `Line ${n}`;
          // r.widgets.weightW.name = `Weight ${n}`;
          // r.widgets.sfxW.name    = `Suffix ${n}`;
          // r.widgets.rmW.name     = `Remove ${n}`;
          // n++;
        // }
      // }

      // // expose clear-all so configure() can reuse it
      // node._fredClearAll = () => {
        // if (!node._fred_rows) node._fred_rows = [];
        // for (const r of node._fred_rows) {
          // const ws = r.widgets;
          // [ws.sep,ws.onW,ws.fileW,ws.lineW,ws.weightW,ws.sfxW,ws.rmW].forEach(w=>{
            // const i = node.widgets.indexOf(w);
            // if (i !== -1) node.widgets.splice(i,1);
          // });
        // }
        // node._fred_rows = [];
        // rowsJson.value = "[]";
        // app.graph.setDirtyCanvas(true, true);
        // app.graph.change();
      // };

      // // top buttons
      // node.addWidget("button", "➕ Add Wildcard", null, ()=> node._fredAddRow(), { serialize:false });
      // const clearBtn = node.addWidget("button", "Clear All", null, ()=> node._fredClearAll(), { serialize:false });

      // // move prefix after Clear All
      // if (prefixW) {
        // const iPref  = node.widgets.indexOf(prefixW);
        // const iAfter = node.widgets.indexOf(clearBtn);
        // if (iPref !== -1 && iAfter !== -1 && iPref < iAfter) {
          // node.widgets.splice(iPref, 1);
          // node.widgets.splice(iAfter + 1, 0, prefixW);
        // }
      // }

      // // Initial build will now be handled by configure(), not here.
    // };

    // // 🧠 Rehydrate AFTER ComfyUI has applied saved widgets_values
    // nodeType.prototype.configure = function (info) {
      // if (superConfigure) superConfigure.apply(this, arguments);
      // const node = this;
      // const rowsJson = (node.widgets || []).find(w => w.name === "rows_json");
      // if (!rowsJson) return;

      // let saved = [];
      // try { saved = JSON.parse(rowsJson.value || "[]"); } catch { saved = []; }

      // node._fredClearAll?.();

      // (async () => {
        // if (Array.isArray(saved) && saved.length) {
          // for (const r of saved) await node._fredAddRow?.(r);
        // } else {
          // await node._fredAddRow?.();
        // }
      // })();
    // };
  // },
// });
