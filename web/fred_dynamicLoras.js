import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

app.registerExtension({
  name: "fred.dynamicLoras",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== "FRED_AutoLoraLoader_Dynamic") return;

    async function fetchLorasList() {
      try {
        // ComfyUI universal model listing API
        const models = await api.getModels?.("loras");
        if (Array.isArray(models) && models.length) {
          const names = models.map((m) => m.name || m).filter(Boolean);
          return ["None", ...names];
        }
      } catch (e) {
        console.warn("[fred.dynamicLoras] fetchLorasList fallback:", e);
      }
      return ["None"];
    }

    nodeType.prototype.onNodeCreated = async function () {
      const node = this;
      const maxLoraNum = 10;

      async function rebuildLoraWidgets(numLoras, mode) {
        const lorasList = await fetchLorasList();

        node.widgets = (node.widgets || []).filter(
          (w) =>
            !/^switch_\d+$/.test(w.name) &&
            !/^search_word_\d+$/.test(w.name) &&
            !/^lora_name_\d+$/.test(w.name) &&
            !/^lora_strength_\d+$/.test(w.name) &&
            !/^lora_model_strength_\d+$/.test(w.name) &&
            !/^lora_clip_strength_\d+$/.test(w.name)
        );

        for (let i = 1; i <= numLoras; i++) {
          node.addWidget("combo", `switch_${i}`, "off", undefined, {
            values: ["auto", "off", "on"],
            serialize: true,
          });
          node.addWidget("string", `search_word_${i}`, "", undefined, {
            serialize: true,
          });
          node.addWidget("combo", `lora_name_${i}`, "None", undefined, {
            values: lorasList,
            serialize: true,
          });

          if (mode === "simple") {
            node.addWidget("number", `lora_strength_${i}`, 1.0, undefined, {
              min: -10,
              max: 10,
              step: 0.01,
              serialize: true,
            });
          } else {
            node.addWidget("number", `lora_model_strength_${i}`, 1.0, undefined, {
              min: -10,
              max: 10,
              step: 0.01,
              serialize: true,
            });
            node.addWidget("number", `lora_clip_strength_${i}`, 1.0, undefined, {
              min: -10,
              max: 10,
              step: 0.01,
              serialize: true,
            });
          }
        }

        node.size = node.computeSize();
        node.graph?.setDirtyCanvas(true, true);
        console.log(`[fred.dynamicLoras] rebuilt num=${numLoras} mode=${mode}`);
      }

      function getNumWidget() {
        return node.widgets?.find((w) => w.name === "num_loras");
      }
      function getModeWidget() {
        return node.widgets?.find((w) => w.name === "mode");
      }

      async function rebuild() {
        const num = parseInt(getNumWidget()?.value || "1", 10);
        const mode = getModeWidget()?.value || "simple";
        await rebuildLoraWidgets(Math.max(1, Math.min(num, maxLoraNum)), mode);
      }

      // Hook mode changes (combo always fires)
      const modeW = getModeWidget();
      if (modeW) {
        const origCb = modeW.callback;
        modeW.callback = async function (v) {
          if (origCb) await origCb(v);
          await rebuild();
        };
      }

      // Poll num_loras every 300ms for smoother updates
      let lastNum = getNumWidget()?.value;
      const poll = setInterval(async () => {
        const cur = getNumWidget()?.value;
        if (cur !== lastNum) {
          lastNum = cur;
          await rebuild();
        }
      }, 300);

      // Cleanup on node delete
      const origOnRemoved = node.onRemoved;
      node.onRemoved = function () {
        clearInterval(poll);
        if (origOnRemoved) origOnRemoved.call(this);
      };

      await rebuild();
    };
  },
});



// import { app } from "/scripts/app.js";
// import { api } from "/scripts/api.js";

// app.registerExtension({
  // name: "fred.dynamicLoras",

  // async beforeRegisterNodeDef(nodeType, nodeData) {
    // if (nodeData?.name !== "FRED_applyLoraStackMerged") return;

    // async function fetchLorasList() {
      // // Try api.getLoras() first, fallback to the REST endpoint
      // try {
        // const loras = await (api.getLoras?.() || (async () => {
          // const res = await fetch("/api/loras/list");
          // const d = await res.json();
          // return d?.loras || [];
        // })());
        // return ["None", ...(loras || [])];
      // } catch (e) {
        // console.error("Error fetching LoRAs:", e);
        // return ["None"];
      // }
    // }

    // nodeType.prototype.onNodeCreated = async function () {
      // const node = this;
      // const maxLoraNum = 10;

      // console.log("[fred.dynamicLoras] onNodeCreated for", node.title, node.id);

      // // let lorasList = await fetchLorasList();

      // async function rebuildLoraWidgets(numLoras, mode) {
        // // try {
          // // lorasList = await fetchLorasList();
        // // } catch (e) {
          // // console.warn("[fred.dynamicLoras] rebuild: fetch failed, using fallback", e);
          // // lorasList = ["None"];
        // // }

        // // remove previous dynamic widgets
        // node.widgets = (node.widgets || []).filter(
          // (w) =>
            // !/^switch_\d+$/.test(w.name) &&
            // !/^search_word_\d+$/.test(w.name) &&
            // !/^lora_name_\d+$/.test(w.name) &&
            // !/^lora_strength_\d+$/.test(w.name) &&
            // !/^lora_model_strength_\d+$/.test(w.name) &&
            // !/^lora_clip_strength_\d+$/.test(w.name)
        // );

        // // add new widgets
        // for (let i = 1; i <= numLoras; i++) {
          // node.addWidget("combo", `switch_${i}`, "off", undefined, {
            // values: ["auto", "off", "on"],
            // serialize: true,
          // });
          // node.addWidget("string", `search_word_${i}`, "", undefined, {
            // serialize: true,
          // });
          // node.addWidget("combo", `lora_name_${i}`, "None", undefined, { serialize: true, });
          // // node.addWidget("combo", `lora_name_${i}`, "None", undefined, {
            // // values: lorasList.length > 0 ? lorasList : ["None"],
            // // serialize: true,
          // // });

          // if (mode === "simple") {
            // node.addWidget("number", `lora_strength_${i}`, 1.0, undefined, {
              // min: -10,
              // max: 10,
              // step: 0.01,
              // serialize: true,
            // });
          // } else {
            // node.addWidget("number", `lora_model_strength_${i}`, 1.0, undefined, {
              // min: -10,
              // max: 10,
              // step: 0.01,
              // serialize: true,
            // });
            // node.addWidget("number", `lora_clip_strength_${i}`, 1.0, undefined, {
              // min: -10,
              // max: 10,
              // step: 0.01,
              // serialize: true,
            // });
          // }
        // }

        // node.size = node.computeSize();
        // node.graph?.setDirtyCanvas(true, true);
        // // console.log(`[fred.dynamicLoras] rebuilt widgets: num=${numLoras} mode=${mode} loras=${lorasList.length}`);
        // console.log(`[fred.dynamicLoras] rebuilt widgets: num=${numLoras} mode=${mode}`);
      // }

      // // helpers to find the num and mode controls that exist on the node
      // const findNumWidget = () => node.widgets?.find((w) => w.name === "num_loras");
      // const findModeWidget = () => node.widgets?.find((w) => w.name === "mode");

      // function getMode() {
        // const mw = findModeWidget();
        // return mw ? mw.value || "simple" : "simple";
      // }

      // async function rebuild() {
        // try {
          // const nw = findNumWidget();
          // const num = parseInt(nw?.value || "1", 10);
          // const mode = getMode();
          // await rebuildLoraWidgets(Math.max(1, Math.min(num, maxLoraNum)), mode);
        // } catch (e) {
          // console.error("[fred.dynamicLoras] rebuild error:", e);
        // }
      // }

      // // For num_loras (INT widgets don't trigger callback reliably)
      // const origOnWidgetChanged = node.onWidgetChanged;
      // node.onWidgetChanged = async function (widget, value, prevValue) {
        // if (origOnWidgetChanged) origOnWidgetChanged.call(this, widget, value, prevValue);
        // if (widget?.name === "num_loras") {
          // console.log("[fred.dynamicLoras] num_loras changed →", value);
          // await rebuild();
        // }
      // };

      // // For mode (COMBO widgets DO fire callback reliably)
      // const modeWidget = node.widgets?.find((w) => w.name === "mode");
      // if (modeWidget) {
        // const origModeCb = modeWidget.callback;
        // modeWidget.callback = async function (v) {
          // if (origModeCb) await origModeCb(v);
          // console.log("[fred.dynamicLoras] mode changed →", v);
          // const numWidget = node.widgets?.find((w) => w.name === "num_loras");
          // const num = parseInt(numWidget?.value || "1", 10);
          // await rebuildLoraWidgets(num, v || "simple");
        // };
      // }

      // // POLL fallback for 'num_loras' in case change events are not emitted by this ComfyUI build.
      // // This runs while the node exists and is cleared when node is removed.
      // let lastNum = findNumWidget()?.value;
      // let pollInterval = setInterval(async () => {
        // const nw = findNumWidget();
        // const current = nw?.value;
        // if (current !== lastNum) {
          // console.log(`[fred.dynamicLoras] poll detected num_loras change: ${lastNum} -> ${current}`);
          // lastNum = current;
          // await rebuild();
        // }
      // }, 200);

      // // Clean up when node is removed from graph (prevents stray timers)
      // const origOnRemoved = node.onRemoved;
      // node.onRemoved = function () {
        // if (pollInterval) {
          // clearInterval(pollInterval);
          // pollInterval = null;
        // }
        // if (this._fred_rebuild_timeout) {
          // clearTimeout(this._fred_rebuild_timeout);
          // this._fred_rebuild_timeout = null;
        // }
        // if (origOnRemoved) origOnRemoved.call(this);
      // };

      // // Initial build
      // await rebuild();
    // };
  // },
// });