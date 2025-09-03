import { app } from "/scripts/app.js";

function setHint(node, val) {
  if (!node?.widgets) return;
  const hint = node.widgets.find(w => w?.name === "source_filename_hint");
  if (!hint) return;
  if (hint.value !== val) {
    hint.value = val || "";
    try { node.onInputChanged?.(hint.name, hint.value); } catch {}
    try { node.onPropertyChanged?.(hint.name, hint.value); } catch {}
    try { app.graph.setDirtyCanvas(true, true); } catch {}
  }
}

function attachToImageWidget(node) {
  if (!node?.widgets?.length) return;
  // Prefer an explicit "image" widget; otherwise, fall back to any widget with image_upload enabled.
  let imgW = node.widgets.find(w => w?.name === "image");
  if (!imgW) imgW = node.widgets.find(w => w?.options?.image_upload === true);
  if (!imgW) return;

  // 1) Initial sync (covers existing value when loading a workflow)
  setHint(node, imgW.value);

  // 2) Wrap the widget's own callback (if any)
  const origCb = imgW.callback?.bind(imgW);
  imgW.callback = (...args) => {
    try { setHint(node, imgW.value); } catch {}
    return origCb ? origCb(...args) : undefined;
  };

  // 3) Also watch generic widget change path
  const origOnWidgetChanged = node.onWidgetChanged?.bind(node);
  node.onWidgetChanged = (name, value, widget) => {
    if (widget === imgW || name === imgW.name) {
      try { setHint(node, value); } catch {}
    }
    return origOnWidgetChanged ? origOnWidgetChanged(name, value, widget) : undefined;
  };
}

app.registerExtension({
  name: "fred.autofillSourceFilenameHint",
  nodeCreated(node) {
    // Only attach on your loader, or attach to all nodes harmlessly.
    // If you want to scope: if (node?.type !== "FRED_LoadImage_V8") return;
    attachToImageWidget(node);
  }
});
