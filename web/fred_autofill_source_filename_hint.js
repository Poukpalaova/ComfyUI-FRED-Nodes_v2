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
    
    // Trouver le widget d'image
    let imgW = node.widgets.find(w => w?.name === "image");
    if (!imgW) imgW = node.widgets.find(w => w?.options?.image_upload === true);
    if (!imgW) return;
    
    // Stocker la dernière valeur réelle (non-clipspace)
    let lastRealValue = imgW.value;
    
    // 1) Sync initial
    setHint(node, imgW.value);
    
    // 2) Wrapper du callback du widget
    const origCb = imgW.callback?.bind(imgW);
    imgW.callback = (...args) => {
        // Sauvegarder si ce n'est pas clipspace
        if (imgW.value && !imgW.value.startsWith("clipspace/")) {
            lastRealValue = imgW.value;
            setHint(node, imgW.value);
        } else if (imgW.value?.startsWith("clipspace/")) {
            // Garder l'ancienne valeur quand on passe en clipspace
            setHint(node, lastRealValue);
        } else {
            setHint(node, imgW.value);
        }
        return origCb ? origCb(...args) : undefined;
    };
    
    // 3) Wrapper de onWidgetChanged
    const origOnWidgetChanged = node.onWidgetChanged?.bind(node);
    node.onWidgetChanged = (name, value, widget) => {
        if (widget === imgW || name === imgW.name) {
            if (value && !value.startsWith("clipspace/")) {
                lastRealValue = value;
                setHint(node, value);
            } else if (value?.startsWith("clipspace/")) {
                setHint(node, lastRealValue);
            } else {
                setHint(node, value);
            }
        }
        return origOnWidgetChanged ? origOnWidgetChanged(name, value, widget) : undefined;
    };
}

app.registerExtension({
    name: "fred.autofillSourceFilenameHint",
    nodeCreated(node) {
        attachToImageWidget(node);
    }
});