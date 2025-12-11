import { app } from "../../scripts/app.js";

// Fonction helper pour parser les couleurs CSS
function parseCssColor(color) {
    const hexMatch = color.match(/^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i);
    if (hexMatch) {
        return {
            values: [
                parseInt(hexMatch[1], 16),
                parseInt(hexMatch[2], 16),
                parseInt(hexMatch[3], 16)
            ]
        };
    }
    return null;
}

// Fonction helper pour déterminer si une couleur est claire
function isColorBright(rgb, threshold = 125) {
    const brightness = (rgb[0] * 299 + rgb[1] * 587 + rgb[2] * 114) / 1000;
    return brightness > threshold;
}

// Définition du widget FRED_COLOR
const FredColorWidget = {
    COLOR(key, val, compute = false) {
        const widget = {};
        widget.y = 0;
        widget.name = key;
        widget.type = "FRED_COLOR";
        widget.options = { default: "#ff0000" };
        widget.value = val || "#ff0000";
        
        widget.draw = function(ctx, node, widgetWidth, widgetY, height) {
            const hide = this.type !== "FRED_COLOR" || app.canvas.ds.scale < 0.5;
            if (hide) return;
            
            const border = 3;
            
            // Fond noir
            ctx.fillStyle = "#000";
            ctx.fillRect(0, widgetY, widgetWidth, height);
            
            // Rectangle de couleur
            ctx.fillStyle = this.value;
            ctx.fillRect(border, widgetY + border, widgetWidth - border * 2, height - border * 2);
            
            // Texte du nom et de la valeur hex
            const color = parseCssColor(this.value || this.options.default);
            if (!color) return;
            
            ctx.fillStyle = isColorBright(color.values, 125) ? "#000" : "#fff";
            ctx.font = "14px Arial";
            ctx.textAlign = "center";
            ctx.fillText(this.name, widgetWidth * 0.5, widgetY + 14);
        };
        
        widget.mouse = function(e, pos, node) {
            if (e.type === "pointerdown") {
                const widgets = node.widgets.filter((w) => w.type === "FRED_COLOR");
                
                for (const w of widgets) {
                    // Vérifier si le clic est dans la zone du widget
                    const rect = [w.last_y, w.last_y + 32];
                    if (pos[1] >= rect[0] && pos[1] <= rect[1]) {
                        // Créer un color picker
                        const picker = document.createElement("input");
                        picker.type = "color";
                        picker.value = this.value;
                        
                        Object.assign(picker.style, {
                            position: "fixed",
                            left: `${e.clientX}px`,
                            top: `${e.clientY}px`,
                            height: "0px",
                            width: "0px",
                            padding: "0px",
                            opacity: "0"
                        });
                        
                        picker.addEventListener("blur", () => {
                            this.callback?.(this.value);
                            node.graph.version++;
                            picker.remove();
                        });
                        
                        picker.addEventListener("input", () => {
                            if (!picker.value) return;
                            this.value = picker.value;
                            app.canvas.setDirty(true);
                        });
                        
                        document.body.appendChild(picker);
                        requestAnimationFrame(() => {
                            picker.showPicker();
                            picker.focus();
                        });
                    }
                }
            }
        };
        
        widget.computeSize = function(width) {
            return [width, 32];
        };
        
        return widget;
    }
};

// Enregistrement de l'extension
app.registerExtension({
    name: "FRED.ColorWidget",
    
    getCustomWidgets() {
        return {
            FRED_COLOR(node, inputName, inputData, app) {
                return {
                    widget: node.addCustomWidget(
                        FredColorWidget.COLOR(
                            inputName,
                            inputData[1]?.default || "#ff0000"
                        )
                    ),
                    minWidth: 150,
                    minHeight: 30
                };
            }
        };
    }
});
