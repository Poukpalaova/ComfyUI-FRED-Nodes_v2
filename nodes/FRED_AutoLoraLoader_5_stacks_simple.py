# FRED_AutoLoraLoader_5_stacks_simple.py

from nodes import LoraLoader

import os
import sys
import comfy.sd
import comfy.utils
import folder_paths

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

HELP_MESSAGE = """
👑 FRED_AutoLoraLoader_5_stacks_simple

🔹 PURPOSE

- Automatically apply up to 5 LoRAs based on keyword detection and simple on/off/auto switches.

📥 INPUTS

- model: Base model to receive the LoRAs (required).
- optional_clip: Optional CLIP to be updated alongside the model.
- toggle: Global ON/OFF switch for this node's application.
- text_prompt: Text used when switches are set to 'auto' (keywords are matched, comma-separated).
- loras_infos (optional): Compact string to prepend to the output list, unchanged and unparsed.
  Format examples: "name[:weight],name2[:weight2],..." or "name,weight,name2,weight2,...".

- For each slot 1..5 (optional):
  - switch_X: one of ["auto","off","on"].
  - lora_name_X: LoRA filename from the loras folder (or "None").
  - lora_weight_X: LoRA strength (float).
  - search_word_X: Comma-separated keywords; used only when switch_X == "auto".

⚙️ KEY BEHAVIOR

- Applies only the LoRAs marked active (on, or auto with matched keywords), in slot order 1..5.
- Updates both model and CLIP via the standard LoraLoader node.
- Builds an output loras_infos string:
  • If the input loras_infos is non-empty, it is placed first unchanged.
  • Then the node's active LoRAs are appended as "name:weight", joined by commas.
  • No parsing or mixing of the input is performed; it is strictly prefixed.

📤 OUTPUTS

- model: Updated model with active LoRAs applied.
- clip: Updated CLIP (or the same if none provided).
- lora_name_1, lora_weight_1, ..., lora_name_5, lora_weight_5: Names (basename without extension) and weights of active slots (empty/0.0 if not active).
- loras_infos: Compact combined list: input first (if any), then this node's active LoRAs "name:weight", comma-separated.
- help: This help message.

📝 NOTES & TIPS

- The input loras_infos is not used to apply LoRAs; it is only concatenated to the output string.
- The node preserves order and skips inactive slots cleanly (empty name/0.0 weight).
- Use 'auto' with search_word_X to trigger a LoRA only when a keyword appears in text_prompt.
"""

class FRED_AutoLoraLoader_5_stacks_simple:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        loras = ["None"] + folder_paths.get_filename_list("loras")
        inputs = {
            "required": {
                "model": ("MODEL",),
                "toggle": ("BOOLEAN", {"label_on": "on", "label_off": "off"}),
            },
            "optional": {
                "optional_clip": ("CLIP",),
                "text_prompt": ("STRING", {"forceInput": True}),
                # ajout: entrée texte simple, non parsée, juste concaténée en tête de l'output
                "loras_infos": ("STRING", {"default": "", "forceInput": True}),
            },
        }
        for i in range(1, 6):
            inputs["optional"][f"switch_{i}"] = (["auto", "off", "on"], {"default": "off"})
            inputs["optional"][f"lora_name_{i}"] = (loras, {"default": "None"})
            inputs["optional"][f"lora_weight_{i}"] = ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01})
            inputs["optional"][f"search_word_{i}"] = ("STRING", {"default": ""})
        return inputs

    RETURN_TYPES = ("MODEL", "CLIP",
                    "STRING", "FLOAT", "STRING", "FLOAT", "STRING", "FLOAT",
                    "STRING", "FLOAT", "STRING", "FLOAT",
                    "STRING", "STRING")
    RETURN_NAMES = ("model", "clip",
                    "lora_name_1", "lora_weight_1", "lora_name_2", "lora_weight_2", "lora_name_3", "lora_weight_3",
                    "lora_name_4", "lora_weight_4", "lora_name_5", "lora_weight_5",
                    "loras_infos", "help")
    OUTPUT_TOOLTIPS = ("", "", "", "", "", "", "", "", "", "", "", "", "", "")
    FUNCTION = "stack"
    CATEGORY = "👑 FRED/loras"
    DESCRIPTION = "Auto-applies up to 5 LoRAs with on/off/auto modes and emits a compact loras_infos string."

    @staticmethod
    def IS_CHANGED(**kwargs):
        # Tout sauf les objets lourds (model/clip) peut participer
        clean = {k: v for k, v in kwargs.items() if k not in {"model", "optional_clip"}}
        try:
            import json
            return json.dumps(clean, sort_keys=True, ensure_ascii=False)
        except Exception:
            return str(clean)

    def stack(self, model, toggle, text_prompt="", optional_clip=None, **kwargs):
        # Récupérer l'entrée loras_infos telle quelle (string)
        loras_infos_input = kwargs.get("loras_infos", "")
        if isinstance(loras_infos_input, (list, tuple)) and loras_infos_input:
            loras_infos_input = loras_infos_input[0]
        loras_infos_input = (loras_infos_input or "").strip()

        # Si désactivé, retourner sans appliquer les LoRAs, mais propager l'entrée dans l'output
        if not toggle:
            return (
                model, optional_clip,
                "", 0.0, "", 0.0, "", 0.0, "", 0.0, "", 0.0,
                loras_infos_input,  # output = entrée seule si OFF
                HELP_MESSAGE,
            )

        text_prompt = (text_prompt or "")
        if isinstance(text_prompt, (list, tuple)) and text_prompt:
            text_prompt = text_prompt[0] or ""
        text_prompt_lc = text_prompt.lower()

        # Sorties des slots
        lora_names_output = ["", "", "", "", ""]
        lora_weights_output = [0.0, 0.0, 0.0, 0.0, 0.0]

        # Liste des LoRA à appliquer (chemin/nom tel que LoraLoader l'attend) et leur poids
        loras_to_apply = []
        # Liste compacte pour l'output (basename sans extension)
        loras_infos_list = []

        # Détecter/appuyer les 5 slots
        for i in range(1, 6):
            switch = kwargs.get(f"switch_{i}", "off")
            lora_name = kwargs.get(f"lora_name_{i}", "None")
            lora_weight = kwargs.get(f"lora_weight_{i}", 1.0)
            search_word = kwargs.get(f"search_word_{i}", "")

            if isinstance(lora_name, (list, tuple)) and lora_name:
                lora_name = lora_name[0]
            if isinstance(lora_weight, (list, tuple)) and lora_weight:
                lora_weight = lora_weight[0]
            if isinstance(search_word, (list, tuple)) and search_word:
                search_word = search_word[0]

            # Skip None
            if not lora_name or lora_name == "None":
                continue

            # Active?
            active = False
            if switch == "on":
                active = True
            elif switch == "auto":
                kws = [w.strip().lower() for w in str(search_word or "").split(",") if w.strip()]
                for w in kws:
                    if w and w in text_prompt_lc:
                        active = True
                        break

            if not active:
                continue

            # Nom de base sans chemin/extension
            base_name = os.path.splitext(os.path.basename(lora_name))[0]
            try:
                weight_f = float(lora_weight)
            except Exception:
                weight_f = 1.0

            # Enregistrer pour sortie compacte
            loras_infos_list.append(f"{base_name}:{weight_f}")

            # Enregistrer pour sorties individuelles
            slot_idx = i - 1
            lora_names_output[slot_idx] = base_name
            lora_weights_output[slot_idx] = weight_f

            # Appliquer via LoraLoader (model, clip, path, strength_model, strength_clip)
            loras_to_apply.append((lora_name, weight_f, weight_f))

        # Appliquer les LoRA actifs (dans l'ordre 1..5)
        clip = optional_clip
        for ln, wm, wc in loras_to_apply:
            model, clip = LoraLoader().load_lora(model, clip, ln, wm, wc)

        # Construire l'output compact: entrée d'abord (inchangée), puis la liste du node
        node_compact = ",".join([x for x in loras_infos_list if x])
        if loras_infos_input and node_compact:
            loras_infos_out = f"{loras_infos_input},{node_compact}"
        elif loras_infos_input:
            loras_infos_out = loras_infos_input
        else:
            loras_infos_out = node_compact

        return (
            model, clip,
            lora_names_output[0], lora_weights_output[0],
            lora_names_output[1], lora_weights_output[1],
            lora_names_output[2], lora_weights_output[2],
            lora_names_output[3], lora_weights_output[3],
            lora_names_output[4], lora_weights_output[4],
            loras_infos_out,
            HELP_MESSAGE,
        )

NODE_CLASS_MAPPINGS = {
    "FRED_AutoLoraLoader_5_stacks_simple": FRED_AutoLoraLoader_5_stacks_simple,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FRED_AutoLoraLoader_5_stacks_simple": "👑 FRED AutoLoRA 5 (simple)",
}