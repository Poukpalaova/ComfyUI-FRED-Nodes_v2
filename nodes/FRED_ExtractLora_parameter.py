from comfy.comfy_types.node_typing import ComfyNodeABC, IO

import os

HELP_MESSAGE = """

👑 FRED_ExtractLora_parameter

🔹 PURPOSE

- Extracting up to 10 LoRA parameters: names (without path/extension) and weights from a connected LoRA loader or stacker node.

📥 INPUTS

- lora_stack: Accept FRED_AutoLoraLoader or compatible LoRA stacker nodes.

⚙️ KEY BEHAVIOR

- Traverses the workflow to find the connected LoRA node via the input link.

- Extracts each LoRA filename and removes path and extension.

- Extracts each LoRA weight with error handling and fallback values.

- Compatible with both formats: lora_name_X/lora_strength_X (1-indexed) and lora_X/strength_X (0-indexed).

📤 OUTPUTS

- lora_name_1, lora_weight_1, lora_name_2, lora_weight_2... up to lora_name_10, lora_weight_10

- help (string): Help message for UI assistance.

📝 NOTES & TIPS

- Ensures robust extraction even if some parameters are missing.

- Compatible with FRED_AutoLoraLoader and most LoRA stacker nodes.

- Returns empty string "" for unused LoRA slots and 0.0 for their weights.

"""

class FRED_ExtractLora_parameter(ComfyNodeABC):

    @classmethod
    def IS_CHANGED(cls, *, lora_stack, **kwargs):
        if lora_stack is not None:
            return float("NaN")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_stack": (IO.ANY, ),
            },
            "hidden": {
                "extra_pnginfo": "EXTRA_PNGINFO",
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = (
        "STRING", "FLOAT", "STRING", "FLOAT", "STRING", "FLOAT",
        "STRING", "FLOAT", "STRING", "FLOAT", "STRING", "FLOAT",
        "STRING", "FLOAT", "STRING", "FLOAT", "STRING", "FLOAT",
        "STRING", "FLOAT", "STRING"
    )

    RETURN_NAMES = (
        "lora_name_1", "lora_weight_1",
        "lora_name_2", "lora_weight_2",
        "lora_name_3", "lora_weight_3",
        "lora_name_4", "lora_weight_4",
        "lora_name_5", "lora_weight_5",
        "lora_name_6", "lora_weight_6",
        "lora_name_7", "lora_weight_7",
        "lora_name_8", "lora_weight_8",
        "lora_name_9", "lora_weight_9",
        "lora_name_10", "lora_weight_10",
        "help"
    )

    FUNCTION = "extract_lora_parameters"
    CATEGORY = "👑FRED/utils"
    DESCRIPTION = "Extracts up to 10 LoRA names (without path/extension) and weights from connected LoRA loader or stacker nodes."

    def extract_lora_parameters(self, lora_stack=None, extra_pnginfo=None, prompt=None, unique_id=None):
        workflow = extra_pnginfo["workflow"]
        link_id = None
        node_id = None
        link_to_node_map = {}

        # Chercher le node Widget qui est CE node, pour récupérer le link_id de lora_stack
        for node in workflow["nodes"]:
            if node["type"] == "FRED_ExtractLora_parameter" and node["id"] == int(unique_id) and not link_id:
                for node_input in node["inputs"]:
                    if node_input["name"] == "lora_stack":
                        link_id = node_input["link"]

            node_outputs = node.get("outputs", None)
            if not node_outputs:
                continue

            for output in node_outputs:
                node_links = output.get("links", None)
                if node_links is None or not isinstance(node_links, (list, tuple)):
                    continue

                for link in node_links:
                    link_to_node_map[link] = node["id"]
                    if link_id and link == link_id:
                        break

        if link_id:
            node_id = link_to_node_map.get(link_id, None)

        if node_id is None:
            raise ValueError("No node connected to lora_stack link found.")

        node_values = prompt[str(node_id)]
        inputs = node_values.get("inputs", {})

        # Liste pour stocker les résultats dans l'ordre entrelacé
        results = []

        # Extraire les 10 LoRAs
        for i in range(1, 11):
            # Essayer de trouver le nom du LoRA dans différents formats
            lora_name_full = None
            
            # Format 1: lora_name_X (1-indexed) - FRED_AutoLoraLoader
            lora_name_full = inputs.get(f"lora_name_{i}", None)
            
            # Format 2: lora_X (0-indexed) - LoRA stackers
            if lora_name_full is None:
                lora_name_full = inputs.get(f"lora_{i-1}", None)

            try:
                # Extraire juste le nom du fichier sans path ni extension
                if lora_name_full and lora_name_full != "None":
                    lora_name_full = str(lora_name_full)
                    # Enlever le path
                    lora_name = os.path.basename(lora_name_full)
                    # Enlever l'extension
                    lora_name = os.path.splitext(lora_name)[0]
                else:
                    lora_name = ""
            except Exception as e:
                lora_name = ""
                print(f"lora_name_{i} widget not found or error extracting name: {e}, fallback to empty string")

            # Essayer de trouver le poids du LoRA dans différents formats
            lora_weight = None
            
            # Format 1: lora_strength_X (1-indexed)
            lora_weight = inputs.get(f"lora_strength_{i}", None)
            
            # Format 1b: lora_model_strength_X (1-indexed) - fallback
            if lora_weight is None:
                lora_weight = inputs.get(f"lora_model_strength_{i}", None)
            
            # Format 2: strength_X (0-indexed) - LoRA stackers
            if lora_weight is None:
                lora_weight = inputs.get(f"strength_{i-1}", None)

            try:
                if lora_name == "":
                    lora_weight = 0.0
                else:
                    lora_weight = round(float(lora_weight), 2)
            except Exception as e:
                lora_weight = 0.0
                print(f"lora_weight_{i} widget not found, fallback to value 0.0: {e}")

            # Ajouter dans l'ordre: name, weight
            results.append(lora_name)
            results.append(lora_weight)

        # Ajouter le message d'aide à la fin
        results.append(HELP_MESSAGE)

        # Retourne tous les paramètres dans l'ordre entrelacé
        return tuple(results)

NODE_CLASS_MAPPINGS = {
    "FRED_ExtractLora_parameter": FRED_ExtractLora_parameter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FRED_ExtractLora_parameter": "👑 FRED_ExtractLora_parameter",
}