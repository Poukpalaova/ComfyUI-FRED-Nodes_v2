import os
import sys
import comfy.sd
import comfy.utils
import folder_paths
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

HELP_MESSAGE = """Automatically apply up to 10 LoRAs based on keyword detection in the input text.
Each LoRA can be set to:
- auto → activates only if its keyword(s) are found in the text prompt
- on → always applied
- off → ignored
You can enter multiple keywords separated by commas."""

class FRED_AutoLoraLoader_Dynamic:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        max_lora_num = 10
        loras = ["None"] + folder_paths.get_filename_list("loras")

        inputs = {
            "required": {
                "model": ("MODEL",),
                "toggle": ("BOOLEAN", {"label_on": "on", "label_off": "off"}),
                "mode": (["simple", "advanced"],),
                "num_loras": ("INT", {"default": 1, "min": 1, "max": max_lora_num}),
            },
            "optional": {
                "optional_clip": ("CLIP",),
                "text_prompt": ("STRING", {"forceInput": True}),
            },
        }

        for i in range(1, max_lora_num + 1):
            inputs["optional"][f"switch_{i}"] = (["auto", "off", "on"], {"default": "off"})
            inputs["optional"][f"search_word_{i}"] = ("STRING", {"default": ""})
            inputs["optional"][f"lora_name_{i}"] = (loras, {"default": "None"})
            inputs["optional"][f"lora_strength_{i}"] = ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01})
            inputs["optional"][f"lora_model_strength_{i}"] = ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01})
            inputs["optional"][f"lora_clip_strength_{i}"] = ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01})

        return inputs

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("model", "clip", "showhelp")
    FUNCTION = "stack"
    CATEGORY = "👑 FRED/lora"

    def stack(self, model, toggle, mode, num_loras, text_prompt="", optional_clip=None, **kwargs):
        if (toggle in [False, None, "False"]) or not kwargs:
            return (None,)

        loras = []
        for i in range(1, num_loras + 1):
            switch = kwargs.get(f"switch_{i}", "off")
            search_word = kwargs.get(f"search_word_{i}", "")
            lora_name = kwargs.get(f"lora_name_{i}")
            if not lora_name or lora_name == "None":
                continue

            is_active = False
            if switch == "on":
                is_active = True
            elif switch == "auto" and search_word:
                for word in [w.strip().lower() for w in search_word.split(",") if w.strip()]:
                    if word in text_prompt.lower():
                        is_active = True
                        break

            if not is_active:
                continue

            if mode == "simple":
                lora_strength = float(kwargs.get(f"lora_strength_{i}"))
                loras.append((lora_name, lora_strength, lora_strength))
            else:
                model_strength = float(kwargs.get(f"lora_model_strength_{i}"))
                clip_strength = float(kwargs.get(f"lora_clip_strength_{i}"))
                loras.append((lora_name, model_strength, clip_strength))

        clip = optional_clip
        for lora in loras:
            model, clip = comfy.sd.load_lora(model, lora[0], lora[1], lora[2])

        return (model, clip, HELP_MESSAGE)


NODE_CLASS_MAPPINGS = {
    "FRED_AutoLoraLoader_Dynamic": FRED_AutoLoraLoader_Dynamic
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FRED_AutoLoraLoader_Dynamic": "👑 FRED_AutoLoraLoader_Dynamic"
}