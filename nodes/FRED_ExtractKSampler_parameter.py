from comfy.comfy_types.node_typing import ComfyNodeABC, InputTypeDict, IO

HELP_MESSAGE = """
👑 FRED_ExtractKSampler_parameter

🔹 PURPOSE
- Extracting KSampler parameters such as seed, steps, cfg scale, sampler name, scheduler, and denoise value from connected node.

📥 INPUTS
- ksampler_input: Accept any KSampler or equivalent node that contains widgets including seed or noise_seed, steps, etc.

⚙️ KEY BEHAVIOR
- Traverses the workflow to find the connected node via the input link.
- Extracts and converts parameters with error handling and fallback values

📤 OUTPUTS
- seed (int): The seed value or noise_seed fallback.
- steps (int): Number of steps.
- cfg_guidance (float): CFG or Guidance scale rounded to 2 decimals.
- sampler (string): Name of the sampler used.
- scheduler (string): Scheduler or scheduler_name string.
- denoise (float): Denoising strength rounded to 2 decimals.

📝 NOTES & TIPS
- Ensures robust extraction even if some parameters are missing.
- Returns help message string as last output for UI assistance.
"""

class FRED_ExtractKSampler_parameter(ComfyNodeABC):
    @classmethod
    def IS_CHANGED(cls, *, ksampler_input, **kwargs):
        if ksampler_input is not None:
            return float("NaN")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ksampler_input": (IO.ANY, ),
            },
            "hidden": {
                "extra_pnginfo": "EXTRA_PNGINFO",
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = ("INT", "INT", "FLOAT", "STRING", "STRING", "FLOAT", "STRING")
    RETURN_NAMES = ("seed", "steps", "cfg_guidance", "sampler", "scheduler", "denoise", "help")
    FUNCTION = "extract_ksampler_parameters"
    CATEGORY = "👑FRED/utils"

    DESCRIPTION = "Extracts multiple parameters from the ksampler node connected via 'ksampler_input'."

    def extract_ksampler_parameters(self, ksampler_input=None, extra_pnginfo=None, prompt=None, unique_id=None):
        workflow = extra_pnginfo["workflow"]
        link_id = None
        node_id = None
        link_to_node_map = {}

        # Chercher le node Widget qui est CE node, pour récupérer le link_id de ksampler_input
        for node in workflow["nodes"]:
            if node["type"] == "FRED_ExtractKSampler_parameter" and node["id"] == int(unique_id) and not link_id:
                for node_input in node["inputs"]:
                    if node_input["name"] == "ksampler_input":
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
            raise ValueError("No node connected to ksampler_input link found.")

        node_values = prompt[str(node_id)]
        inputs = node_values.get("inputs", {})

        # Seed en int
        seed = inputs.get("seed", None)
        if seed is None:
            seed = inputs.get("noise_seed", None)
        try:
            seed = int(seed)
        except Exception:
            seed = 0
            print("seed or noise_seed widget not found, fallback to value 0")

        # Steps en int
        steps = inputs.get("steps", None)
        try:
            steps = int(steps)
        except Exception:
            steps = 0
            print("steps widget not found, fallback to value 0")

        # cfg en float arrondi à 2 décimales
        cfg = inputs.get("cfg", None)
        if cfg is None:
            cfg = inputs.get("guidance", None)
        try:
            cfg = round(float(cfg), 2)
        except Exception:
            cfg = 1.0
            print("cfg or guidance widget not found, fallback to value 1.0")

        # sampler_name en string
        sampler = inputs.get("sampler_name", None)
        if sampler is None:
            sampler = inputs.get("sampler", None)
        try:
            sampler = str(sampler)
        except Exception:
            sampler = "sampler_name"
            print("sampler_name or sampler widget not found, fallback to value sampler_name")

        # scheduler ou scheduler_name en string
        scheduler = inputs.get("scheduler", None)
        if scheduler is None:
            scheduler = inputs.get("scheduler_name", None)
        try:
            scheduler = str(scheduler)
        except Exception:
            scheduler = "scheduler"
            print("scheduler or scheduler_name widget not found, fallback to value scheduler")

        # denoise en float arrondi à 2 décimales
        denoise = inputs.get("denoise", None)
        try:
            denoise = round(float(denoise), 2)
        except Exception:
            denoise = 1.0
            print("denoise widget not found, fallback to value 1.0")

        # Retourne tous les paramètres plus le message d'aide
        return (seed, steps, cfg, sampler, scheduler, denoise, HELP_MESSAGE)

NODE_CLASS_MAPPINGS = {
    "FRED_ExtractKSampler_parameter": FRED_ExtractKSampler_parameter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FRED_ExtractKSampler_parameter": "👑 FRED_ExtractKSampler_parameter",
}