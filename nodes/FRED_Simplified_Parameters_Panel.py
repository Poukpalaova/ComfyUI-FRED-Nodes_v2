from __future__ import annotations
import os
import comfy.samplers
import comfy.sample
from comfy_extras.nodes_custom_sampler import Noise_RandomNoise, Noise_EmptyNoise
from comfy.comfy_types.node_typing import ComfyNodeABC, InputTypeDict, IO
import torch

_DEFAULT_WIDGET_CANDIDATES = [
    "ckpt_name", "ckpt", "model_name", "model",
    "model_path", "checkpoint", "checkpoint_path", "filename", "NAME_STRING",
]

HELP_MESSAGE = """
👑 FRED_Simplified_Parameters_Panel

🔹 PURPOSE:
Provide a simplified panel to control common generation parameters (scale, denoise, guidance, steps, sampler/scheduler) and output valid ComfyUI objects.

📥 INPUTS:
- model • input MODEL
- scale • scaling factor
- denoise • denoising strength
- guidance • classifier-free guidance
- steps • number of steps
- noise_seed • seed value
- sampler • sampler choice
- scheduler • scheduler choice

⚙️ KEY OPTIONS:
- Automatically computes sigmas from steps & denoise
- Generates proper NOISE object
- Returns both sampler/scheduler objects and their names

📤 OUTPUTS:
- scale, denoise, guidance, steps
- noise (NOISE object)
- seed (INT)
- sampler (object) + sampler_name (STRING)
- sigmas (SIGMAS tensor)
- scheduler_name (STRING)
"""

class FRED_Simplified_Parameters_Panel(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "model": ("MODEL",),
                "scale": (IO.FLOAT, {"default": 1.4, "min": 0.1, "max": 10.0, "step": 0.01}),
                "denoise": (IO.FLOAT, {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
                "guidance": (IO.FLOAT, {"default": 2.2, "min": 0.0, "max": 30.0, "step": 0.1}),
                "steps": (IO.INT, {"default": 8, "min": 1, "max": 200, "control_after_generate": True}),
                "noise_seed": (IO.INT, {"default": 42, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "sampler": (comfy.samplers.SAMPLER_NAMES,),
                "scheduler": (comfy.samplers.SCHEDULER_NAMES,),
            },
            "optional": {
                # If your loader uses different widget names, provide a comma-separated list here.
                "custom_widget_names": (IO.STRING, {"default": "", "multiline": False}),
            },
            "hidden": {
                # needed to walk the graph and find which node is linked to our 'model' input
                "extra_pnginfo": "EXTRA_PNGINFO",
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = (
        IO.FLOAT,   # scale
        IO.FLOAT,   # denoise
        IO.FLOAT,   # guidance
        IO.INT,     # steps
        "NOISE",    # noise
        IO.INT,     # seed
        "SAMPLER",  # sampler
        "STRING",   # sampler_name
        "SIGMAS",   # scheduler
        "STRING",   # scheduler_name
        "STRING",   # model_name (basename, no path/ext)  <-- NEW
        "STRING",   # help_message
    )
    RETURN_NAMES = (
        "scale",
        "denoise",
        "guidance",
        "steps",
        "noise",
        "seed",
        "sampler",
        "sampler_name",
        "sigmas",
        "scheduler_name",
        "model_name",
        "help",
    )

    FUNCTION = "execute"
    CATEGORY = "👑FRED/utils"

    # --- helpers (same pattern as WidgetToString, but following the 'model' link) ---
    def _find_upstream_id_from_model_input(self, extra_pnginfo, unique_id):
        workflow = (extra_pnginfo or {}).get("workflow", {})
        if not workflow: 
            return None

        # 1) find THIS node in the UI graph; get link id that feeds its 'model' input
        link_id = None
        for node in workflow.get("nodes", []):
            if node.get("type") == "FRED_Simplified_Parameters_Panel" and node.get("id") == int(unique_id):
                for node_input in node.get("inputs", []):
                    if node_input.get("name") == "model":
                        link_id = node_input.get("link")
                break
        if not link_id:
            return None

        # 2) map link id -> producing node id
        link_to_node = {}
        for node in workflow.get("nodes", []):
            for outp in node.get("outputs", []) or []:
                for lnk in outp.get("links", []) or []:
                    link_to_node[lnk] = node.get("id")

        return link_to_node.get(link_id)

    def _to_basename(self, raw: str) -> str:
        if not raw:
            return ""
        s = str(raw).strip()
        if os.path.isdir(s):
            return os.path.basename(os.path.normpath(s))
        base = os.path.basename(s)
        name, _ext = os.path.splitext(base)
        return name or base or s

    def _model_name_from_upstream_widgets(self, custom_widget_names, extra_pnginfo, prompt, unique_id) -> str:
        uid = self._find_upstream_id_from_model_input(extra_pnginfo, unique_id)
        if uid is None:
            return ""

        try_list = []
        if custom_widget_names:
            try_list.extend([x.strip() for x in custom_widget_names.split(",") if x.strip()])
        try_list.extend(_DEFAULT_WIDGET_CANDIDATES)

        upstream = (prompt or {}).get(str(uid), {})
        inputs = upstream.get("inputs", {}) if isinstance(upstream, dict) else {}
        for key in try_list:
            if key in inputs:
                return self._to_basename(inputs[key])
        return ""

    def _model_name_from_model_object(self, model) -> str:
        inner = getattr(model, "model", None)
        candidates = []
        if inner is not None:
            for attr in ("model_path", "model_name", "ckpt_path", "checkpoint_path", "path", "filename"):
                if hasattr(inner, attr):
                    candidates.append(getattr(inner, attr))
        for attr in ("model_path", "model_name", "ckpt_path", "checkpoint_path", "path", "filename"):
            if hasattr(model, attr):
                candidates.append(getattr(model, attr))
        for val in candidates:
            if val:
                return self._to_basename(str(val))
        return getattr(inner or model, "__class__", type("X", (), {"__name__": ""})).__name__

    # --- main ---
    def execute(self, model, scale, denoise, guidance, steps, noise_seed, sampler, scheduler,
                custom_widget_names="", extra_pnginfo=None, prompt=None, unique_id=None):

        sampler_name = sampler
        scheduler_name = scheduler
        sampler_obj = comfy.samplers.sampler_object(sampler)

        total_steps = steps
        if denoise < 1.0:
            if denoise <= 0.0:
                return (torch.FloatTensor([]),)
            total_steps = int(steps / denoise)

        sigmas = comfy.samplers.calculate_sigmas(
            model.get_model_object("model_sampling"), scheduler, total_steps
        ).cpu()
        sigmas = sigmas[-(steps + 1):]

        noise = Noise_RandomNoise(noise_seed)

        # derive model_name via the 'model' link; fallback to MODEL object sniffing
        model_name = ""
        try:
            model_name = self._model_name_from_upstream_widgets(custom_widget_names, extra_pnginfo or {}, prompt or {}, unique_id or 0)
        except Exception:
            model_name = ""
        if not model_name:
            model_name = self._model_name_from_model_object(model)

        return (
            scale,
            denoise,
            guidance,
            steps,
            noise,
            noise_seed,
            sampler_obj,
            sampler_name,
            sigmas,
            scheduler_name,
            model_name,
            HELP_MESSAGE
        )

# Dictionary mapping node names to their corresponding classes
NODE_CLASS_MAPPINGS = {
    "FRED_Simplified_Parameters_Panel": FRED_Simplified_Parameters_Panel
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FRED_Simplified_Parameters_Panel": "👑 FRED Simplified Parameters Panel"
}