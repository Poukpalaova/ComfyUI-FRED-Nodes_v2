# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import json
import time
import math
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image, PngImagePlugin

# ComfyUI core helpers
import folder_paths
import comfy.utils
from comfy.comfy_types.node_typing import ComfyNodeABC, InputTypeDict, IO

print(hasattr(comfy.utils, 'save_image'))

HELP_MESSAGE = """
👑 FRED_Image_Saver

🔹 PURPOSE

Save images flexibly with dynamic filename/path tokens, Automatic1111-style metadata embedding, and optional grid saving.

Accepts **list or batch** as input (mapping disabled on images), so the node runs **once** for a whole set.

📥 INPUTS

- images • one or more images to save (**list or batch accepted; mapping disabled**).

• If a list contains a batch (e.g. [ (B,H,W,3) ]), it is flattened to B images.

• All images must share the same H×W.

- save_single_image • if ON, save each input image as its own file (in addition to the grid, if enabled).

- filename / path • templates with tokens for naming (a counter is always appended for uniqueness).

- time_format • used by %datetime.

- extension / quality_jpeg_or_webp / optimize_png / lossless_webp • single-image format & compression.

- save_workflow_as_json • writes a sidecar JSON file (metadata).

- embed_workflow_in_png • embeds workflow/prompt/app metadata into PNG tEXt.

- metadata fields (record-only; pixels are not changed):

width, height, scale, denoise, guidance, clip_skip, steps, seed_value,

sampler_name, scheduler_name, positive, negative, model_name

• Positive/Negative are written **only if non-empty** (prevents “Negative prompt:” swallowing params).

- lora_name_X / lora_weight_X (X=1..3) • up to 3 LoRAs. Weights are included **only if the name is set**.

- Grid options:

• save_as_grid_if_multi • if ON and there are ≥2 images, also save a tiled grid image.

• grid_column_max / grid_row_max • maximum grid size (auto layout chooses rows/cols ≤ these maxima).

• grid_filename / grid_path / grid_extension / grid_quality_jpeg_or_webp • grid output settings.

⚙️ KEY BEHAVIOR

- Single image saves:

• Saved under “path” (relative to Comfy output unless absolute).

• Uses tokenized filename; unique counter `_00001`, `_00002`, … appended.

• PNG can embed JSON + optimize (lossless). JPEG/WEBP use quality; WEBP can be lossless.

- Grid saves:

• A grid image is **returned** as the first output (tensor `(1,H,W,3)`) for preview/flow.

• A grid **file** is **saved only if** `save_as_grid_if_multi = ON` **and** there are ≥2 images.

• Auto layout respects `grid_column_max` / `grid_row_max`. If not all images fit, extras are ignored with a log.

• Grid metadata adds: `img_count`, `grid_rows`, `grid_cols`.

• If there is only 1 image, the returned tensor is a 1×1 grid; no grid file is saved.

🧾 METADATA (PNG tEXt “parameters” like Automatic1111)

- “parameters” line(s):

• Positive prompt line (if non-empty)

• “Negative prompt: …” (if non-empty)

• “Steps, Sampler, Scheduler, Guidance, Scale, Denoise, Seed, Size, Clip skip, Model”

• “Version: ComfyUI ”

• One line per LoRA: `LoRA_X: , LoRA_X_Weight: `

- Also embeds “prompt”, “workflow” (if present), and “app” details.

🔑 TOKENS (usable in filename and path)

- Time/date: `%date`, `%date_dash`, `%time`, `%datetime`

- Params: `%seed`/`%seed_value`, `%model`/`%model_name`/`%basemodelname`, `%sampler`/`%sampler_name`,

`%scheduler`/`%scheduler_name`, `%width`, `%height`, `%steps`, `%cfg`/`%guidance`,

`%scale`, `%denoise`, `%clip_skip`

- Batch info: `%img_count` (grid/batch size)

- LoRAs: `%lora_name_1..3`, `%lora_weight_1..3` (weight expands only if corresponding name is set)

📤 OUTPUTS

- GRID_IMAGE • the grid as `(1,H,W,3)` (or 1×1 if single input).

- last_saved_path • path of the **last** saved file (single image or grid). Empty if nothing saved.

- help • this message.

📝 NOTES & TIPS

- This node does **not** resize; width/height/steps/etc. are **recorded only**.

- `%basemodelname` expands to `model_name` without extension.

- PNG `optimize=True` reduces size but is slower. WEBP `lossless_webp=True` ignores quality.

- Unique filenames are guaranteed (Comfy counter + safety loop).

- **Mapping is disabled on `images`**: the node runs once for a whole list/batch.

- If other inputs arrive as lists (from widgets), they are coerced to their **first** value to avoid remapping.

- Set `save_single_image=False` if you only want the grid file.

- Mixed sizes? Make sure all inputs share the same H×W before saving a grid.

📚 EXAMPLES

- filename: "%basemodelname_%datetime_seed_%seed"

- path: "Fred_nodes/%date_dash/"

- grid_filename: "%model_name_%date_%time_grid_%img_count"

- grid_path: "Test/%model_name/Grid/%date_dash/"
"""

INVALID_FS_CHARS = '<>:"/\\|?*'

class FRED_ImageSaver(ComfyNodeABC):
    # Dites Ã  Comfy: j'accepte les listes sur les inputs â†’ ne mappe pas le node sur 'images'
    INPUT_IS_LIST = True
    # On ne retourne pas des listes (3 sorties scalaires)
    OUTPUT_IS_LIST = (False, False, False)

    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                # --- IMAGE & FILE OUTPUT ---
                "images": ("IMAGE", {"tooltip": "image(s) to save (list or batch)."}),
                "save_single_image": ("BOOLEAN", {"default": True, "tooltip": "Save each input image as its own file."}),
                "filename": (IO.STRING, {"default": "%model_name_%date_dash", "multiline": False, "tooltip": "base filename (counter appended)"}),
                "path": (IO.STRING, {"default": "Fred_nodes/%date_dash/", "multiline": False, "tooltip": "relative to Comfy output (or absolute)"}),
                "time_format": (IO.STRING, {"default": "%Y-%m-%d-%H%M%S", "tooltip": "used by %datetime token"}),
                "extension": (["png", "jpeg", "webp"], {"default": "png", "tooltip": "output file format for single images"}),
                "optimize_png": ("BOOLEAN", {"default": False, "tooltip": "optimize PNG (lossless); smaller files but slower"}),
                "lossless_webp": ("BOOLEAN", {"default": True, "tooltip": "save WEBP in lossless mode (quality ignored)"}),
                "quality_jpeg_or_webp": (IO.INT, {"default": 100, "min": 1, "max": 100, "tooltip": "JPEG/WEBP quality (ignored if WEBP lossless)"}),

                "save_workflow_as_json": ("BOOLEAN", {"default": False, "tooltip": "write a sidecar JSON next to image"}),
                "embed_workflow_in_png": ("BOOLEAN", {"default": True, "tooltip": "embed workflow/prompt JSON into PNG metadata"}),

                # --- CORE PARAMS (recording only; does not modify pixels) ---
                "width": (IO.INT, {"default": 1024, "min": 1, "max": 16384, "tooltip": "recorded in metadata; does not resize"}),
                "height": (IO.INT, {"default": 1024, "min": 1, "max": 16384, "tooltip": "recorded in metadata; does not resize"}),
                "scale": (IO.FLOAT, {"default": 1.40, "min": 0.0, "max": 100000.0, "step": 0.01, "tooltip": "record-only parameter"}),
                "denoise": (IO.FLOAT, {"default": 0.60, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "record-only parameter"}),
                "guidance": (IO.FLOAT, {"default": 2.20, "min": 0.0, "max": 100.0, "step": 0.01, "tooltip": "record-only parameter (CFG/guidance)"}),
                "clip_skip": (IO.INT, {"default": 1, "min": -24, "max": 24, "step": 1, "tooltip": "record-only parameter"}),
                "steps": (IO.INT, {"default": 20, "min": 1, "max": 4096, "tooltip": "record-only parameter"}),
                "seed_value": (IO.INT, {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "record-only parameter (seed)"}),
                "sampler_name": (IO.STRING, {"default": "euler", "multiline": False, "tooltip": "record-only parameter (sampler name)"}),
                "scheduler_name": (IO.STRING, {"default": "simple", "multiline": False, "tooltip": "record-only parameter (scheduler name)"}),
                "positive": (IO.STRING, {"default": "", "multiline": True, "tooltip": "positive prompt (saved in metadata)"}),
                "negative": (IO.STRING, {"default": "", "multiline": True, "tooltip": "negative prompt (saved in metadata)"}),
                "model_name": (IO.STRING, {"default": "flux", "tooltip": "record-only parameter (model name)"}),

                # --- OPTIONAL LORAs ---
                "lora_name_1": (IO.STRING, {"default": "", "tooltip": "optional LoRA name #1 (if empty: weight omitted everywhere)"}),
                "lora_weight_1": (IO.FLOAT, {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01, "tooltip": "LoRA weight #1"}),
                "lora_name_2": (IO.STRING, {"default": "", "tooltip": "optional LoRA name #2 (if empty: weight omitted everywhere)"}),
                "lora_weight_2": (IO.FLOAT, {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01, "tooltip": "LoRA weight #2"}),
                "lora_name_3": (IO.STRING, {"default": "", "tooltip": "optional LoRA name #3 (if empty: weight omitted everywhere)"}),
                "lora_weight_3": (IO.FLOAT, {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01, "tooltip": "LoRA weight #3"}),

                # --- GRID SAVE OPTIONS ---
                "save_as_grid_if_multi": ("BOOLEAN", {"default": False, "tooltip": "Also save a single tiled grid image of all inputs."}),
                "grid_filename": (IO.STRING, {"default": "%lora_name_1_weight_%lora_weight_1_%datetime_%model_name grid_%img_count", "multiline": False, "tooltip": "filename prefix for the grid image"}),
                "grid_path": (IO.STRING, {"default": "Test/%model_name/Grid/%date_dash/", "multiline": False, "tooltip": "save folder for the grid image"}),
                "grid_column_max": (IO.INT, {"default": 5, "min": 1, "step": 1, "tooltip": "x_size (columns)."}),
                "grid_row_max": (IO.INT, {"default": 1, "min": 1, "step": 1, "tooltip": "y_size (rows)."}),
                "grid_extension": (["png", "jpeg", "webp"], {"default": "png", "tooltip": "output file format for the grid image"}),
                "grid_quality_jpeg_or_webp": (IO.INT, {"default": 100, "min": 1, "max": 100, "tooltip": "JPEG/WEBP quality for the grid image"}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("GRID_IMAGE", "last_saved_path", "help")
    OUTPUT_TOOLTIPS = ("path to last saved image", "help / usage tips")
    FUNCTION = "save"
    CATEGORY = "ðŸ‘‘FRED/image"
    DESCRIPTION = "Save images with A1111-style parameters text and robust tokenized filenames/paths. Can also save a grid when there is multiple images."
    OUTPUT_NODE = True

    # ------------------------------ helpers ------------------------------
    def _scalar(self, x):
        return (x[0] if isinstance(x, (list, tuple)) and x else x)

    def _to_int(self, x, default=0):
        x = self._scalar(x)
        try:
            return int(x)
        except Exception:
            try:
                return int(float(x))
            except Exception:
                return int(default)

    def _to_float(self, x, default=0.0):
        x = self._scalar(x)
        try:
            return float(x)
        except Exception:
            return float(default)

    def _pil_to_tensor(self, img: Image.Image) -> torch.Tensor:
        arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(arr)  # H,W,3

    def _sanitize_component(self, s: str) -> str:
        if s is None:
            return ""
        s = str(s).strip()
        out = ''.join(c for c in s if c not in INVALID_FS_CHARS)
        return __import__("re").sub(r"\s+", " ", out).strip()

    def _tensor_to_pil(self, t: torch.Tensor) -> Image.Image:
        # 1) Squeeze batch dim if present
        if isinstance(t, torch.Tensor) and t.ndim == 4:
            # If this is a batched tensor, take the first image;
            # for (1,H,W,C) this simply squeezes.
            t = t[0]

        if not isinstance(t, torch.Tensor):
            raise ValueError(f"Unsupported image type: {type(t)}")

        # 2) Move to CPU and scale to uint8
        if t.dtype in (torch.float16, torch.float32, torch.float64):
            t = t.clamp(0, 1)
            arr = (t.detach().cpu().numpy() * 255.0).astype(np.uint8)
        else:
            # assume already in [0,255]
            arr = t.detach().clamp(0, 255).byte().cpu().numpy()

        # 3) Normalize channel order/shape to (H,W,3)
        # Cases:
        #   (H,W)              -> grayscale
        #   (H,W,1)           -> grayscale
        #   (H,W,3|4)         -> channels-last
        #   (1|3|4,H,W)       -> channels-first
        if arr.ndim == 2:
            # (H,W) grayscale -> RGB
            return Image.fromarray(arr, mode="L").convert("RGB")

        if arr.ndim != 3:
            raise ValueError(f"Unsupported image tensor shape for saving: {arr.shape}")

        H, W, C_last = arr.shape
        C_first = arr.shape[0]

        # channels-last common cases
        if C_last in (1, 3, 4):
            if C_last == 1:
                # (H,W,1) -> (H,W,3)
                arr3 = np.repeat(arr, 3, axis=2)
                return Image.fromarray(arr3, mode="RGB")
            # (H,W,3|4) -> RGB (drop alpha if present)
            return Image.fromarray(arr[..., :3], mode="RGB")

        # channels-first common cases: (3,H,W) or (4,H,W) or (1,H,W)
        if C_first in (1, 3, 4) and arr.shape[1] == W and arr.shape[2] != C_last:
            # We likely have (C,H,W): transpose to (H,W,C)
            arr = np.transpose(arr, (1, 2, 0))
            # Now handle as channels-last
            if arr.shape[2] == 1:
                arr3 = np.repeat(arr, 3, axis=2)
                return Image.fromarray(arr3, mode="RGB")
            return Image.fromarray(arr[..., :3], mode="RGB")

        # If we reach here, shape is unexpected
        raise ValueError(f"Unsupported image tensor shape for saving: {arr.shape}")

    def _get_comfyui_version(self) -> str:
        """Return short git commit for ComfyUI if available; else empty string."""
        try:
            root = os.path.abspath(os.path.join(os.path.dirname(folder_paths.__file__), '..'))
            head_path = os.path.join(root, '.git', 'HEAD')
            if os.path.exists(head_path):
                with open(head_path, 'r', encoding='utf-8') as f:
                    head = f.read().strip()
                if head.startswith('ref: '):
                    ref = head[5:].strip()
                    ref_path = os.path.join(root, '.git', ref.replace('/', os.sep))
                    if os.path.exists(ref_path):
                        with open(ref_path, 'r', encoding='utf-8') as f:
                            commit = f.read().strip()
                            return commit[:7]
                return head[:7]
        except Exception:
            pass
        return ""

    def _build_metadata(self,
                        width: int,
                        height: int,
                        scale: float,
                        denoise: float,
                        guidance: float,
                        clip_skip: int,
                        steps: int,
                        seed_value: int,
                        sampler_name: str,
                        scheduler_name: str,
                        positive: str,
                        negative: str,
                        model_name: str,
                        loras: List[Tuple[str, float]],
                        extra_pnginfo: Dict[str, Any],
                        prompt: Dict[str, Any]) -> Dict[str, Any]:
        meta: Dict[str, Any] = {
            "width": int(width),
            "height": int(height),
            "scale": float(scale),
            "denoise": float(denoise),
            "guidance": float(guidance),
            "clip_skip": int(clip_skip),
            "steps": int(steps),
            "seed_value": int(seed_value),
            "sampler_name": sampler_name,
            "scheduler_name": scheduler_name,
            "model_name": model_name,
        }
        # Include prompts only if non-empty (avoid "Negative prompt:" swallowing params)
        if str(positive or "").strip() != "":
            meta["positive"] = positive
        if str(negative or "").strip() != "":
            meta["negative"] = negative

        clean_loras = [(n, float(w)) for (n, w) in loras if str(n).strip() != ""]
        if clean_loras:
            meta["loras"] = [{"name": n, "weight": w} for (n, w) in clean_loras]
        # App info
        v = self._get_comfyui_version()
        meta["app"] = {"name": "ComfyUI", **({"version": v} if v else {})}
        meta["extra_pnginfo"] = extra_pnginfo if isinstance(extra_pnginfo, dict) else (extra_pnginfo or {})
        meta["prompt"] = prompt if isinstance(prompt, dict) else (prompt or {})
        return meta

    def _fmt_f(self, x) -> str:
        try:
            s = f"{float(x):.3f}"
            s = s.rstrip('0').rstrip('.')
            return s if s else "0"
        except Exception:
            return str(x)

    def _build_a111_parameters(self, m: Dict[str, Any]) -> str:
        pos = (m.get("positive") or "")
        neg = (m.get("negative") or "")
        steps = int(m.get("steps") or 0)
        sampler = (m.get("sampler_name") or m.get("sampler") or "").strip()
        scheduler = (m.get("scheduler_name") or m.get("scheduler") or "").strip()
        guidance = m.get("guidance")
        scale = m.get("scale")
        denoise = m.get("denoise")
        seed = m.get("seed_value")
        w = m.get("width")
        h = m.get("height")
        clip_skip = int(m.get("clip_skip") or 0)
        model = (m.get("model_name") or "").strip()

        parts = []

        # Positive line: include only if non-empty.
        if pos.strip() != "":
            parts.append(pos)

        # Negative line: include only if non-empty.
        if neg.strip() != "":
            parts.append(f"Negative prompt: {neg}")

        # Parameters line (always present)
        tail = [
            f"Steps: {steps}",
            (f"Sampler: {sampler}" if sampler else None),
            (f"Scheduler: {scheduler}" if scheduler else None),
            (f"Guidance: {self._fmt_f(guidance)}" if guidance is not None else None),
            (f"Scale: {self._fmt_f(scale)}" if scale is not None else None),
            (f"Denoise: {self._fmt_f(denoise)}" if denoise is not None else None),
            (f"Seed: {seed}" if seed is not None else None),
            (f"Size: {w}x{h}" if w and h else None),
            ((f"Clip skip: {abs(clip_skip)}") if clip_skip != 0 else None),
            ((f"Model: {model}") if model else None),
        ]
        parts.append(", ".join([t for t in tail if t]))

        # Add ComfyUI version right before LoRA section
        ver = (m.get("app") or {}).get("version") or self._get_comfyui_version()
        parts.append(f"Version: ComfyUI{(' ' + ver) if ver else ''}")

        # LoRA lines in requested format
        loras = m.get("loras") or []
        for idx, d in enumerate(loras, start=1):
            name = (d.get('name') or '').strip()
            if not name:
                continue
            weight = d.get('weight')
            try:
                weight = float(weight)
            except Exception:
                weight = 0.0
            parts.append(f"LoRA_{idx}: {name}, LoRA_{idx}_Weight: {self._fmt_f(weight)}")
        return "\n".join(parts)

    def _embed_pnginfo(self, metadata):
        pnginfo = PngImagePlugin.PngInfo()
        try:
            a111 = self._build_a111_parameters(metadata)
            pnginfo.add_text("parameters", a111)

            # Always embed 'prompt' (stringify even if list/dict)
            if "prompt" in metadata:
                pnginfo.add_text("prompt", json.dumps(metadata["prompt"]))

            # Robust workflow extraction
            wf = self._extract_workflow(metadata.get("extra_pnginfo"))
            if wf is not None:
                pnginfo.add_text("workflow", json.dumps(wf))

            if "app" in metadata:
                pnginfo.add_text("app", json.dumps(metadata["app"]))
        except Exception as e:
            print(f"[FRED_Image_Saver] PNGInfo embed failed: {e}")
        return pnginfo

    def _extract_workflow(self, extra_pnginfo):
        # extra_pnginfo can be a dict (core Comfy) or a list of dicts (some front-ends)
        def _maybe_json(x):
            if isinstance(x, str):
                try:
                    return json.loads(x)
                except Exception:
                    return x
            return x

        if isinstance(extra_pnginfo, dict):
            return _maybe_json(extra_pnginfo.get("workflow"))

        if isinstance(extra_pnginfo, list):
            for item in extra_pnginfo:
                if isinstance(item, dict) and "workflow" in item:
                    return _maybe_json(item["workflow"])
        return None

    def _expand_tokens(self, s: str, meta: Dict[str, Any], time_format: str) -> str:
        if not s:
            return ""
        s = os.path.expandvars(s)

        token_map: Dict[str, str] = {}
        for k, v in meta.items():
            if isinstance(v, (str, int, float)):
                token_map[k] = str(v)

        model_nm = token_map.get("model_name", "")
        token_map["basemodelname"] = os.path.splitext(model_nm)[0] if model_nm else ""

        # IMPORTANT : toujours injecter les tokens de date
        token_map["date"] = time.strftime("%Y%m%d")
        token_map["date_dash"] = time.strftime("%Y-%m-%d")
        token_map["time"] = time.strftime("%H%M%S")
        token_map["datetime"] = time.strftime(time_format or "%Y-%m-%d-%H%M%S")

        alias_map = {
            "model": "model_name",
            "sampler": "sampler_name",
            "scheduler": "scheduler_name",
            "seed": "seed_value",
            "cfg": "guidance"
        }
        for a, b in alias_map.items():
            if a not in token_map and b in token_map:
                token_map[a] = token_map[b]

        # LoRA token filtering: include weight_N only if name_N is set
        for i in (1, 2, 3):
            name_k = f"lora_name_{i}"
            weight_k = f"lora_weight_{i}"
            name_v = str(meta.get(name_k, "") or "").strip()
            if name_v == "":
                token_map.pop(weight_k, None)
            else:
                token_map[name_k] = name_v
                token_map[weight_k] = str(meta.get(weight_k, ""))

        # Replace tokens (supports %token and %token%)
        for key in sorted(token_map.keys(), key=len, reverse=True):
            val = self._sanitize_component(token_map[key])
            for tok in (f"%{key}%", f"%{key}"):
                if tok in s:
                    s = s.replace(tok, val)

        # Supprime les tokens non reconnus
        s = __import__("re").sub(r"%[A-Za-z0-9_]+%?", "", s)

        return s
    # def _expand_tokens(self, s: str, meta: Dict[str, Any], time_format: str) -> str:
        # if not s:
            # return ""
        # s = os.path.expandvars(s)

        # token_map: Dict[str, str] = {}
        # for k, v in meta.items():
            # if isinstance(v, (str, int, float)):
                # token_map[k] = str(v)
        # model_nm = token_map.get("model_name", "")
        # token_map["basemodelname"] = os.path.splitext(model_nm)[0] if model_nm else ""
        # token_map["date"] = time.strftime("%Y%m%d")
        # token_map["date_dash"] = time.strftime("%Y-%m-%d")
        # token_map["time"] = time.strftime("%H%M%S")
        # token_map["datetime"] = time.strftime(time_format or "%Y-%m-%d-%H%M%S")
        # alias_map = {"model": "model_name", "sampler": "sampler_name", "scheduler": "scheduler_name", "seed": "seed_value", "cfg": "guidance"}
        # for a, b in alias_map.items():
            # if a not in token_map and b in token_map:
                # token_map[a] = token_map[b]

        # # LoRA token filtering: include weight_N only if name_N is set
        # for i in (1, 2, 3):
            # name_k = f"lora_name_{i}"
            # weight_k = f"lora_weight_{i}"
            # name_v = str(meta.get(name_k, "") or "").strip()
            # if name_v == "":
                # token_map.pop(weight_k, None)
            # else:
                # token_map[name_k] = name_v
                # token_map[weight_k] = str(meta.get(weight_k, ""))

        # # Replace tokens (supports %token and %token%)
        # for key in sorted(token_map.keys(), key=len, reverse=True):
            # val = self._sanitize_component(token_map[key])
            # for tok in (f"%{key}%", f"%{key}"):
                # if tok in s:
                    # s = s.replace(tok, val)
        # s = __import__("re").sub(r"%[A-Za-z0-9_]+%?", "", s)
        # return s

    def _ensure_outdir(self, subdir: str, meta: Dict[str, Any], time_format: str) -> str:
        base = folder_paths.get_output_directory()
        subdir = self._expand_tokens(subdir or "", meta, time_format)
        if subdir and os.path.isabs(subdir):
            parts = []
            drive, tail = os.path.splitdrive(subdir)
            for part in tail.replace("\\", "/").split("/"):
                if part:
                    parts.append(self._sanitize_component(part))
            outdir = os.path.join(drive + os.sep if drive else os.sep, *parts) if parts else subdir
        else:
            parts = []
            for part in subdir.replace("\\", "/").split("/"):
                if part:
                    parts.append(self._sanitize_component(part))
            outdir = os.path.join(base, *parts) if parts else base
        outdir = os.path.normpath(outdir)
        os.makedirs(outdir, exist_ok=True)
        return outdir

    def _compute_grid_dims(self, n: int, col_max: int, row_max: int) -> Tuple[int, int]:
        """
        Choose rows/cols respecting the provided maxima (x_size = columns, y_size = rows),
        while avoiding blank tiles when n < col_max*row_max.
        """
        if n <= 0:
            return 1, 1
        cols = min(col_max, n)
        rows = max(1, math.ceil(n / cols))
        rows = min(rows, row_max)

        # Ensure capacity >= n; expand within limits.
        while rows * cols < n and rows < row_max:
            rows += 1
        while rows * cols < n and cols < col_max:
            cols += 1

        rows = max(1, min(rows, row_max))
        cols = max(1, min(cols, col_max))
        return rows, cols  # (rows, cols)

    def _assemble_grid(self, images_pil: List[Image.Image], rows: int, cols: int) -> Image.Image:
        n = len(images_pil)
        if n == 0:
            raise ValueError("No images to grid")
        w, h = images_pil[0].size
        grid = Image.new("RGB", (cols * w, rows * h), color="#ffffff")
        idx = 0
        for r in range(rows):
            for c in range(cols):
                if idx >= n:
                    break
                grid.paste(images_pil[idx], (c * w, r * h))
                idx += 1
        return grid

    # ------------------------------ main ------------------------------
    def save(self,
             images,
             save_single_image: bool,
             filename: str,
             path: str,
             time_format: str,
             extension: str,
             optimize_png: bool,
             lossless_webp: bool,
             quality_jpeg_or_webp: int,
             save_workflow_as_json: bool,
             embed_workflow_in_png: bool,
             width: int,
             height: int,
             scale: float,
             denoise: float,
             guidance: float,
             clip_skip: int,
             steps: int,
             seed_value: int,
             sampler_name: str,
             scheduler_name: str,
             positive: str,
             negative: str,
             model_name: str,
             lora_name_1: str,
             lora_weight_1: float,
             lora_name_2: str,
             lora_weight_2: float,
             lora_name_3: str,
             lora_weight_3: float,
             save_as_grid_if_multi: bool,
             grid_filename: str,
             grid_path: str,
             grid_column_max: int,
             grid_row_max: int,
             grid_extension: str,
             grid_quality_jpeg_or_webp: int,
             prompt: Dict[str, Any] = None,
             extra_pnginfo: Dict[str, Any] = None,
             unique_id: str = ""):

        # --- Coercer tout ce qui pourrait arriver en liste (widgets / autres nodes) ---
        # ---- COERCE scalar/numeric inputs (avoid list types) ----
        filename      = self._scalar(filename)
        path          = self._scalar(path)
        time_format   = self._scalar(time_format)
        extension     = self._scalar(extension)
        optimize_png  = bool(self._scalar(optimize_png))
        lossless_webp = bool(self._scalar(lossless_webp))
        quality_jpeg_or_webp = self._to_int(quality_jpeg_or_webp, 100)
        save_workflow_as_json = bool(self._scalar(save_workflow_as_json))
        embed_workflow_in_png = bool(self._scalar(embed_workflow_in_png))

        save_as_grid_if_multi = bool(self._scalar(save_as_grid_if_multi))

        grid_filename = self._scalar(grid_filename)
        grid_path     = self._scalar(grid_path)
        grid_extension = self._scalar(grid_extension)
        grid_column_max = self._to_int(grid_column_max, 5)
        grid_row_max    = self._to_int(grid_row_max, 1)
        grid_quality_jpeg_or_webp = self._to_int(grid_quality_jpeg_or_webp, 100)

        # prompts / meta numeric
        positive = self._scalar(positive)
        negative = self._scalar(negative)
        sampler_name   = self._scalar(sampler_name)
        scheduler_name = self._scalar(scheduler_name)
        model_name     = self._scalar(model_name)

        width      = self._to_int(width, 1024)
        height     = self._to_int(height, 1024)
        scale      = self._to_float(scale, 1.40)
        denoise    = self._to_float(denoise, 0.60)
        guidance   = self._to_float(guidance, 2.20)
        clip_skip  = self._to_int(clip_skip, 1)
        steps      = self._to_int(steps, 20)
        seed_value = self._to_int(seed_value, 0)

        lora_name_1 = self._scalar(lora_name_1); lora_weight_1 = self._to_float(lora_weight_1, 1.0)
        lora_name_2 = self._scalar(lora_name_2); lora_weight_2 = self._to_float(lora_weight_2, 1.0)
        lora_name_3 = self._scalar(lora_name_3); lora_weight_3 = self._to_float(lora_weight_3, 1.0)

        # --- Normalize input: accept list/tuple OR a torch batch tensor ---
        img_list: List[torch.Tensor] = []
        if isinstance(images, (list, tuple)):
            input_kind = "list"
            for im in images:
                if not isinstance(im, torch.Tensor):
                    raise ValueError(f"[FRED_Image_Saver] Unsupported list element type: {type(im)}")
                if im.ndim == 4:  # (B,H,W,C) batch FOUND inside a list â†’ flatten it
                    b_in = im.shape[0]
                    for i in range(b_in):
                        img_list.append(im[i])
                elif im.ndim == 3:  # (H,W,C)
                    img_list.append(im)
                else:
                    raise ValueError(f"[FRED_Image_Saver] Unsupported tensor shape in list: {tuple(im.shape)}")
        elif isinstance(images, torch.Tensor):
            input_kind = "tensor"
            if images.ndim == 4:  # (B,H,W,C)
                for i in range(images.shape[0]):
                    img_list.append(images[i])
            elif images.ndim == 3:  # (H,W,C) single
                img_list.append(images)
            else:
                raise ValueError(f"[FRED_Image_Saver] Unsupported tensor shape: {tuple(images.shape)}")
        else:
            raise ValueError(f"[FRED_Image_Saver] Unsupported images input type: {type(images)}")

        b = len(img_list)
        is_multi_input = b > 1
        first_t = img_list[0]
        in_dev = first_t.device
        in_dt = first_t.dtype

        shapes = [tuple(t.shape) for t in img_list]
        print(f"[FRED_Image_Saver] input_kind={input_kind}, count={b}, is_multi_input={is_multi_input}, shapes={shapes[:5]}{'...' if len(shapes)>5 else ''}")

        # --- metadata base (shared) ---
        metadata = self._build_metadata(
            width=width, height=height, scale=scale, denoise=denoise, guidance=guidance,
            clip_skip=clip_skip, steps=steps, seed_value=seed_value,
            sampler_name=sampler_name, scheduler_name=scheduler_name,
            positive=positive, negative=negative, model_name=model_name,
            loras=[(lora_name_1, lora_weight_1), (lora_name_2, lora_weight_2), (lora_name_3, lora_weight_3)],
            extra_pnginfo=extra_pnginfo or {}, prompt=prompt or {},
        )
        # Expose LoRA fields for token expansion (weights only if name non-empty)
        if (lora_name_1 or "").strip():
            metadata["lora_name_1"] = lora_name_1
            metadata["lora_weight_1"] = lora_weight_1
        if (lora_name_2 or "").strip():
            metadata["lora_name_2"] = lora_name_2
            metadata["lora_weight_2"] = lora_weight_2
        if (lora_name_3 or "").strip():
            metadata["lora_name_3"] = lora_name_3
            metadata["lora_weight_3"] = lora_weight_3

        saved_paths: List[str] = []

        # --- Single images (optionnel) ---
        if save_single_image:
            outdir = self._ensure_outdir(path, metadata, time_format)
            base_template = filename.strip() if filename else "image"
            base = self._expand_tokens(base_template, metadata, time_format)
            base = self._sanitize_component(base) or "image"

            for t in img_list:
                save_dir, filename_base, counter, subfolder, _ = folder_paths.get_save_image_path(base, outdir)
                name = f"{filename_base}_{counter:05d}"
                full = os.path.join(save_dir, f"{name}.{extension}")
                while os.path.exists(full):
                    counter += 1
                    name = f"{filename_base}_{counter:05d}"
                    full = os.path.join(save_dir, f"{name}.{extension}")

                img = self._tensor_to_pil(t)

                pnginfo = None
                if extension.lower() == "png" and embed_workflow_in_png:
                    pnginfo = self._embed_pnginfo(metadata)

                try:
                    if extension.lower() == 'png':
                        params = {"optimize": bool(optimize_png)}
                        if pnginfo is not None:
                            params["pnginfo"] = pnginfo
                        img.save(full, format='PNG', **params)
                    elif extension.lower() == 'jpeg':
                        img.save(full, format='JPEG', quality=int(quality_jpeg_or_webp), optimize=True)
                    elif extension.lower() == 'webp':
                        if lossless_webp:
                            img.save(full, format='WEBP', lossless=True)
                        else:
                            img.save(full, format='WEBP', quality=int(quality_jpeg_or_webp))
                    else:
                        comfy.utils.save_image(img, save_dir, name)
                except Exception as e:
                    print(f"[FRED_Image_Saver] Save error: {e}. Falling back to Comfy save_image.")
                    comfy.utils.save_image(img, save_dir, name)

                if save_workflow_as_json:
                    try:
                        sidecar = os.path.join(save_dir, f"{name}.json")
                        wf = self._extract_workflow(metadata.get("extra_pnginfo"))
                        if wf is None:
                            # optional: fall back to UI-export-style workflow if you ever stash it elsewhere
                            print("[FRED_Image_Saver] No workflow found in extra_pnginfo; skipping sidecar.")
                        else:
                            with open(sidecar, "w", encoding="utf-8") as f:
                                json.dump(wf, f, ensure_ascii=False, indent=2)
                    except Exception as e:
                        print(f"[FRED_Image_Saver] Could not write JSON sidecar: {e}")

                saved_paths.append(full)

        # --- GRID (une seule fois, liste ou batch) ---
        imgs_pil = [self._tensor_to_pil(t) for t in img_list]
        rows, cols = self._compute_grid_dims(len(imgs_pil), grid_column_max, grid_row_max)
        capacity = rows * cols
        if len(imgs_pil) > capacity:
            print(f"[FRED_Image_Saver] Grid capacity {capacity} (rows={rows}, cols={cols}) cannot fit all {len(imgs_pil)} images. {len(imgs_pil) - capacity} left over.")
            imgs_pil = imgs_pil[:capacity]

        grid_img = self._assemble_grid(imgs_pil, rows, cols)

        gi_np = np.asarray(grid_img, dtype=np.uint8)
        gi_t  = torch.from_numpy(gi_np).to(in_dev).float() / 255.0
        grid_image_out = gi_t.unsqueeze(0).to(dtype=in_dt)

        if save_as_grid_if_multi and len(imgs_pil) > 1:
            meta_grid = dict(metadata)
            meta_grid["img_count"] = len(imgs_pil)
            meta_grid["grid_rows"] = rows
            meta_grid["grid_cols"] = cols

            outdir_grid = self._ensure_outdir(grid_path, meta_grid, time_format)
            base_grid_t = grid_filename.strip() if grid_filename else "grid"
            base_grid = self._expand_tokens(base_grid_t, meta_grid, time_format)
            base_grid = self._sanitize_component(base_grid) or "grid"

            save_dir, filename_base, counter, subfolder, _ = folder_paths.get_save_image_path(
                base_grid, outdir_grid, grid_img.width, grid_img.height
            )
            name = f"{filename_base}_{counter:05d}"
            full = os.path.join(save_dir, f"{name}.{grid_extension}")
            while os.path.exists(full):
                counter += 1
                name = f"{filename_base}_{counter:05d}"
                full = os.path.join(save_dir, f"{name}.{grid_extension}")

            pnginfo_grid = None
            if grid_extension.lower() == "png" and embed_workflow_in_png:
                pnginfo_grid = self._embed_pnginfo(meta_grid)

            try:
                if grid_extension.lower() == 'png':
                    params = {"optimize": bool(optimize_png)}
                    if pnginfo_grid is not None:
                        params["pnginfo"] = pnginfo_grid
                    grid_img.save(full, format='PNG', **params)
                elif grid_extension.lower() == 'jpeg':
                    grid_img.save(full, format='JPEG', quality=int(grid_quality_jpeg_or_webp), optimize=True)
                elif grid_extension.lower() == 'webp':
                    if lossless_webp:
                        grid_img.save(full, format='WEBP', lossless=True)
                    else:
                        grid_img.save(full, format='WEBP', quality=int(grid_quality_jpeg_or_webp))
                else:
                    comfy.utils.save_image(grid_img, save_dir, name)
            except Exception as e:
                print(f"[FRED_Image_Saver] Grid save error: {e}. Falling back to Comfy save_image.")
                comfy.utils.save_image(grid_img, save_dir, name)

            if save_workflow_as_json:
                try:
                    sidecar = os.path.join(save_dir, f"{name}.json")
                    with open(sidecar, "w", encoding="utf-8") as f:
                        json.dump(meta_grid, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    print(f"[FRED_Image_Saver] Could not write GRID JSON sidecar: {e}")

            saved_paths.append(full)

        # Retour
        return (grid_image_out, saved_paths[-1] if saved_paths else "", HELP_MESSAGE)

    @staticmethod
    def IS_CHANGED(*args, **kwargs):
        # Ignore data-like inputs so accumulation doesn't retrigger saves unnecessarily
        drop = {"images", "prompt", "extra_pnginfo", "unique_id"}
        clean = {k: v for k, v in kwargs.items() if k not in drop}
        return json.dumps(clean, sort_keys=True, ensure_ascii=False)


NODE_CLASS_MAPPINGS = {
    "FRED_ImageSaver": FRED_ImageSaver,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FRED_ImageSaver": "👑 FRED Image Saver",
}

# # -*- coding: utf-8 -*-
# from __future__ import annotations

# import os
# import json
# import time
# import math
# from typing import Any, Dict, List, Tuple

# import numpy as np
# import torch
# from PIL import Image, PngImagePlugin

# # ComfyUI core helpers
# import folder_paths
# import comfy.utils
# from comfy.comfy_types.node_typing import ComfyNodeABC, InputTypeDict, IO

# print(hasattr(comfy.utils, 'save_image'))

# HELP_MESSAGE = """
# 👑 FRED_Image_Saver

# 🔹 PURPOSE
# Save images flexibly with dynamic filename/path tokens, Automatic1111-style metadata embedding, and optional grid saving.
# Accepts **list or batch** as input (mapping disabled on images), so the node runs **once** for a whole set.

# 📥 INPUTS
# - images • one or more images to save (**list or batch accepted; mapping disabled**).
  # • If a list contains a batch (e.g. [ (B,H,W,3) ]), it is flattened to B images.
  # • All images must share the same H×W.
# - save_single_image • if ON, save each input image as its own file (in addition to the grid, if enabled).
# - filename / path • templates with tokens for naming (a counter is always appended for uniqueness).
# - time_format • used by %datetime.
# - extension / quality_jpeg_or_webp / optimize_png / lossless_webp • single-image format & compression.
# - save_workflow_as_json • writes a sidecar JSON file (metadata).
# - embed_workflow_in_png • embeds workflow/prompt/app metadata into PNG tEXt.

# - metadata fields (record-only; pixels are not changed):
  # width, height, scale, denoise, guidance, clip_skip, steps, seed_value,
  # sampler_name, scheduler_name, positive, negative, model_name
  # • Positive/Negative are written **only if non-empty** (prevents “Negative prompt:” swallowing params).

# - lora_name_X / lora_weight_X (X=1..3) • up to 3 LoRAs. Weights are included **only if the name is set**.

# - Grid options:
  # • save_as_grid_if_multi • if ON and there are ≥2 images, also save a tiled grid image.
  # • grid_column_max / grid_row_max • maximum grid size (auto layout chooses rows/cols ≤ these maxima).
  # • grid_filename / grid_path / grid_extension / grid_quality_jpeg_or_webp • grid output settings.

# ⚙️ KEY BEHAVIOR
# - Single image saves:
  # • Saved under “path” (relative to Comfy output unless absolute).
  # • Uses tokenized filename; unique counter `_00001`, `_00002`, … appended.
  # • PNG can embed JSON + optimize (lossless). JPEG/WEBP use quality; WEBP can be lossless.

# - Grid saves:
  # • A grid image is **returned** as the first output (tensor `(1,H,W,3)`) for preview/flow.
  # • A grid **file** is **saved only if** `save_as_grid_if_multi = ON` **and** there are ≥2 images.
  # • Auto layout respects `grid_column_max` / `grid_row_max`. If not all images fit, extras are ignored with a log.
  # • Grid metadata adds: `img_count`, `grid_rows`, `grid_cols`.
  # • If there is only 1 image, the returned tensor is a 1×1 grid; no grid file is saved.

# 🧾 METADATA (PNG tEXt “parameters” like Automatic1111)
# - “parameters” line(s):
  # • Positive prompt line (if non-empty)
  # • “Negative prompt: …” (if non-empty)
  # • “Steps, Sampler, Scheduler, Guidance, Scale, Denoise, Seed, Size, Clip skip, Model”
  # • “Version: ComfyUI <short-git>”
  # • One line per LoRA: `LoRA_X: <name>, LoRA_X_Weight: <w>`
# - Also embeds “prompt”, “workflow” (if present), and “app” details.

# 🔑 TOKENS (usable in filename and path)
# - Time/date: `%date`, `%date_dash`, `%time`, `%datetime`
# - Params: `%seed`/`%seed_value`, `%model`/`%model_name`/`%basemodelname`, `%sampler`/`%sampler_name`,
           # `%scheduler`/`%scheduler_name`, `%width`, `%height`, `%steps`, `%cfg`/`%guidance`,
           # `%scale`, `%denoise`, `%clip_skip`
# - Batch info: `%img_count` (grid/batch size)
# - LoRAs: `%lora_name_1..3`, `%lora_weight_1..3` (weight expands only if corresponding name is set)

# 📤 OUTPUTS
# - GRID_IMAGE • the grid as `(1,H,W,3)` (or 1×1 if single input).
# - last_saved_path • path of the **last** saved file (single image or grid). Empty if nothing saved.
# - help • this message.

# 📝 NOTES & TIPS
# - This node does **not** resize; width/height/steps/etc. are **recorded only**.
# - `%basemodelname` expands to `model_name` without extension.
# - PNG `optimize=True` reduces size but is slower. WEBP `lossless_webp=True` ignores quality.
# - Unique filenames are guaranteed (Comfy counter + safety loop).
# - **Mapping is disabled on `images`**: the node runs once for a whole list/batch.  
  # If other inputs arrive as lists (from widgets), they are coerced to their **first** value to avoid remapping.
# - Set `save_single_image=False` if you only want the grid file.
# - Mixed sizes? Make sure all inputs share the same H×W before saving a grid.

# 📚 EXAMPLES
# - filename:      "%basemodelname_%datetime_seed_%seed"
# - path:          "Fred_nodes/%date_dash/"
# - grid_filename: "%model_name_%date_%time_grid_%img_count"
# - grid_path:     "Test/%model_name/Grid/%date_dash/"
# """

# INVALID_FS_CHARS = '<>:"/\\|?*'

# class FRED_ImageSaver(ComfyNodeABC):
    # # Dites à Comfy: j'accepte les listes sur les inputs → ne mappe pas le node sur 'images'
    # INPUT_IS_LIST = True
    # # On ne retourne pas des listes (3 sorties scalaires)
    # OUTPUT_IS_LIST = (False, False, False)

    # @classmethod
    # def INPUT_TYPES(cls) -> InputTypeDict:
        # return {
            # "required": {
                # # --- IMAGE & FILE OUTPUT ---
                # "images": ("IMAGE", {"tooltip": "image(s) to save (list or batch)."}),
                # "save_single_image": ("BOOLEAN", {"default": True, "tooltip": "Save each input image as its own file."}),
                # "filename": (IO.STRING, {"default": "%model_name_%date_dash", "multiline": False, "tooltip": "base filename (counter appended)"}),
                # "path": (IO.STRING, {"default": "Fred_nodes/%date_dash/", "multiline": False, "tooltip": "relative to Comfy output (or absolute)"}),
                # "time_format": (IO.STRING, {"default": "%Y-%m-%d-%H%M%S", "tooltip": "used by %datetime token"}),
                # "extension": (["png", "jpeg", "webp"], {"default": "png", "tooltip": "output file format for single images"}),
                # "optimize_png": ("BOOLEAN", {"default": False, "tooltip": "optimize PNG (lossless); smaller files but slower"}),
                # "lossless_webp": ("BOOLEAN", {"default": True, "tooltip": "save WEBP in lossless mode (quality ignored)"}),
                # "quality_jpeg_or_webp": (IO.INT, {"default": 100, "min": 1, "max": 100, "tooltip": "JPEG/WEBP quality (ignored if WEBP lossless)"}),

                # "save_workflow_as_json": ("BOOLEAN", {"default": False, "tooltip": "write a sidecar JSON next to image"}),
                # "embed_workflow_in_png": ("BOOLEAN", {"default": True, "tooltip": "embed workflow/prompt JSON into PNG metadata"}),

                # # --- CORE PARAMS (recording only; does not modify pixels) ---
                # "width": (IO.INT, {"default": 1024, "min": 1, "max": 16384, "tooltip": "recorded in metadata; does not resize"}),
                # "height": (IO.INT, {"default": 1024, "min": 1, "max": 16384, "tooltip": "recorded in metadata; does not resize"}),
                # "scale": (IO.FLOAT, {"default": 1.40, "min": 0.0, "max": 100000.0, "step": 0.01, "tooltip": "record-only parameter"}),
                # "denoise": (IO.FLOAT, {"default": 0.60, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "record-only parameter"}),
                # "guidance": (IO.FLOAT, {"default": 2.20, "min": 0.0, "max": 100.0, "step": 0.01, "tooltip": "record-only parameter (CFG/guidance)"}),
                # "clip_skip": (IO.INT, {"default": 1, "min": -24, "max": 24, "step": 1, "tooltip": "record-only parameter"}),
                # "steps": (IO.INT, {"default": 20, "min": 1, "max": 4096, "tooltip": "record-only parameter"}),
                # "seed_value": (IO.INT, {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "record-only parameter (seed)"}),
                # "sampler_name": (IO.STRING, {"default": "euler", "multiline": False, "tooltip": "record-only parameter (sampler name)"}),
                # "scheduler_name": (IO.STRING, {"default": "simple", "multiline": False, "tooltip": "record-only parameter (scheduler name)"}),
                # "positive": (IO.STRING, {"default": "", "multiline": True, "tooltip": "positive prompt (saved in metadata)"}),
                # "negative": (IO.STRING, {"default": "", "multiline": True, "tooltip": "negative prompt (saved in metadata)"}),
                # "model_name": (IO.STRING, {"default": "flux", "tooltip": "record-only parameter (model name)"}),

                # # --- OPTIONAL LORAs ---
                # "lora_name_1": (IO.STRING, {"default": "", "tooltip": "optional LoRA name #1 (if empty: weight omitted everywhere)"}),
                # "lora_weight_1": (IO.FLOAT, {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01, "tooltip": "LoRA weight #1"}),
                # "lora_name_2": (IO.STRING, {"default": "", "tooltip": "optional LoRA name #2 (if empty: weight omitted everywhere)"}),
                # "lora_weight_2": (IO.FLOAT, {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01, "tooltip": "LoRA weight #2"}),
                # "lora_name_3": (IO.STRING, {"default": "", "tooltip": "optional LoRA name #3 (if empty: weight omitted everywhere)"}),
                # "lora_weight_3": (IO.FLOAT, {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01, "tooltip": "LoRA weight #3"}),

                # # --- GRID SAVE OPTIONS ---
                # "save_as_grid_if_multi": ("BOOLEAN", {"default": False, "tooltip": "Also save a single tiled grid image of all inputs."}),
                # "grid_filename": (IO.STRING, {"default": "%lora_name_1_weight_%lora_weight_1_%datetime_%model_name grid_%img_count", "multiline": False, "tooltip": "filename prefix for the grid image"}),
                # "grid_path": (IO.STRING, {"default": "Test/%model_name/Grid/%date_dash/", "multiline": False, "tooltip": "save folder for the grid image"}),
                # "grid_column_max": (IO.INT, {"default": 5, "min": 1, "step": 1, "tooltip": "x_size (columns)."}),
                # "grid_row_max": (IO.INT, {"default": 1, "min": 1, "step": 1, "tooltip": "y_size (rows)."}),
                # "grid_extension": (["png", "jpeg", "webp"], {"default": "png", "tooltip": "output file format for the grid image"}),
                # "grid_quality_jpeg_or_webp": (IO.INT, {"default": 100, "min": 1, "max": 100, "tooltip": "JPEG/WEBP quality for the grid image"}),
            # },
            # "hidden": {
                # "prompt": "PROMPT",
                # "extra_pnginfo": "EXTRA_PNGINFO",
                # "unique_id": "UNIQUE_ID",
            # },
        # }

    # RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    # RETURN_NAMES = ("GRID_IMAGE", "last_saved_path", "help")
    # OUTPUT_TOOLTIPS = ("path to last saved image", "help / usage tips")
    # FUNCTION = "save"
    # CATEGORY = "👑FRED/image"
    # DESCRIPTION = "Save images with A1111-style parameters text and robust tokenized filenames/paths. Can also save a grid when there is multiple images."
    # OUTPUT_NODE = True

    # # ------------------------------ helpers ------------------------------
    # def _scalar(self, x):
        # return (x[0] if isinstance(x, (list, tuple)) and x else x)

    # def _to_int(self, x, default=0):
        # x = self._scalar(x)
        # try:
            # return int(x)
        # except Exception:
            # try:
                # return int(float(x))
            # except Exception:
                # return int(default)

    # def _to_float(self, x, default=0.0):
        # x = self._scalar(x)
        # try:
            # return float(x)
        # except Exception:
            # return float(default)

    # def _pil_to_tensor(self, img: Image.Image) -> torch.Tensor:
        # arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        # return torch.from_numpy(arr)  # H,W,3

    # def _sanitize_component(self, s: str) -> str:
        # if s is None:
            # return ""
        # s = str(s).strip()
        # out = ''.join(c for c in s if c not in INVALID_FS_CHARS)
        # return __import__("re").sub(r"\s+", " ", out).strip()

    # def _tensor_to_pil(self, t: torch.Tensor) -> Image.Image:
        # # 1) Squeeze batch dim if present
        # if isinstance(t, torch.Tensor) and t.ndim == 4:
            # # If this is a batched tensor, take the first image;
            # # for (1,H,W,C) this simply squeezes.
            # t = t[0]

        # if not isinstance(t, torch.Tensor):
            # raise ValueError(f"Unsupported image type: {type(t)}")

        # # 2) Move to CPU and scale to uint8
        # if t.dtype in (torch.float16, torch.float32, torch.float64):
            # t = t.clamp(0, 1)
            # arr = (t.detach().cpu().numpy() * 255.0).astype(np.uint8)
        # else:
            # # assume already in [0,255]
            # arr = t.detach().clamp(0, 255).byte().cpu().numpy()

        # # 3) Normalize channel order/shape to (H,W,3)
        # # Cases:
        # #   (H,W)              -> grayscale
        # #   (H,W,1)           -> grayscale
        # #   (H,W,3|4)         -> channels-last
        # #   (1|3|4,H,W)       -> channels-first
        # if arr.ndim == 2:
            # # (H,W) grayscale -> RGB
            # return Image.fromarray(arr, mode="L").convert("RGB")

        # if arr.ndim != 3:
            # raise ValueError(f"Unsupported image tensor shape for saving: {arr.shape}")

        # H, W, C_last = arr.shape
        # C_first = arr.shape[0]

        # # channels-last common cases
        # if C_last in (1, 3, 4):
            # if C_last == 1:
                # # (H,W,1) -> (H,W,3)
                # arr3 = np.repeat(arr, 3, axis=2)
                # return Image.fromarray(arr3, mode="RGB")
            # # (H,W,3|4) -> RGB (drop alpha if present)
            # return Image.fromarray(arr[..., :3], mode="RGB")

        # # channels-first common cases: (3,H,W) or (4,H,W) or (1,H,W)
        # if C_first in (1, 3, 4) and arr.shape[1] == W and arr.shape[2] != C_last:
            # # We likely have (C,H,W): transpose to (H,W,C)
            # arr = np.transpose(arr, (1, 2, 0))
            # # Now handle as channels-last
            # if arr.shape[2] == 1:
                # arr3 = np.repeat(arr, 3, axis=2)
                # return Image.fromarray(arr3, mode="RGB")
            # return Image.fromarray(arr[..., :3], mode="RGB")

        # # If we reach here, shape is unexpected
        # raise ValueError(f"Unsupported image tensor shape for saving: {arr.shape}")

    # def _get_comfyui_version(self) -> str:
        # """Return short git commit for ComfyUI if available; else empty string."""
        # try:
            # root = os.path.abspath(os.path.join(os.path.dirname(folder_paths.__file__), '..'))
            # head_path = os.path.join(root, '.git', 'HEAD')
            # if os.path.exists(head_path):
                # with open(head_path, 'r', encoding='utf-8') as f:
                    # head = f.read().strip()
                # if head.startswith('ref: '):
                    # ref = head[5:].strip()
                    # ref_path = os.path.join(root, '.git', ref.replace('/', os.sep))
                    # if os.path.exists(ref_path):
                        # with open(ref_path, 'r', encoding='utf-8') as f:
                            # commit = f.read().strip()
                            # return commit[:7]
                # return head[:7]
        # except Exception:
            # pass
        # return ""

    # def _build_metadata(self,
                        # width: int,
                        # height: int,
                        # scale: float,
                        # denoise: float,
                        # guidance: float,
                        # clip_skip: int,
                        # steps: int,
                        # seed_value: int,
                        # sampler_name: str,
                        # scheduler_name: str,
                        # positive: str,
                        # negative: str,
                        # model_name: str,
                        # loras: List[Tuple[str, float]],
                        # extra_pnginfo: Dict[str, Any],
                        # prompt: Dict[str, Any]) -> Dict[str, Any]:
        # meta: Dict[str, Any] = {
            # "width": int(width),
            # "height": int(height),
            # "scale": float(scale),
            # "denoise": float(denoise),
            # "guidance": float(guidance),
            # "clip_skip": int(clip_skip),
            # "steps": int(steps),
            # "seed_value": int(seed_value),
            # "sampler_name": sampler_name,
            # "scheduler_name": scheduler_name,
            # "model_name": model_name,
        # }
        # # Include prompts only if non-empty (avoid "Negative prompt:" swallowing params)
        # if str(positive or "").strip() != "":
            # meta["positive"] = positive
        # if str(negative or "").strip() != "":
            # meta["negative"] = negative

        # clean_loras = [(n, float(w)) for (n, w) in loras if str(n).strip() != ""]
        # if clean_loras:
            # meta["loras"] = [{"name": n, "weight": w} for (n, w) in clean_loras]
        # # App info
        # v = self._get_comfyui_version()
        # meta["app"] = {"name": "ComfyUI", **({"version": v} if v else {})}
        # meta["extra_pnginfo"] = extra_pnginfo if isinstance(extra_pnginfo, dict) else (extra_pnginfo or {})
        # meta["prompt"] = prompt if isinstance(prompt, dict) else (prompt or {})
        # return meta

    # def _fmt_f(self, x) -> str:
        # try:
            # s = f"{float(x):.3f}"
            # s = s.rstrip('0').rstrip('.')
            # return s if s else "0"
        # except Exception:
            # return str(x)

    # def _build_a111_parameters(self, m: Dict[str, Any]) -> str:
        # pos = (m.get("positive") or "")
        # neg = (m.get("negative") or "")
        # steps = int(m.get("steps") or 0)
        # sampler = (m.get("sampler_name") or m.get("sampler") or "").strip()
        # scheduler = (m.get("scheduler_name") or m.get("scheduler") or "").strip()
        # guidance = m.get("guidance")
        # scale = m.get("scale")
        # denoise = m.get("denoise")
        # seed = m.get("seed_value")
        # w = m.get("width")
        # h = m.get("height")
        # clip_skip = int(m.get("clip_skip") or 0)
        # model = (m.get("model_name") or "").strip()

        # parts = []

        # # Positive line: include only if non-empty.
        # if pos.strip() != "":
            # parts.append(pos)

        # # Negative line: include only if non-empty.
        # if neg.strip() != "":
            # parts.append(f"Negative prompt: {neg}")

        # # Parameters line (always present)
        # tail = [
            # f"Steps: {steps}",
            # (f"Sampler: {sampler}" if sampler else None),
            # (f"Scheduler: {scheduler}" if scheduler else None),
            # (f"Guidance: {self._fmt_f(guidance)}" if guidance is not None else None),
            # (f"Scale: {self._fmt_f(scale)}" if scale is not None else None),
            # (f"Denoise: {self._fmt_f(denoise)}" if denoise is not None else None),
            # (f"Seed: {seed}" if seed is not None else None),
            # (f"Size: {w}x{h}" if w and h else None),
            # ((f"Clip skip: {abs(clip_skip)}") if clip_skip != 0 else None),
            # ((f"Model: {model}") if model else None),
        # ]
        # parts.append(", ".join([t for t in tail if t]))

        # # Add ComfyUI version right before LoRA section
        # ver = (m.get("app") or {}).get("version") or self._get_comfyui_version()
        # parts.append(f"Version: ComfyUI{(' ' + ver) if ver else ''}")

        # # LoRA lines in requested format
        # loras = m.get("loras") or []
        # for idx, d in enumerate(loras, start=1):
            # name = (d.get('name') or '').strip()
            # if not name:
                # continue
            # weight = d.get('weight')
            # try:
                # weight = float(weight)
            # except Exception:
                # weight = 0.0
            # parts.append(f"LoRA_{idx}: {name}, LoRA_{idx}_Weight: {self._fmt_f(weight)}")
        # return "\n".join(parts)

    # def _embed_pnginfo(self, metadata):
        # pnginfo = PngImagePlugin.PngInfo()
        # try:
            # a111 = self._build_a111_parameters(metadata)
            # pnginfo.add_text("parameters", a111)

            # # Always embed 'prompt' (stringify even if list/dict)
            # if "prompt" in metadata:
                # pnginfo.add_text("prompt", json.dumps(metadata["prompt"]))

            # # Robust workflow extraction
            # wf = self._extract_workflow(metadata.get("extra_pnginfo"))
            # if wf is not None:
                # pnginfo.add_text("workflow", json.dumps(wf))

            # if "app" in metadata:
                # pnginfo.add_text("app", json.dumps(metadata["app"]))
        # except Exception as e:
            # print(f"[FRED_Image_Saver] PNGInfo embed failed: {e}")
        # return pnginfo

    # def _extract_workflow(self, extra_pnginfo):
        # # extra_pnginfo can be a dict (core Comfy) or a list of dicts (some front-ends)
        # def _maybe_json(x):
            # if isinstance(x, str):
                # try:
                    # return json.loads(x)
                # except Exception:
                    # return x
            # return x

        # if isinstance(extra_pnginfo, dict):
            # return _maybe_json(extra_pnginfo.get("workflow"))

        # if isinstance(extra_pnginfo, list):
            # for item in extra_pnginfo:
                # if isinstance(item, dict) and "workflow" in item:
                    # return _maybe_json(item["workflow"])
        # return None

    # def _expand_tokens(self, s: str, meta: Dict[str, Any], time_format: str) -> str:
        # if not s:
            # return ""
        # s = os.path.expandvars(s)

        # token_map: Dict[str, str] = {}
        # for k, v in meta.items():
            # if isinstance(v, (str, int, float)):
                # token_map[k] = str(v)

        # model_nm = token_map.get("model_name", "")
        # token_map["basemodelname"] = os.path.splitext(model_nm)[0] if model_nm else ""

        # # IMPORTANT : toujours injecter les tokens de date
        # token_map["date"] = time.strftime("%Y%m%d")
        # token_map["date_dash"] = time.strftime("%Y-%m-%d")
        # token_map["time"] = time.strftime("%H%M%S")
        # token_map["datetime"] = time.strftime(time_format or "%Y-%m-%d-%H%M%S")

        # alias_map = {
            # "model": "model_name",
            # "sampler": "sampler_name",
            # "scheduler": "scheduler_name",
            # "seed": "seed_value",
            # "cfg": "guidance"
        # }
        # for a, b in alias_map.items():
            # if a not in token_map and b in token_map:
                # token_map[a] = token_map[b]

        # # LoRA token filtering: include weight_N only if name_N is set
        # for i in (1, 2, 3):
            # name_k = f"lora_name_{i}"
            # weight_k = f"lora_weight_{i}"
            # name_v = str(meta.get(name_k, "") or "").strip()
            # if name_v == "":
                # token_map.pop(weight_k, None)
            # else:
                # token_map[name_k] = name_v
                # token_map[weight_k] = str(meta.get(weight_k, ""))

        # # Replace tokens (supports %token and %token%)
        # for key in sorted(token_map.keys(), key=len, reverse=True):
            # val = self._sanitize_component(token_map[key])
            # for tok in (f"%{key}%", f"%{key}"):
                # if tok in s:
                    # s = s.replace(tok, val)

        # # Supprime les tokens non reconnus
        # s = __import__("re").sub(r"%[A-Za-z0-9_]+%?", "", s)

        # return s
    # # def _expand_tokens(self, s: str, meta: Dict[str, Any], time_format: str) -> str:
        # # if not s:
            # # return ""
        # # s = os.path.expandvars(s)

        # # token_map: Dict[str, str] = {}
        # # for k, v in meta.items():
            # # if isinstance(v, (str, int, float)):
                # # token_map[k] = str(v)
        # # model_nm = token_map.get("model_name", "")
        # # token_map["basemodelname"] = os.path.splitext(model_nm)[0] if model_nm else ""
        # # token_map["date"] = time.strftime("%Y%m%d")
        # # token_map["date_dash"] = time.strftime("%Y-%m-%d")
        # # token_map["time"] = time.strftime("%H%M%S")
        # # token_map["datetime"] = time.strftime(time_format or "%Y-%m-%d-%H%M%S")
        # # alias_map = {"model": "model_name", "sampler": "sampler_name", "scheduler": "scheduler_name", "seed": "seed_value", "cfg": "guidance"}
        # # for a, b in alias_map.items():
            # # if a not in token_map and b in token_map:
                # # token_map[a] = token_map[b]

        # # # LoRA token filtering: include weight_N only if name_N is set
        # # for i in (1, 2, 3):
            # # name_k = f"lora_name_{i}"
            # # weight_k = f"lora_weight_{i}"
            # # name_v = str(meta.get(name_k, "") or "").strip()
            # # if name_v == "":
                # # token_map.pop(weight_k, None)
            # # else:
                # # token_map[name_k] = name_v
                # # token_map[weight_k] = str(meta.get(weight_k, ""))

        # # # Replace tokens (supports %token and %token%)
        # # for key in sorted(token_map.keys(), key=len, reverse=True):
            # # val = self._sanitize_component(token_map[key])
            # # for tok in (f"%{key}%", f"%{key}"):
                # # if tok in s:
                    # # s = s.replace(tok, val)
        # # s = __import__("re").sub(r"%[A-Za-z0-9_]+%?", "", s)
        # # return s

    # def _ensure_outdir(self, subdir: str, meta: Dict[str, Any], time_format: str) -> str:
        # base = folder_paths.get_output_directory()
        # subdir = self._expand_tokens(subdir or "", meta, time_format)
        # if subdir and os.path.isabs(subdir):
            # parts = []
            # drive, tail = os.path.splitdrive(subdir)
            # for part in tail.replace("\\", "/").split("/"):
                # if part:
                    # parts.append(self._sanitize_component(part))
            # outdir = os.path.join(drive + os.sep if drive else os.sep, *parts) if parts else subdir
        # else:
            # parts = []
            # for part in subdir.replace("\\", "/").split("/"):
                # if part:
                    # parts.append(self._sanitize_component(part))
            # outdir = os.path.join(base, *parts) if parts else base
        # outdir = os.path.normpath(outdir)
        # os.makedirs(outdir, exist_ok=True)
        # return outdir

    # def _compute_grid_dims(self, n: int, col_max: int, row_max: int) -> Tuple[int, int]:
        # """
        # Choose rows/cols respecting the provided maxima (x_size = columns, y_size = rows),
        # while avoiding blank tiles when n < col_max*row_max.
        # """
        # if n <= 0:
            # return 1, 1
        # cols = min(col_max, n)
        # rows = max(1, math.ceil(n / cols))
        # rows = min(rows, row_max)

        # # Ensure capacity >= n; expand within limits.
        # while rows * cols < n and rows < row_max:
            # rows += 1
        # while rows * cols < n and cols < col_max:
            # cols += 1

        # rows = max(1, min(rows, row_max))
        # cols = max(1, min(cols, col_max))
        # return rows, cols  # (rows, cols)

    # def _assemble_grid(self, images_pil: List[Image.Image], rows: int, cols: int) -> Image.Image:
        # n = len(images_pil)
        # if n == 0:
            # raise ValueError("No images to grid")
        # w, h = images_pil[0].size
        # grid = Image.new("RGB", (cols * w, rows * h), color="#ffffff")
        # idx = 0
        # for r in range(rows):
            # for c in range(cols):
                # if idx >= n:
                    # break
                # grid.paste(images_pil[idx], (c * w, r * h))
                # idx += 1
        # return grid

    # # ------------------------------ main ------------------------------
    # def save(self,
             # images,
             # save_single_image: bool,
             # filename: str,
             # path: str,
             # time_format: str,
             # extension: str,
             # optimize_png: bool,
             # lossless_webp: bool,
             # quality_jpeg_or_webp: int,
             # save_workflow_as_json: bool,
             # embed_workflow_in_png: bool,
             # width: int,
             # height: int,
             # scale: float,
             # denoise: float,
             # guidance: float,
             # clip_skip: int,
             # steps: int,
             # seed_value: int,
             # sampler_name: str,
             # scheduler_name: str,
             # positive: str,
             # negative: str,
             # model_name: str,
             # lora_name_1: str,
             # lora_weight_1: float,
             # lora_name_2: str,
             # lora_weight_2: float,
             # lora_name_3: str,
             # lora_weight_3: float,
             # save_as_grid_if_multi: bool,
             # grid_filename: str,
             # grid_path: str,
             # grid_column_max: int,
             # grid_row_max: int,
             # grid_extension: str,
             # grid_quality_jpeg_or_webp: int,
             # prompt: Dict[str, Any] = None,
             # extra_pnginfo: Dict[str, Any] = None,
             # unique_id: str = ""):

        # # --- Coercer tout ce qui pourrait arriver en liste (widgets / autres nodes) ---
        # # ---- COERCE scalar/numeric inputs (avoid list types) ----
        # filename      = self._scalar(filename)
        # path          = self._scalar(path)
        # time_format   = self._scalar(time_format)
        # extension     = self._scalar(extension)
        # optimize_png  = bool(self._scalar(optimize_png))
        # lossless_webp = bool(self._scalar(lossless_webp))
        # quality_jpeg_or_webp = self._to_int(quality_jpeg_or_webp, 100)
        # save_workflow_as_json = bool(self._scalar(save_workflow_as_json))
        # embed_workflow_in_png = bool(self._scalar(embed_workflow_in_png))

        # save_as_grid_if_multi = bool(self._scalar(save_as_grid_if_multi))

        # grid_filename = self._scalar(grid_filename)
        # grid_path     = self._scalar(grid_path)
        # grid_extension = self._scalar(grid_extension)
        # grid_column_max = self._to_int(grid_column_max, 5)
        # grid_row_max    = self._to_int(grid_row_max, 1)
        # grid_quality_jpeg_or_webp = self._to_int(grid_quality_jpeg_or_webp, 100)

        # # prompts / meta numeric
        # positive = self._scalar(positive)
        # negative = self._scalar(negative)
        # sampler_name   = self._scalar(sampler_name)
        # scheduler_name = self._scalar(scheduler_name)
        # model_name     = self._scalar(model_name)

        # width      = self._to_int(width, 1024)
        # height     = self._to_int(height, 1024)
        # scale      = self._to_float(scale, 1.40)
        # denoise    = self._to_float(denoise, 0.60)
        # guidance   = self._to_float(guidance, 2.20)
        # clip_skip  = self._to_int(clip_skip, 1)
        # steps      = self._to_int(steps, 20)
        # seed_value = self._to_int(seed_value, 0)

        # lora_name_1 = self._scalar(lora_name_1); lora_weight_1 = self._to_float(lora_weight_1, 1.0)
        # lora_name_2 = self._scalar(lora_name_2); lora_weight_2 = self._to_float(lora_weight_2, 1.0)
        # lora_name_3 = self._scalar(lora_name_3); lora_weight_3 = self._to_float(lora_weight_3, 1.0)

        # # --- Normalize input: accept list/tuple OR a torch batch tensor ---
        # img_list: List[torch.Tensor] = []
        # if isinstance(images, (list, tuple)):
            # input_kind = "list"
            # for im in images:
                # if not isinstance(im, torch.Tensor):
                    # raise ValueError(f"[FRED_Image_Saver] Unsupported list element type: {type(im)}")
                # if im.ndim == 4:  # (B,H,W,C) batch FOUND inside a list → flatten it
                    # b_in = im.shape[0]
                    # for i in range(b_in):
                        # img_list.append(im[i])
                # elif im.ndim == 3:  # (H,W,C)
                    # img_list.append(im)
                # else:
                    # raise ValueError(f"[FRED_Image_Saver] Unsupported tensor shape in list: {tuple(im.shape)}")
        # elif isinstance(images, torch.Tensor):
            # input_kind = "tensor"
            # if images.ndim == 4:  # (B,H,W,C)
                # for i in range(images.shape[0]):
                    # img_list.append(images[i])
            # elif images.ndim == 3:  # (H,W,C) single
                # img_list.append(images)
            # else:
                # raise ValueError(f"[FRED_Image_Saver] Unsupported tensor shape: {tuple(images.shape)}")
        # else:
            # raise ValueError(f"[FRED_Image_Saver] Unsupported images input type: {type(images)}")

        # b = len(img_list)
        # is_multi_input = b > 1
        # first_t = img_list[0]
        # in_dev = first_t.device
        # in_dt = first_t.dtype

        # shapes = [tuple(t.shape) for t in img_list]
        # print(f"[FRED_Image_Saver] input_kind={input_kind}, count={b}, is_multi_input={is_multi_input}, shapes={shapes[:5]}{'...' if len(shapes)>5 else ''}")

        # # --- metadata base (shared) ---
        # metadata = self._build_metadata(
            # width=width, height=height, scale=scale, denoise=denoise, guidance=guidance,
            # clip_skip=clip_skip, steps=steps, seed_value=seed_value,
            # sampler_name=sampler_name, scheduler_name=scheduler_name,
            # positive=positive, negative=negative, model_name=model_name,
            # loras=[(lora_name_1, lora_weight_1), (lora_name_2, lora_weight_2), (lora_name_3, lora_weight_3)],
            # extra_pnginfo=extra_pnginfo or {}, prompt=prompt or {},
        # )
        # # Expose LoRA fields for token expansion (weights only if name non-empty)
        # if (lora_name_1 or "").strip():
            # metadata["lora_name_1"] = lora_name_1
            # metadata["lora_weight_1"] = lora_weight_1
        # if (lora_name_2 or "").strip():
            # metadata["lora_name_2"] = lora_name_2
            # metadata["lora_weight_2"] = lora_weight_2
        # if (lora_name_3 or "").strip():
            # metadata["lora_name_3"] = lora_name_3
            # metadata["lora_weight_3"] = lora_weight_3

        # saved_paths: List[str] = []

        # # --- Single images (optionnel) ---
        # if save_single_image:
            # outdir = self._ensure_outdir(path, metadata, time_format)
            # base_template = filename.strip() if filename else "image"
            # base = self._expand_tokens(base_template, metadata, time_format)
            # base = self._sanitize_component(base) or "image"

            # for t in img_list:
                # save_dir, filename_base, counter, subfolder, _ = folder_paths.get_save_image_path(base, outdir)
                # name = f"{filename_base}_{counter:05d}"
                # full = os.path.join(save_dir, f"{name}.{extension}")
                # while os.path.exists(full):
                    # counter += 1
                    # name = f"{filename_base}_{counter:05d}"
                    # full = os.path.join(save_dir, f"{name}.{extension}")

                # img = self._tensor_to_pil(t)

                # pnginfo = None
                # if extension.lower() == "png" and embed_workflow_in_png:
                    # pnginfo = self._embed_pnginfo(metadata)

                # try:
                    # if extension.lower() == 'png':
                        # params = {"optimize": bool(optimize_png)}
                        # if pnginfo is not None:
                            # params["pnginfo"] = pnginfo
                        # img.save(full, format='PNG', **params)
                    # elif extension.lower() == 'jpeg':
                        # img.save(full, format='JPEG', quality=int(quality_jpeg_or_webp), optimize=True)
                    # elif extension.lower() == 'webp':
                        # if lossless_webp:
                            # img.save(full, format='WEBP', lossless=True)
                        # else:
                            # img.save(full, format='WEBP', quality=int(quality_jpeg_or_webp))
                    # else:
                        # comfy.utils.save_image(img, save_dir, name)
                # except Exception as e:
                    # print(f"[FRED_Image_Saver] Save error: {e}. Falling back to Comfy save_image.")
                    # comfy.utils.save_image(img, save_dir, name)

                # if save_workflow_as_json:
                    # try:
                        # sidecar = os.path.join(save_dir, f"{name}.json")
                        # wf = self._extract_workflow(metadata.get("extra_pnginfo"))
                        # if wf is None:
                            # # optional: fall back to UI-export-style workflow if you ever stash it elsewhere
                            # print("[FRED_Image_Saver] No workflow found in extra_pnginfo; skipping sidecar.")
                        # else:
                            # with open(sidecar, "w", encoding="utf-8") as f:
                                # json.dump(wf, f, ensure_ascii=False, indent=2)
                    # except Exception as e:
                        # print(f"[FRED_Image_Saver] Could not write JSON sidecar: {e}")

                # saved_paths.append(full)

        # # --- GRID (une seule fois, liste ou batch) ---
        # imgs_pil = [self._tensor_to_pil(t) for t in img_list]
        # rows, cols = self._compute_grid_dims(len(imgs_pil), grid_column_max, grid_row_max)
        # capacity = rows * cols
        # if len(imgs_pil) > capacity:
            # print(f"[FRED_Image_Saver] Grid capacity {capacity} (rows={rows}, cols={cols}) cannot fit all {len(imgs_pil)} images. {len(imgs_pil) - capacity} left over.")
            # imgs_pil = imgs_pil[:capacity]

        # grid_img = self._assemble_grid(imgs_pil, rows, cols)

        # gi_np = np.asarray(grid_img, dtype=np.uint8)
        # gi_t  = torch.from_numpy(gi_np).to(in_dev).float() / 255.0
        # grid_image_out = gi_t.unsqueeze(0).to(dtype=in_dt)

        # if save_as_grid_if_multi and len(imgs_pil) > 1:
            # meta_grid = dict(metadata)
            # meta_grid["img_count"] = len(imgs_pil)
            # meta_grid["grid_rows"] = rows
            # meta_grid["grid_cols"] = cols

            # outdir_grid = self._ensure_outdir(grid_path, meta_grid, time_format)
            # base_grid_t = grid_filename.strip() if grid_filename else "grid"
            # base_grid = self._expand_tokens(base_grid_t, meta_grid, time_format)
            # base_grid = self._sanitize_component(base_grid) or "grid"

            # save_dir, filename_base, counter, subfolder, _ = folder_paths.get_save_image_path(
                # base_grid, outdir_grid, grid_img.width, grid_img.height
            # )
            # name = f"{filename_base}_{counter:05d}"
            # full = os.path.join(save_dir, f"{name}.{grid_extension}")
            # while os.path.exists(full):
                # counter += 1
                # name = f"{filename_base}_{counter:05d}"
                # full = os.path.join(save_dir, f"{name}.{grid_extension}")

            # pnginfo_grid = None
            # if grid_extension.lower() == "png" and embed_workflow_in_png:
                # pnginfo_grid = self._embed_pnginfo(meta_grid)

            # try:
                # if grid_extension.lower() == 'png':
                    # params = {"optimize": bool(optimize_png)}
                    # if pnginfo_grid is not None:
                        # params["pnginfo"] = pnginfo_grid
                    # grid_img.save(full, format='PNG', **params)
                # elif grid_extension.lower() == 'jpeg':
                    # grid_img.save(full, format='JPEG', quality=int(grid_quality_jpeg_or_webp), optimize=True)
                # elif grid_extension.lower() == 'webp':
                    # if lossless_webp:
                        # grid_img.save(full, format='WEBP', lossless=True)
                    # else:
                        # grid_img.save(full, format='WEBP', quality=int(grid_quality_jpeg_or_webp))
                # else:
                    # comfy.utils.save_image(grid_img, save_dir, name)
            # except Exception as e:
                # print(f"[FRED_Image_Saver] Grid save error: {e}. Falling back to Comfy save_image.")
                # comfy.utils.save_image(grid_img, save_dir, name)

            # if save_workflow_as_json:
                # try:
                    # sidecar = os.path.join(save_dir, f"{name}.json")
                    # with open(sidecar, "w", encoding="utf-8") as f:
                        # json.dump(meta_grid, f, ensure_ascii=False, indent=2)
                # except Exception as e:
                    # print(f"[FRED_Image_Saver] Could not write GRID JSON sidecar: {e}")

            # saved_paths.append(full)

        # # Retour
        # return (grid_image_out, saved_paths[-1] if saved_paths else "", HELP_MESSAGE)

    # @staticmethod
    # def IS_CHANGED(*args, **kwargs):
        # # Ignore data-like inputs so accumulation doesn't retrigger saves unnecessarily
        # drop = {"images", "prompt", "extra_pnginfo", "unique_id"}
        # clean = {k: v for k, v in kwargs.items() if k not in drop}
        # return json.dumps(clean, sort_keys=True, ensure_ascii=False)


# NODE_CLASS_MAPPINGS = {
    # "FRED_ImageSaver": FRED_ImageSaver,
# }

# NODE_DISPLAY_NAME_MAPPINGS = {
    # "FRED_ImageSaver": "👑 FRED Image Saver",
# }