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

HELP_MESSAGE = """
👑 FRED_Image_Saver

🔹 PURPOSE:
Save images flexibly with dynamic filename/path tokens, Automatic1111-style metadata embedding, and optional grid saving.  
Supports single images, tiled grid images, or both at the same time.

📥 INPUTS:
- images • one or more images to save
- save_single_image • if ON, save each input as its own file
- filename / path • templates with tokens for naming (counter is always appended for uniqueness)
- time_format • used by %datetime
- extension / quality_jpeg_or_webp / optimize_png / lossless_webp • file format and compression options
- save_workflow_as_json • writes a sidecar JSON file with metadata
- embed_workflow_in_png • embeds workflow/prompt/app metadata into PNG tEXt
- metadata fields (width, height, scale, denoise, guidance, clip_skip, steps, seed_value, sampler_name, scheduler_name, positive, negative, model_name)
- lora_name_X / lora_weight_X • up to 3 LoRA names/weights (weights only expand if name is provided)
- save_as_grid_if_multi • if ON, also save a tiled grid when multiple images are given
- grid_column_max / grid_row_max • maximum grid size
- grid_column_label / grid_row_label • optional labels drawn on grid
- grid_filename / grid_path / grid_extension / grid_quality_jpeg_or_webp • grid output settings

⚙️ KEY OPTIONS:
- Single image saves:
  • Saved to “path” (relative to Comfy’s output folder unless absolute).
  • Uses filename template with tokens; unique counter (_00001, _00002, …) appended.
  • PNG: can embed JSON + optimize (lossless).  
  • JPEG/WEBP: adjustable quality, WEBP can be forced lossless.
  • Sidecar JSON: includes parameters, prompts, model, LoRAs, app version, workflow.

- Grid saves:
  • Only created if ≥2 input images and save_as_grid_if_multi is ON.
  • Grid size auto-derived up to max cols/rows; if not all images fit, the first N are used and leftovers are logged.
  • Grid uses same token expansion as single images (grid_filename, grid_path).
  • Metadata mirrors single-image save but adds “img_count”, “grid_rows”, “grid_cols”.
  • Labels: optional column/row labels drawn with a font.

- Metadata embedding (PNG tEXt keys):
  • "parameters": Automatic1111-style text (Positive, Negative, Steps, Sampler, Scheduler, Guidance/Scale/Denoise, Seed, Size, Clip skip, Model, Version, LoRAs).
  • "prompt": workflow prompt JSON.
  • "workflow": workflow graph JSON (if available).
  • "app": ComfyUI name + short git commit.

- Token system (usable in both filename and path):
  • Time/date: %date, %date_dash, %time, %datetime
  • Params: %seed / %seed_value, %model / %model_name / %basemodelname, %sampler / %sampler_name, %scheduler / %scheduler_name, %width, %height, %steps, %cfg, %guidance, %scale, %denoise, %clip_skip
  • Batch info: %img_count (grid/batch size)
  • LoRAs: %lora_name_1..3, %lora_weight_1..3 (weights only if name present)

📤 OUTPUTS:
- last_image_saved_path • path of the last saved single image (empty if none saved)
- last_grid_saved_path • path of the last saved grid image (empty if none saved)
- help • this message

📝 NOTES & TIPS:
- This node does NOT resize or alter pixel data; width/height/steps/etc. are recorded only.
- %basemodelname expands to model_name without extension.
- PNG optimize=True reduces size but slows saves.
- WEBP: if “lossless_webp” is ON, quality slider is ignored.
- LoRA tokens are included only when a name is set.
- Unique filename generation is guaranteed (Comfy counter + safety loop).
- IS_CHANGED ignores large tensor inputs (“images”, “prompt”, “extra_pnginfo”, “unique_id”) so reruns don’t trigger unnecessary re-saves.

Examples:
- filename: "%basemodelname_%datetime_seed_%seed"
- path:     "Fred_nodes/%date_dash/"
- grid_filename: "%model_name_%date_%time_grid_%img_count"
- grid_path:     "Test/%model_name/Grid/%date_dash/"
"""


INVALID_FS_CHARS = '<>:"/\\|?*'

class FRED_ImageSaver(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                # --- IMAGE & FILE OUTPUT ---
                "images": ("IMAGE", {"tooltip": "image(s) to save"}),
                "save_single_image": ("BOOLEAN", {"default": True, "tooltip": "Save each input image as its own file."}),
                "filename": (IO.STRING, {"default": "%lora_name_1_weight_%lora_weight_1_%model_name_%datetime", "multiline": False, "tooltip": "base filename (counter appended)"}),
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
                "grid_filename": (IO.STRING, {"default": "%lora_name_1_weight_%lora_weight_1_%datetime_%model_name test grid_img count_%img_count", "multiline": False, "tooltip": "filename prefix for the grid image"}),
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
    CATEGORY = "👑FRED/image"
    DESCRIPTION = "Save images with A1111‑style parameters text and robust tokenized filenames/paths. Can also save a grid when there is multiple images."
    OUTPUT_NODE = True

    # ------------------------------ helpers ------------------------------
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
        if t.dtype in (torch.float16, torch.float32, torch.float64):
            arr = (t.clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
        else:
            arr = t.clamp(0, 255).byte().cpu().numpy()
        if arr.ndim == 3 and arr.shape[2] >= 3:
            return Image.fromarray(arr[..., :3], mode="RGB")
        elif arr.ndim == 2:
            return Image.fromarray(arr, mode="L").convert("RGB")
        raise ValueError("Unsupported image tensor shape for saving")

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
            "positive": positive,
            "negative": negative,
            "model_name": model_name,
        }
        clean_loras = [(n, float(w)) for (n, w) in loras if str(n).strip() != ""]
        if clean_loras:
            meta["loras"] = [{"name": n, "weight": w} for (n, w) in clean_loras]
        # App info
        v = self._get_comfyui_version()
        meta["app"] = {"name": "ComfyUI", **({"version": v} if v else {})}
        # Optional embeds
        if extra_pnginfo and isinstance(extra_pnginfo, dict):
            meta["extra_pnginfo"] = extra_pnginfo
        if prompt and isinstance(prompt, dict):
            meta["prompt"] = prompt
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

        # Always emit a first line for "positive", even if empty.
        # Using a single space keeps line #1 present without adding visible text.
        parts.append(pos if pos.strip() != "" else " ")

        # Negative line always present (may be empty after the colon)
        parts.append(f"Negative prompt: {neg}")

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

        # Add LoRA lines in requested format
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

    def _embed_pnginfo(self, metadata: Dict[str, Any]) -> PngImagePlugin.PngInfo:
        pnginfo = PngImagePlugin.PngInfo()
        try:
            a111 = self._build_a111_parameters(metadata)
            pnginfo.add_text("parameters", a111)
            if "prompt" in metadata:
                pnginfo.add_text("prompt", json.dumps(metadata["prompt"]))
            if "extra_pnginfo" in metadata:
                wf = metadata["extra_pnginfo"].get("workflow", {}) if isinstance(metadata["extra_pnginfo"], dict) else {}
                pnginfo.add_text("workflow", json.dumps(wf))
            if "app" in metadata:
                pnginfo.add_text("app", json.dumps(metadata["app"]))
        except Exception as e:
            print(f"[FRED_Image_Saver] PNGInfo embed failed: {e}")
        return pnginfo

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
        token_map["date"] = time.strftime("%Y%m%d")
        token_map["date_dash"] = time.strftime("%Y-%m-%d")
        token_map["time"] = time.strftime("%H%M%S")
        token_map["datetime"] = time.strftime(time_format or "%Y-%m-%d-%H%M%S")
        alias_map = {
            "model": "model_name",
            "sampler": "sampler_name",
            "scheduler": "scheduler_name",
            "seed": "seed_value",
            "cfg": "guidance",
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
        s = __import__("re").sub(r"%[A-Za-z0-9_]+%?", "", s)
        return s

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

        - Start with as many columns as possible up to col_max, but not more than n.
        - Grow rows up to row_max until rows*cols >= n.
        - If still not enough (because cols was tiny), grow cols up to col_max.
        """
        if n <= 0:
            return 1, 1
        cols = min(col_max, n)
        rows = max(1, math.ceil(n / cols))
        rows = min(rows, row_max)

        # Ensure capacity >= n; if not, expand within limits.
        while rows * cols < n and rows < row_max:
            rows += 1
        while rows * cols < n and cols < col_max:
            cols += 1

        # Final clamps
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
             images: torch.Tensor,
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

        # ui_images: List[Dict[str, str]] = []
        grid_image_out = None  # Will hold a single-image batch (1,H,W,3) or stay None
        output_base = folder_paths.get_output_directory()
        b, h, w, c = images.shape  # (B, H, W, C)

        metadata = self._build_metadata(
            width=width,
            height=height,
            scale=scale,
            denoise=denoise,
            guidance=guidance,
            clip_skip=clip_skip,
            steps=steps,
            seed_value=seed_value,
            sampler_name=sampler_name,
            scheduler_name=scheduler_name,
            positive=positive,
            negative=negative,
            model_name=model_name,
            loras=[(lora_name_1, lora_weight_1), (lora_name_2, lora_weight_2), (lora_name_3, lora_weight_3)],
            extra_pnginfo=extra_pnginfo or {},
            prompt=prompt or {},
        )
        # Expose LoRA fields for token expansion (weights only if name is non-empty)
        if (lora_name_1 or "").strip():
            metadata["lora_name_1"] = lora_name_1
            metadata["lora_weight_1"] = lora_weight_1
        if (lora_name_2 or "").strip():
            metadata["lora_name_2"] = lora_name_2
            metadata["lora_weight_2"] = lora_weight_2
        if (lora_name_3 or "").strip():
            metadata["lora_name_3"] = lora_name_3
            metadata["lora_weight_3"] = lora_weight_3

        # Single images
        saved_paths: List[str] = []
        if save_single_image:
            outdir = self._ensure_outdir(path, metadata, time_format)
            base_template = filename.strip() if filename else "image"
            base = self._expand_tokens(base_template, metadata, time_format)
            base = self._sanitize_component(base) or "image"

            for i in range(b):
                save_dir, filename_base, counter, subfolder, _ = folder_paths.get_save_image_path(base, outdir)
                name = f"{filename_base}_{counter:05d}"
                full = os.path.join(save_dir, f"{name}.{extension}")
                while os.path.exists(full):
                    counter += 1
                    name = f"{filename_base}_{counter:05d}"
                    full = os.path.join(save_dir, f"{name}.{extension}")

                img = self._tensor_to_pil(images[i])

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
                    # If the saved path is inside Comfy's output tree, add a UI entry so it shows in the node
                    # try:
                        # rel = os.path.relpath(full, output_base)
                        # if not rel.startswith(".."):
                            # # `subfolder` came from get_save_image_path; filename is just "<name>.<ext>"
                            # ui_images.append({"filename": f"{name}.{extension}", "subfolder": subfolder, "type": "output"})
                        # else:
                            # # Outside output dir → Comfy can't preview it inline
                            # pass
                    # except Exception:
                        # pass
                except Exception as e:
                    print(f"[FRED_Image_Saver] Save error: {e}. Falling back to Comfy save_image.")
                    comfy.utils.save_image(img, save_dir, name)
                    # ui_images.append({"filename": f"{name}.{extension}", "subfolder": subfolder, "type": "output"})

                if save_workflow_as_json:
                    try:
                        sidecar = os.path.join(save_dir, f"{name}.json")
                        with open(sidecar, "w", encoding="utf-8") as f:
                            json.dump(metadata, f, ensure_ascii=False, indent=2)
                    except Exception as e:
                        print(f"[FRED_Image_Saver] Could not write JSON sidecar: {e}")

                saved_paths.append(full)

        # Grid image
        imgs_pil = [self._tensor_to_pil(images[i]) for i in range(b)]

        # Decide grid shape using the provided maxima (x_size = columns, y_size = rows)
        rows, cols = self._compute_grid_dims(b, grid_column_max, grid_row_max)
        capacity = rows * cols

        # If there are more images than capacity, use only the first batch and warn.
        if b > capacity:
            print(f"[FRED_Image_Saver] Grid capacity {capacity} (rows={rows}, cols={cols}) "
                  f"cannot fit all {b} images. {b - capacity} image(s) left over.")
            imgs_pil = imgs_pil[:capacity]

        grid_img = self._assemble_grid(imgs_pil, rows=rows, cols=cols)

        # Provide a tensor output for preview/flow
        grid_image_out = self._pil_to_tensor(grid_img).to(dtype=images.dtype, device=images.device).unsqueeze(0)  # 1,H,W,3

        if save_as_grid_if_multi and b >= 2:
            meta_grid = dict(metadata)
            meta_grid["img_count"] = len(imgs_pil)
            meta_grid["grid_rows"] = rows
            meta_grid["grid_cols"] = cols
            # -------------

            outdir_grid = self._ensure_outdir(grid_path, meta_grid, time_format)
            base_grid_t = grid_filename.strip() if grid_filename else "grid"
            base_grid = self._expand_tokens(base_grid_t, meta_grid, time_format)
            base_grid = self._sanitize_component(base_grid) or "grid"

            save_dir, filename_base, counter, subfolder, _ = folder_paths.get_save_image_path(base_grid, outdir_grid, grid_img.width, grid_img.height)
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
                # try:
                    # rel = os.path.relpath(full, output_base)
                    # if not rel.startswith(".."):
                        # ui_images.append({"filename": f"{name}.{grid_extension}", "subfolder": subfolder, "type": "output"})
                # except Exception:
                    pass
            except Exception as e:
                print(f"[FRED_Image_Saver] Grid save error: {e}. Falling back to Comfy save_image.")
                comfy.utils.save_image(grid_img, save_dir, name)
                # ui_images.append({"filename": f"{name}.{grid_extension}", "subfolder": subfolder, "type": "output"})

            if save_workflow_as_json:
                try:
                    sidecar = os.path.join(save_dir, f"{name}.json")
                    with open(sidecar, "w", encoding="utf-8") as f:
                        json.dump(meta_grid, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    print(f"[FRED_Image_Saver] Could not write GRID JSON sidecar: {e}")

            saved_paths.append(full)

        if grid_image_out is None:
            # No grid created (e.g., b < 2). To keep downstream nodes happy,
            # return the first input image as a 1-image batch.
            grid_image_out = images[:1].clone()

        elif save_as_grid_if_multi and b == 1:
            print("[FRED_Image_Saver] save_as_grid_if_multi is ON but only 1 image was provided — skipping grid.")

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