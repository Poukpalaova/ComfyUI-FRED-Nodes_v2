# FRED_ImageSaver_v2.py

# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import json
import time
import math
import re
from typing import Any, Dict, List, Tuple, Optional, Union

import numpy as np
import torch
from PIL import Image, PngImagePlugin

import folder_paths
import comfy.utils

HELP_MESSAGE = """
👑 FRED_ImageSaver_v2

🔹 PURPOSE
Save images flexibly with dynamic filename/path tokens, Automatic1111-style metadata embedding, and optional grid saving.
Accepts **list or batch** as input (mapping disabled on images), so the node runs **once** for a whole set.

📥 NEW FEATURES (v2)
- **parameters** • A1111 "tail" string (e.g., "Steps: 30, Sampler: euler, Seed: 123, Size: 1024x1024, …").
  Use with **use_parameters_input** to control precedence over widgets (when ON, parsed values override widgets).
- **loras_infos** • Single compact field for any number of LoRAs. Supports two syntaxes:
  • A: "name[:weight],name2[:weight2],..." (weight default 1.0; use '\\,' for literal comma in names)
  • B: "name,weight,name2,weight2,..." (pairs)
- **Multi-image list mapping** • When other inputs (seed, steps, sampler, etc.) are lists, each image gets its
  corresponding indexed value (cyclic if list is shorter). Previously only the first value was used for all images.

📥 INPUTS (same as v1 plus the above)
- images • one or more images to save (**list or batch accepted; mapping disabled**).
- save_single_image • if ON, save each input image as its own file (in addition to the grid, if enabled).
- filename / path • templates with tokens for naming (a counter is always appended for uniqueness).
- time_format • used by %datetime.
- extension / quality_jpeg_or_webp / optimize_png / lossless_webp • single-image format & compression.
- save_workflow_as_json • writes a sidecar JSON file (metadata).
- embed_workflow_in_png • embeds workflow/prompt/app metadata into PNG tEXt.
- metadata fields (record-only; pixels are not changed):
  width, height, scale, denoise, guidance, clip_skip, steps, seed_value, sampler_name, scheduler_name, positive, negative, model_name

⚙️ KEY BEHAVIOR
- Single image saves: tokenized filename; unique counter appended.
- Grid saves: grid image saved only if save_as_grid_if_multi = ON and there are ≥2 images.
- Metadata (PNG tEXt "parameters" like Automatic1111): includes all params + unlimited loras_infos.

🔑 TOKENS (usable in filename and path)
- Time/date: %date, %date_dash, %time, %datetime
- Params: %seed/%seed_value, %model/%model_name/%basemodelname, %sampler/%sampler_name, %scheduler/%scheduler_name,
  %width, %height, %steps, %cfg/%guidance, %scale, %denoise, %clip_skip
- Batch info: %img_count (grid/batch size)
- LoRAs: %lora_name_1..N, %lora_weight_1..N

📤 OUTPUTS
- GRID_IMAGE • the grid as (1,H,W,3) (or 1×1 if single input).
- last_saved_path • path of the **last** saved file (single image or grid). Empty if nothing saved.
- help • this message.

📝 NOTES & TIPS
- This node does **not** resize; width/height/steps/etc. are **recorded only**.
- Mapping is disabled on `images`: the node runs once for a whole list/batch.
- Set `save_single_image=False` if you only want the grid file.
"""

INVALID_FS_CHARS = r'<>:"|?*'

# ==========================
# Parseurs pour nouveaux inputs
# ==========================

def _escape_comma_split(s: str) -> List[str]:
    parts = []
    buf = []
    escape = False
    for ch in s:
        if escape:
            buf.append(ch)
            escape = False
        else:
            if ch == '\\':
                escape = True
            elif ch == ',':
                parts.append(''.join(buf))
                buf = []
            else:
                buf.append(ch)
    parts.append(''.join(buf))
    return parts

def parse_loras_infos_unified(s: str) -> List[Tuple[str, float]]:
    if not s:
        return []
    # Heuristique: si ":" présent ou "\," utilisé → Syntaxe A
    if (":" in s) or ("\\," in s):
        items = _escape_comma_split(s)
        out: List[Tuple[str, float]] = []
        for raw in items:
            part = raw.strip()
            if not part:
                continue
            if ":" in part:
                name, w = part.split(":", 1)
                name = name.strip()
                try:
                    weight = float(w.strip())
                except:
                    weight = 1.0
            else:
                name, weight = part, 1.0
            if name:
                out.append((name, weight))
        return out
    # Sinon, tenter Syntaxe B: name,weight,name2,weight2,...
    tokens = [t.strip() for t in s.split(",")]
    if len(tokens) % 2 == 0 and len(tokens) > 0:
        out2: List[Tuple[str, float]] = []
        i = 0
        while i + 1 < len(tokens):
            name = tokens[i]
            w_raw = tokens[i+1]
            if name:
                try:
                    w = float(w_raw)
                except:
                    w = 1.0
                out2.append((name, w))
            i += 2
        if out2:
            return out2
    # Fallback: liste de noms poids 1.0
    names = [t.strip() for t in s.split(",") if t.strip()]
    return [(n, 1.0) for n in names]

# A1111 tail parsing
_A_RE_LORA_NAME = re.compile(r'\bLoRA_(\d+)\s*:\s*([^,\n]+)', re.IGNORECASE)
_A_RE_LORA_WEIGHT = re.compile(r'\bLoRA_(\d+)_Weight\s*:\s*([0-9.]+)', re.IGNORECASE)
_A_RE_STEPS = re.compile(r'\bSteps\s*:\s*(\d+)', re.IGNORECASE)
_A_RE_SAMPLER = re.compile(r'\bSampler\s*:\s*([^,]+)', re.IGNORECASE)
_A_RE_SCHEDULER = re.compile(r'\bScheduler\s*:\s*([^,]+)', re.IGNORECASE)
_A_RE_GUIDANCE = re.compile(r'\bGuidance\s*:\s*([0-9.]+)', re.IGNORECASE)
_A_RE_SCALE = re.compile(r'\bScale\s*:\s*([0-9.]+)', re.IGNORECASE)
_A_RE_DENOISE = re.compile(r'\bDenoise\s*:\s*([0-9.]+)', re.IGNORECASE)
_A_RE_SEED = re.compile(r'\bSeed\s*:\s*([-\d]+)', re.IGNORECASE)
_A_RE_SIZE = re.compile(r'\bSize\s*:\s*(\d+)\s*x\s*(\d+)', re.IGNORECASE)
_A_RE_CLIPSKIP = re.compile(r'\bClip\s*skip\s*:\s*(\d+)', re.IGNORECASE)
_A_RE_MODEL = re.compile(r'\bModel\s*:\s*([^,]+)', re.IGNORECASE)

def parse_a1111_tail(tail: str) -> Dict[str, Any]:
    d: Dict[str, Any] = {}
    if not tail:
        return d
    m = _A_RE_STEPS.search(tail)
    if m: d["steps"] = int(m.group(1))
    m = _A_RE_SAMPLER.search(tail)
    if m: d["sampler_name"] = m.group(1).strip()
    m = _A_RE_SCHEDULER.search(tail)
    if m: d["scheduler_name"] = m.group(1).strip()
    m = _A_RE_GUIDANCE.search(tail)
    if m:
        try: d["guidance"] = float(m.group(1))
        except: pass
    m = _A_RE_SCALE.search(tail)
    if m:
        try: d["scale"] = float(m.group(1))
        except: pass
    m = _A_RE_DENOISE.search(tail)
    if m:
        try: d["denoise"] = float(m.group(1))
        except: pass
    m = _A_RE_SEED.search(tail)
    if m:
        try: d["seed_value"] = int(m.group(1))
        except: pass
    m = _A_RE_SIZE.search(tail)
    if m:
        d["width"] = int(m.group(1))
        d["height"] = int(m.group(2))
    m = _A_RE_CLIPSKIP.search(tail)
    if m:
        try: d["clip_skip"] = int(m.group(1))
        except: pass
    m = _A_RE_MODEL.search(tail)
    if m: d["model_name"] = m.group(1).strip()
    # LoRA_N / LoRA_N_Weight → d["loras_infos"]
    name_by_idx: Dict[int, str] = {}
    for m2 in _A_RE_LORA_NAME.finditer(tail):
        try:
            idx = int(m2.group(1))
            name_by_idx[idx] = m2.group(2).strip()
        except Exception:
            pass
    w_by_idx: Dict[int, float] = {}
    for m2 in _A_RE_LORA_WEIGHT.finditer(tail):
        try:
            idx = int(m2.group(1))
            w_by_idx[idx] = float(m2.group(2))
        except Exception:
            pass
    if name_by_idx:
        pairs: List[Tuple[str, float]] = []
        for idx in sorted(name_by_idx.keys()):
            nm = name_by_idx[idx]
            wt = w_by_idx.get(idx, 1.0)
            if nm:
                pairs.append((nm, wt))
        if pairs:
            d["loras_infos"] = pairs
    return d

def _pick_for_index(value: Any, idx: int) -> Any:
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return None
        return value[idx % len(value)]
    return value

# ==========================
# Node
# ==========================
class FRED_ImageSaver_v2:
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (False, False, False)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "images to save (list or batch)."}),
                "save_single_image": ("BOOLEAN", {"default": True, "tooltip": "Save each input image as its own file."}),
                "filename": ("STRING", {"default": "%lora_name_1_%lora_weight_1_%model_name_%date_dash", "multiline": False, "tooltip": "base filename (counter appended)"}),
                "path": ("STRING", {"default": "Fred_nodes/%date_dash/", "multiline": False, "tooltip": "relative to Comfy output or absolute"}),
                "time_format": ("STRING", {"default": "%Y-%m-%d-%H%M%S", "tooltip": "used by %datetime token"}),
                "extension": (["png", "jpeg", "webp"], {"default": "png", "tooltip": "output file format for single images"}),
                "optimize_png": ("BOOLEAN", {"default": False, "tooltip": "optimize PNG (lossless, smaller files but slower)"}),
                "lossless_webp": ("BOOLEAN", {"default": True, "tooltip": "save WEBP in lossless mode (quality ignored)"}),
                "quality_jpeg_or_webp": ("INT", {"default": 100, "min": 1, "max": 100, "tooltip": "JPEG/WEBP quality (ignored if WEBP lossless)"}),
                "save_workflow_as_json": ("BOOLEAN", {"default": False, "tooltip": "write a sidecar JSON next to image"}),
                "embed_workflow_in_png": ("BOOLEAN", {"default": True, "tooltip": "embed workflow/prompt JSON into PNG metadata"}),
                # Grid options
                "save_as_grid_if_multi": ("BOOLEAN", {"default": False, "tooltip": "Also save a single tiled grid image of all inputs."}),
                "grid_filename": ("STRING", {"default": "%lora_name_1_%lora_weight_1_%date_dash_%model_name_grid%_img_count", "multiline": False, "tooltip": "filename prefix for the grid image"}),
                "grid_path": ("STRING", {"default": "Test/%model_name/Grid/%date_dash/", "multiline": False, "tooltip": "save folder for the grid image"}),
                "grid_column_max": ("INT", {"default": 5, "min": 1, "step": 1, "tooltip": "xsize (columns)."}),
                "grid_row_max": ("INT", {"default": 2, "min": 1, "step": 1, "tooltip": "ysize (rows)."}),
                "grid_extension": (["png", "jpeg", "webp"], {"default": "png", "tooltip": "output file format for the grid image"}),
                "grid_quality_jpeg_or_webp": ("INT", {"default": 100, "min": 1, "max": 100, "tooltip": "JPEG/WEBP quality for the grid image"}),
                # Manual parameters following
                "positive": ("STRING", {"default": "", "multiline": False, "tooltip": "positive prompt (saved in metadata) note: will not come from parameters"}),
                "negative": ("STRING", {"default": "", "multiline": False, "tooltip": "negative prompt (saved in metadata) note: will not come from parameters"}),
                "use_parameters_input": ("BOOLEAN", {"default": False, "tooltip": "If ON, 'parameters' overwrite following widgets if exist"}),
                "width": ("INT", {"default": 1024, "min": 1, "max": 16384, "tooltip": "recorded in metadata (does not resize)"}),
                "height": ("INT", {"default": 1024, "min": 1, "max": 16384, "tooltip": "recorded in metadata (does not resize)"}),
                "scale": ("FLOAT", {"default": 1.40, "min": 0.0, "max": 100000.0, "step": 0.01, "tooltip": "record-only parameter"}),
                "denoise": ("FLOAT", {"default": 0.60, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "record-only parameter"}),
                "guidance": ("FLOAT", {"default": 2.20, "min": 0.0, "max": 100.0, "step": 0.01, "tooltip": "record-only parameter (CFG/guidance)"}),
                "clip_skip": ("INT", {"default": 1, "min": -24, "max": 24, "step": 1, "tooltip": "record-only parameter"}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 4096, "tooltip": "record-only parameter"}),
                "seed_value": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "record-only parameter (seed)"}),
                "sampler_name": ("STRING", {"default": "euler", "multiline": False, "tooltip": "record-only parameter (sampler name)"}),
                "scheduler_name": ("STRING", {"default": "simple", "multiline": False, "tooltip": "record-only parameter (scheduler name)"}),
                "model_name": ("STRING", {"default": "flux", "tooltip": "record-only parameter (model name)"}),

            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID",
            },
            "optional": {
                "parameters": ("STRING", {"default": "", "forceInput": True, "tooltip": "A1111 tail (Steps, Sampler, ...). When 'use_parameters_input' is ON, overrides individual fields"}),
                "loras_infos": ("STRING", {"default": "", "forceInput": True, "tooltip": "name[:weight],name2[:weight2],... or name,weight,name2,weight2,...; \\, for literal comma"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("GRID_IMAGE", "last_saved_path", "help")
    OUTPUT_TOOLTIPS = ("", "path to last saved image", "help / usage tips")
    FUNCTION = "save"
    CATEGORY = "👑 FRED/image"
    DESCRIPTION = "Save images with A1111-style parameters text and robust tokenized filenames/paths (v2: parameters input with explicit override switch, unlimited loras_infos, multi-image list mapping)."
    OUTPUT_NODE = True

    @staticmethod
    def IS_CHANGED(**kwargs):
        drop = {"images", "prompt", "extra_pnginfo", "unique_id"}
        clean = {k: v for k, v in kwargs.items() if k not in drop}
        return json.dumps(clean, sort_keys=True, ensure_ascii=False)

    def scalar(self, x):
        return x[0] if isinstance(x, (list, tuple)) and x else x

    def to_int(self, x, default=0):
        x = self.scalar(x)
        try:
            return int(x)
        except Exception:
            try:
                return int(float(x))
            except Exception:
                return int(default)

    def to_float(self, x, default=0.0):
        x = self.scalar(x)
        try:
            return float(x)
        except Exception:
            return float(default)

    def save(self, images, save_single_image, filename, path, time_format, extension,
             optimize_png, lossless_webp, quality_jpeg_or_webp, save_workflow_as_json, embed_workflow_in_png,
             use_parameters_input,
             width, height, scale, denoise, guidance, clip_skip, steps, seed_value,
             sampler_name, scheduler_name, positive, negative, model_name,
             save_as_grid_if_multi, grid_filename, grid_path, grid_column_max, grid_row_max,
             grid_extension, grid_quality_jpeg_or_webp,
             prompt=None, extra_pnginfo=None, unique_id=None, parameters=None, loras_infos=None):

        # Coercion scalaires
        use_parameters_input = bool(self.scalar(use_parameters_input))
        parameters = self.scalar(parameters) or ""
        # fallback rétrocompat: si loras_infos vide, prendre loras
        loras_infos = (self.scalar(loras_infos) or "")

        filename = self.scalar(filename)
        path = self.scalar(path)
        time_format = self.scalar(time_format)
        extension = self.scalar(extension)
        optimize_png = bool(self.scalar(optimize_png))
        lossless_webp = bool(self.scalar(lossless_webp))
        quality_jpeg_or_webp = self.to_int(quality_jpeg_or_webp, 100)
        save_workflow_as_json = bool(self.scalar(save_workflow_as_json))
        embed_workflow_in_png = bool(self.scalar(embed_workflow_in_png))
        save_as_grid_if_multi = bool(self.scalar(save_as_grid_if_multi))
        grid_filename = self.scalar(grid_filename)
        grid_path = self.scalar(grid_path)
        grid_extension = self.scalar(grid_extension)
        grid_column_max = self.to_int(grid_column_max, 5)
        grid_row_max = self.to_int(grid_row_max, 1)
        grid_quality_jpeg_or_webp = self.to_int(grid_quality_jpeg_or_webp, 100)

        # Normaliser images → liste de tensors HWC float[0..1]
        imglist: List[torch.Tensor] = []
        if isinstance(images, (list, tuple)):
            for im in images:
                if not isinstance(im, torch.Tensor):
                    raise ValueError(f"Unsupported list element type {type(im)}")
                if im.ndim == 4:
                    for i in range(im.shape[0]):
                        imglist.append(im[i])
                elif im.ndim == 3:
                    imglist.append(im)
                else:
                    raise ValueError(f"Unsupported tensor shape {im.shape}")
        elif isinstance(images, torch.Tensor):
            if images.ndim == 4:
                for i in range(images.shape[0]):
                    imglist.append(images[i])
            elif images.ndim == 3:
                imglist.append(images)
            else:
                raise ValueError(f"Unsupported tensor shape {images.shape}")
        else:
            raise ValueError(f"Unsupported images input type {type(images)}")

        if len(imglist) == 0:
            return (torch.zeros(1, 8, 8, 3), "", HELP_MESSAGE)

        # Parse parameters uniquement si le switch est ON
        parsed = parse_a1111_tail(parameters) if (use_parameters_input and parameters) else {}
        set_by_parsed = set(parsed.keys()) if (use_parameters_input and parameters) else set()

        # Base meta: priorité aux parsed si présents et switch ON; sinon widgets
        base_meta = {
            "width": parsed.get("width") if "width" in parsed else self.to_int(width, 1024),
            "height": parsed.get("height") if "height" in parsed else self.to_int(height, 1024),
            "scale": parsed.get("scale") if "scale" in parsed else self.to_float(scale, 1.40),
            "denoise": parsed.get("denoise") if "denoise" in parsed else self.to_float(denoise, 0.60),
            "guidance": parsed.get("guidance") if "guidance" in parsed else self.to_float(guidance, 2.20),
            "clip_skip": parsed.get("clip_skip") if "clip_skip" in parsed else self.to_int(clip_skip, 1),
            "steps": parsed.get("steps") if "steps" in parsed else self.to_int(steps, 20),
            "seed_value": parsed.get("seed_value") if "seed_value" in parsed else self.to_int(seed_value, 0),
            "sampler_name": parsed.get("sampler_name") if "sampler_name" in parsed else self.scalar(sampler_name),
            "scheduler_name": parsed.get("scheduler_name") if "scheduler_name" in parsed else self.scalar(scheduler_name),
            "model_name": parsed.get("model_name") if "model_name" in parsed else self.scalar(model_name),
            "positive": self.scalar(positive) or "",
            "negative": self.scalar(negative) or "",
        }
        # Si le tail a des LoRA et switch ON, prépare-les dans base_meta
        if use_parameters_input and ("loras_infos" in parsed):
            base_meta["loras_infos"] = [{"name": n, "weight": w} for n, w in (parsed.get("loras_infos") or [])]

        # Parse loras_infos (global)
        loras_infos_str = self.scalar(loras_infos) or ""
        loras_infos_global = parse_loras_infos_unified(loras_infos_str)

        # Helper: meta par image avec mapping indexé
        def meta_for_index(i):
            m = dict(base_meta)
            # Ne PAS écraser les champs posés par parsed quand le switch est ON
            for field, inp in [
                ("width", width), ("height", height), ("scale", scale), ("denoise", denoise),
                ("guidance", guidance), ("clip_skip", clip_skip), ("steps", steps), ("seed_value", seed_value),
                ("sampler_name", sampler_name), ("scheduler_name", scheduler_name), ("model_name", model_name),
                ("positive", positive), ("negative", negative)
            ]:
                if field in set_by_parsed:
                    continue
                if isinstance(inp, (list, tuple)):
                    m[field] = _pick_for_index(inp, i)

            # Priorité loras_infos
            if use_parameters_input:
                if "loras_infos" in parsed:
                    m["loras_infos"] = [{"name": n, "weight": w} for n, w in (parsed.get("loras_infos") or [])]
                elif loras_infos_global:
                    m["loras_infos"] = [{"name": n, "weight": w} for n, w in loras_infos_global]
                else:
                    m.setdefault("loras_infos", [])
            else:
                # Option stricte: n'injecter global que si parameters est vide
                if (not parameters) and loras_infos_global:
                    m["loras_infos"] = [{"name": n, "weight": w} for n, w in loras_infos_global]
                else:
                    m.setdefault("loras_infos", [])
            return m

        # Convert tensors → PIL
        imgs_pil = [self.tensor_to_pil(t) for t in imglist]

        # Grid
        rows, cols = self.compute_grid_dims(len(imgs_pil), grid_column_max, grid_row_max)
        capacity = rows * cols
        if len(imgs_pil) > capacity:
            print(f"FRED_ImageSaver_v2: Grid capacity {capacity} cannot fit all {len(imgs_pil)} images. {len(imgs_pil)-capacity} left over.")
            imgs_pil = imgs_pil[:capacity]
        grid_img = self.assemble_grid(imgs_pil, rows, cols)
        grid_tensor = self.pil_to_tensor(grid_img).unsqueeze(0)

        saved_paths = []

        # Save grid si demandé
        if save_as_grid_if_multi and len(imgs_pil) > 1:
            # Construire la meta grid d'abord depuis base_meta
            meta_grid = dict(base_meta)
            meta_grid["img_count"] = len(imgs_pil)
            meta_grid["grid_rows"] = rows
            meta_grid["grid_cols"] = cols

            # Loras pour grid: même logique que meta_for_index (sans écraser ensuite)
            if use_parameters_input:
                if "loras_infos" in parsed:
                    meta_grid["loras_infos"] = [{"name": n, "weight": w} for n, w in (parsed.get("loras_infos") or [])]
                elif loras_infos_global:
                    meta_grid["loras_infos"] = [{"name": n, "weight": w} for n, w in loras_infos_global]
                else:
                    meta_grid.setdefault("loras_infos", [])
            else:
                if (not parameters) and loras_infos_global:
                    meta_grid["loras_infos"] = [{"name": n, "weight": w} for n, w in loras_infos_global]
                else:
                    meta_grid.setdefault("loras_infos", [])

            outdir_grid = self.ensure_outdir(grid_path, meta_grid, time_format)
            base_grid = self.expand_tokens(grid_filename.strip() or "grid", meta_grid, time_format)
            base_grid = self.sanitize_component(base_grid) or "grid"
            savedir, filename_base, counter, subfolder, info = folder_paths.get_save_image_path(base_grid, outdir_grid, grid_img.width, grid_img.height)
            fname = f"{filename_base}_{counter:05d}"
            full = os.path.join(savedir, f"{fname}.{grid_extension}")
            while os.path.exists(full):
                counter += 1
                fname = f"{filename_base}_{counter:05d}"
                full = os.path.join(savedir, f"{fname}.{grid_extension}")

            pnginfo_grid = None
            if grid_extension.lower() == "png" and embed_workflow_in_png:
                pnginfo_grid = self.embed_pnginfo(meta_grid, extra_pnginfo, prompt)

            try:
                if grid_extension.lower() == "png":
                    params = {"optimize": bool(optimize_png)}
                    if pnginfo_grid is not None:
                        params["pnginfo"] = pnginfo_grid
                    grid_img.save(full, format="PNG", **params)
                elif grid_extension.lower() == "jpeg":
                    grid_img.save(full, format="JPEG", quality=int(grid_quality_jpeg_or_webp), optimize=True)
                elif grid_extension.lower() == "webp":
                    if lossless_webp:
                        grid_img.save(full, format="WEBP", lossless=True)
                    else:
                        grid_img.save(full, format="WEBP", quality=int(grid_quality_jpeg_or_webp))
                else:
                    comfy.utils.save_image(grid_img, savedir, fname)
            except Exception as e:
                print(f"FRED_ImageSaver_v2: Grid save error {e}.")
                comfy.utils.save_image(grid_img, savedir, fname)

            if save_workflow_as_json:
                try:
                    sidecar = os.path.join(savedir, f"{fname}.json")
                    with open(sidecar, "w", encoding="utf-8") as f:
                        json.dump(meta_grid, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    print(f"FRED_ImageSaver_v2: Could not write GRID JSON sidecar {e}")

            saved_paths.append(full)

        # Save single images avec mapping par index
        if save_single_image:
            outdir = self.ensure_outdir(path, base_meta, time_format)
            base_template = filename.strip() or "image"
            for idx, img_pil in enumerate(imgs_pil):
                meta_i = meta_for_index(idx)
                # meta_i["loras_infos"] = [{"name": n, "weight": w} for n, w in loras_infos_global]
                base = self.expand_tokens(base_template, meta_i, time_format)
                base = self.sanitize_component(base) or "image"
                savedir, filename_base, counter, subfolder, info = folder_paths.get_save_image_path(base, outdir)
                fname = f"{filename_base}_{counter:05d}"
                full = os.path.join(savedir, f"{fname}.{extension}")
                while os.path.exists(full):
                    counter += 1
                    fname = f"{filename_base}_{counter:05d}"
                    full = os.path.join(savedir, f"{fname}.{extension}")

                pnginfo = None
                if extension.lower() == "png" and embed_workflow_in_png:
                    pnginfo = self.embed_pnginfo(meta_i, extra_pnginfo, prompt)

                try:
                    if extension.lower() == "png":
                        params = {"optimize": bool(optimize_png)}
                        if pnginfo is not None:
                            params["pnginfo"] = pnginfo
                        img_pil.save(full, format="PNG", **params)
                    elif extension.lower() == "jpeg":
                        img_pil.save(full, format="JPEG", quality=int(quality_jpeg_or_webp), optimize=True)
                    elif extension.lower() == "webp":
                        if lossless_webp:
                            img_pil.save(full, format="WEBP", lossless=True)
                        else:
                            img_pil.save(full, format="WEBP", quality=int(quality_jpeg_or_webp))
                    else:
                        comfy.utils.save_image(img_pil, savedir, fname)
                except Exception as e:
                    print(f"FRED_ImageSaver_v2: Save error {e}.")
                    comfy.utils.save_image(img_pil, savedir, fname)

                if save_workflow_as_json:
                    try:
                        sidecar = os.path.join(savedir, f"{fname}.json")
                        wf = self.extract_workflow(extra_pnginfo)
                        if wf is None:
                            print("FRED_ImageSaver_v2: No workflow found in extra_pnginfo, skipping sidecar.")
                        else:
                            with open(sidecar, "w", encoding="utf-8") as f:
                                json.dump(wf, f, ensure_ascii=False, indent=2)
                    except Exception as e:
                        print(f"FRED_ImageSaver_v2: Could not write JSON sidecar {e}")

                saved_paths.append(full)

        return (grid_tensor, saved_paths[-1] if saved_paths else "", HELP_MESSAGE)

    # Helpers
    def tensor_to_pil(self, t: torch.Tensor) -> Image.Image:
        if isinstance(t, torch.Tensor) and t.ndim == 4:
            t = t[0]
        if not isinstance(t, torch.Tensor):
            raise ValueError(f"Unsupported image type {type(t)}")
        if t.dtype in [torch.float16, torch.float32, torch.float64]:
            t = t.clamp(0, 1)
            arr = (t.detach().cpu().numpy() * 255.0).astype(np.uint8)
        else:
            arr = t.detach().clamp(0, 255).byte().cpu().numpy()
        if arr.ndim == 2:
            return Image.fromarray(arr, mode="L").convert("RGB")
        if arr.ndim != 3:
            raise ValueError(f"Unsupported image tensor shape {arr.shape}")
        H, W, C_last = arr.shape
        C_first = arr.shape[0]
        if C_last in [1, 3, 4]:
            if C_last == 1:
                arr3 = np.repeat(arr, 3, axis=2)
                return Image.fromarray(arr3, mode="RGB")
            return Image.fromarray(arr[..., :3], mode="RGB")
        if C_first in [1, 3, 4] and arr.shape[1] == W and arr.shape[2] != C_last:
            arr = np.transpose(arr, (1, 2, 0))
            if arr.shape[2] == 1:
                arr3 = np.repeat(arr, 3, axis=2)
                return Image.fromarray(arr3, mode="RGB")
            return Image.fromarray(arr[..., :3], mode="RGB")
        raise ValueError(f"Unsupported image tensor shape {arr.shape}")

    def pil_to_tensor(self, img: Image.Image) -> torch.Tensor:
        arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(arr)

    def sanitize_component(self, s: str) -> str:
        if s is None:
            return ""
        s = str(s).strip()
        out = "".join(c for c in s if c not in INVALID_FS_CHARS)
        return re.sub(r'_+', '_', out).strip("_")

    def get_comfyui_version(self) -> str:
        try:
            root = os.path.abspath(os.path.join(os.path.dirname(folder_paths.__file__), ".."))
            head_path = os.path.join(root, ".git", "HEAD")
            if os.path.exists(head_path):
                with open(head_path, "r", encoding="utf-8") as f:
                    head = f.read().strip()
                if head.startswith("ref:"):
                    ref = head[5:].strip()
                    ref_path = os.path.join(root, ".git", ref.replace("/", os.sep))
                    if os.path.exists(ref_path):
                        with open(ref_path, "r", encoding="utf-8") as f:
                            commit = f.read().strip()
                        return commit[:7]
                return head[:7]
        except Exception:
            pass
        return ""

    def build_a1111_parameters(self, m: Dict[str, Any]) -> str:
        pos = m.get("positive") or ""
        neg = m.get("negative") or ""
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
        if pos.strip():
            parts.append(pos)
        if neg.strip():
            parts.append(f"Negative prompt: {neg}")

        def fmt_f(x):
            try:
                s = f"{float(x):.3f}"
                s = s.rstrip("0").rstrip(".")
                return s if s else "0"
            except Exception:
                return str(x)

        tail = (
            f"Steps: {steps}",
            f"Sampler: {sampler}" if sampler else None,
            f"Scheduler: {scheduler}" if scheduler else None,
            f"Guidance: {fmt_f(guidance)}" if guidance is not None else None,
            f"Scale: {fmt_f(scale)}" if scale is not None else None,
            f"Denoise: {fmt_f(denoise)}" if denoise is not None else None,
            f"Seed: {seed}" if seed is not None else None,
            f"Size: {w}x{h}" if w and h else None,
            f"Clip skip: {abs(clip_skip)}" if clip_skip != 0 else None,
            f"Model: {model}" if model else None,
        )
        parts.append(", ".join(t for t in tail if t))

        ver = (m.get("app") or {}).get("version") or self.get_comfyui_version()
        parts.append(f"Version: ComfyUI {ver}" if ver else "")

        loras_infos = m.get("loras_infos") or []
        for idx, d in enumerate(loras_infos, start=1):
            name = (d.get("name") or "").strip()
            if not name:
                continue
            weight = d.get("weight")
            try:
                weight = float(weight)
            except Exception:
                weight = 0.0
            parts.append(f"LoRA_{idx}: {name}, LoRA_{idx}_Weight: {fmt_f(weight)}")

        return "\n".join(parts)

    def embed_pnginfo(self, metadata, extra_pnginfo, prompt):
        pnginfo = PngImagePlugin.PngInfo()
        try:
            a111 = self.build_a1111_parameters(metadata)
            pnginfo.add_text("parameters", a111)
            if "prompt" in metadata:
                pnginfo.add_text("prompt", json.dumps(metadata["prompt"]))
            wf = self.extract_workflow(extra_pnginfo)
            if wf is not None:
                pnginfo.add_text("workflow", json.dumps(wf))
            if "app" in metadata:
                pnginfo.add_text("app", json.dumps(metadata["app"]))
        except Exception as e:
            print(f"FRED_ImageSaver_v2: PNGInfo embed failed {e}")
        return pnginfo

    def extract_workflow(self, extra_pnginfo):
        def maybe_json(x):
            if isinstance(x, str):
                try:
                    return json.loads(x)
                except Exception:
                    return x
            return x

        if isinstance(extra_pnginfo, dict):
            return maybe_json(extra_pnginfo.get("workflow"))
        if isinstance(extra_pnginfo, list):
            for item in extra_pnginfo:
                if isinstance(item, dict) and "workflow" in item:
                    return maybe_json(item["workflow"])
        return None

    def expand_tokens(self, s: str, meta: Dict[str, Any], time_format: str) -> str:
        # loras = meta.get("loras_infos") or []
        # for idx, it in enumerate(loras, start=1):
            # tokens[f"%lora_name_{idx}"]  = it.get("name", "")
            # tokens[f"%lora_weight_{idx}"] = str(it.get("weight", ""))
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

        alias_map = {"model": "model_name", "sampler": "sampler_name", "scheduler": "scheduler_name", "seed": "seed_value", "cfg": "guidance"}
        for a, b in alias_map.items():
            if a not in token_map and b in token_map:
                token_map[a] = token_map[b]

        loras_infos_list = meta.get("loras_infos") or []
        for i in range(1, len(loras_infos_list)+1):
            item = loras_infos_list[i-1]
            name_k = f"lora_name_{i}"
            weight_k = f"lora_weight_{i}"
            name_v = str(item.get("name", "")).strip()
            if name_v:
                token_map[name_k] = name_v
                token_map[weight_k] = str(item.get("weight", ""))
            else:
                token_map.pop(weight_k, None)

        for key in sorted(token_map.keys(), key=len, reverse=True):
            val = self.sanitize_component(token_map[key])
            for tok in [f"%{key}", f"%{key}%"]:
                if tok in s:
                    s = s.replace(tok, val)

        s = re.sub(r'%[A-Za-z0-9_]+%?', '', s)
        return s

    def ensure_outdir(self, subdir: str, meta: Dict[str, Any], time_format: str) -> str:
        base = folder_paths.get_output_directory()
        subdir = self.expand_tokens(subdir or "", meta, time_format)
        if subdir and os.path.isabs(subdir):
            parts = []
            drive, tail = os.path.splitdrive(subdir)
            for part in tail.replace("/", "\\").split("\\"):
                if part:
                    parts.append(self.sanitize_component(part))
            outdir = os.path.join(drive + os.sep if drive else os.sep, *parts) if parts else subdir
        else:
            parts = []
            for part in subdir.replace("/", "\\").split("\\"):
                if part:
                    parts.append(self.sanitize_component(part))
            outdir = os.path.join(base, *parts) if parts else base
        outdir = os.path.normpath(outdir)
        os.makedirs(outdir, exist_ok=True)
        return outdir

    def compute_grid_dims(self, n: int, col_max: int, row_max: int) -> Tuple[int, int]:
        if n == 0:
            return (1, 1)
        cols = min(col_max, n)
        rows = max(1, math.ceil(n / cols))
        rows = min(rows, row_max)
        while rows * cols < n and rows < row_max:
            rows += 1
        while rows * cols < n and cols < col_max:
            cols += 1
        rows = max(1, min(rows, row_max))
        cols = max(1, min(cols, col_max))
        return (rows, cols)

    def assemble_grid(self, images_pil: List[Image.Image], rows: int, cols: int) -> Image.Image:
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


NODE_CLASS_MAPPINGS = {
    "FRED_ImageSaver_v2": FRED_ImageSaver_v2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FRED_ImageSaver_v2": "👑 FRED ImageSaver v2",
}