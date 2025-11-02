# -*- coding: utf-8 -*-
# Updated version of FRED_ImageLoad.py that uses source_filename_hint for name/path/meta
from __future__ import annotations

import io
import re
import os
import cv2
import numpy as np
import torch
import hashlib
import folder_paths
import node_helpers
import random
import fnmatch
import json
import time
import pillow_avif  # must be imported before PIL.Image
from PIL import Image, ImageOps, ImageSequence

import sys
from comfy.comfy_types.node_typing import ComfyNodeABC, InputTypeDict, IO

# --- A1111 / SD Prompt Reader–style parsing ---
A1111_PARAM_KEY = "parameters"
NEGATIVE_HEAD = re.compile(r"(^|\n)Negative prompt:\s*", re.IGNORECASE)

ALLOWED_EXT = ('.jpeg', '.jpg', '.png', '.tiff', '.gif', '.bmp', '.webp', '.avif')
CACHE_VERSION = 2

HELP_MESSAGE = """
👑 FRED_LoadImage_V8

🔹 PURPOSE:
Load images from disk or direct input with index management, caching, logging and optional preview.

📥 INPUTS:
- image • input filename or direct image
- Get_metadata_from_file  • will provide the "Positive", "Negative" and generation data if available
- mode • "no_folder" (direct) or "image_from_folder"
- path • folder path
- index • image index
- control after generate
- stop_at_max_index • create an exception that stop iteration with a message that the max index is reached
- next_index_if_any_invalid • skip bad images in current folder and keep a log of it
- include_subdirectories • include subfolders
- show_image_from_path • preview loaded image
- clear_cache • clear cached index/logs. This is required before parsing a very high amount of images. It's faster.
- filename_text_extension • output filename with/without extension
- source_filename • record frontend image name before it get modified by a manual mask getting the image 
  to clipspace and loosing all it metadata.

⚙️ KEY OPTIONS:
- Very efficient node to parse multiple image in multiple folder/subfolder. 
  Tested on a batch of 1 million images without crash and it also skip the broken images and log it for 
  reference if you want to delete them.
- Can show the image behing seen in a folder_mode within this node.
- Show valuables images and folder infos.
- Can work in single image mode (no_folder) or image_from_folder

📤 OUTPUTS:
- IMAGE, MASK, WIDTH, HEIGHT
- INDEX • current index
- TOTAL IMAGES QTY IN ALL FOLDER(S)/SUBFOLDER(S)
- IMAGES QTY IN CURRENT FOLDER
- FOLDER_PATH / FULL_FOLDER_PATH
- filename_text • name without extension (optional)
- skipped_report • list of invalid images
"""

def _resolve_image_path_from_ui(image: str) -> str | None:
    """Resolve what the UI passed into a concrete on-disk path (supports 'pasted/...')."""
    try:
        if hasattr(folder_paths, "exists_annotated_filepath") and folder_paths.exists_annotated_filepath(image):
            return folder_paths.get_annotated_filepath(image)
    except Exception:
        pass
    input_dir = folder_paths.get_input_directory()
    cand = os.path.join(input_dir, image)
    if os.path.exists(cand):
        return cand
    if image.startswith("pasted/"):
        cand = os.path.join(input_dir, image)
        if os.path.exists(cand):
            return cand
    if os.path.isabs(image) and os.path.exists(image):
        return image
    return None

def _resolve_hint_path(hint: str) -> str | None:
    """Resolve source_filename_hint to a file path (absolute, annotated, or inside input dir)."""
    if not hint:
        return None
    # annotated path?
    try:
        if hasattr(folder_paths, "exists_annotated_filepath") and folder_paths.exists_annotated_filepath(hint):
            return folder_paths.get_annotated_filepath(hint)
    except Exception:
        pass
    # absolute?
    if os.path.isabs(hint) and os.path.exists(hint):
        return hint
    # relative to input dir (and supports things like 'pasted/xxx.png')
    inp = folder_paths.get_input_directory()
    cand = os.path.join(inp, hint)
    if os.path.exists(cand):
        return cand
    return None

def _extract_pos_neg_and_params(a111_text: str) -> tuple[str, str, str]:
    if not a111_text:
        return "", "", ""
    s = a111_text.replace("\r\n", "\n").replace("\r", "\n")
    m = NEGATIVE_HEAD.search(s)
    if m:
        positive = s[:m.start()].strip()
        rest = s[m.end():]
    else:
        first, _, tail = s.partition("\n")
        return first.strip(), "", tail.strip()
    steps_idx = rest.find("\nSteps:")
    if steps_idx == -1:
        steps_line = re.search(r"\n\s*Steps\s*:", rest)
        if steps_line:
            steps_idx = steps_line.start()
    if steps_idx != -1:
        negative = rest[:steps_idx].strip()
        params_tail = rest[steps_idx+1:].strip()
    else:
        negative = rest.strip()
        params_tail = ""
    return positive, negative, params_tail

def _read_meta_triplet_from_path(path: str) -> tuple[str, str, str]:
    try:
        pil = Image.open(path)
        info = dict(pil.info) if isinstance(pil.info, dict) else {}
        text = ""
        if isinstance(info.get(A1111_PARAM_KEY), str):
            text = info[A1111_PARAM_KEY]
        elif isinstance(info.get("Description"), str):
            text = info["Description"]
        elif isinstance(info.get("UserComment"), str):
            text = info["UserComment"]
        if not text:
            return "", "", ""
        return _extract_pos_neg_and_params(text)
    except Exception as e:
        print(f"[FRED_ImageLoad] Metadata read warn: {e}")
        return "", "", ""

class _FRED_PreviewHelper:
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_image_preview_" + ''.join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(5))

    def get_unique_filename(self, filename_prefix):
        os.makedirs(self.output_dir, exist_ok=True)
        counter = 1
        while True:
            file = f"{filename_prefix}{self.prefix_append}_{counter:04d}.png"
            full_path = os.path.join(self.output_dir, file)
            if not os.path.exists(full_path):
                return full_path, file
            counter += 1

    def save_image(self, image_tensor, filename_prefix):
        results = []
        try:
            full_path, file = self.get_unique_filename(filename_prefix)
            img = image_tensor[0].clamp(0, 1).mul(255).byte().cpu().numpy()
            img = Image.fromarray(img)
            img.save(full_path)
            results.append({"filename": file, "subfolder": "", "type": self.type})
        except Exception as e:
            print(f"[Preview Error] {e}")
        return {"images": results}

class FRED_ImageLoad(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        input_dir = folder_paths.get_input_directory()
        files = sorted(
            f for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f))
        )
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True}),
                "Get_metadata_from_file": ("BOOLEAN", {"default": True}),
                "mode": (["no_folder", "image_from_folder"],),
                "index": (IO.INT, {"default": 0, "min": 0, "max": 999999999, "control_after_generate": True}),
                "stop_at_max_index": ("INT", {"default": 999999999, "min": 0, "max": 999999999, "step": 1}),
                "next_index_if_any_invalid": ("BOOLEAN", {"default": True}),
                "path": ("STRING", {"default": '', "multiline": False}),
                "include_subdirectories": ("BOOLEAN", {"default": False}),
                "show_image_from_path": ("BOOLEAN", {"default": False}),
                "clear_cache": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "filename_text_extension": (["true", "false"], {"default": "false"}),
                "source_filename_hint": (IO.STRING, {"default": "", "multiline": False, "tooltip": "Auto-filled by the frontend with the dropped filename."}),
            },
        }

    RETURN_TYPES = (
        "IMAGE", "MASK", "FLOAT", "INT", "INT", IO.INT,
        "INT", "INT", "STRING", "STRING", "STRING",
        "STRING",              # skipped_report
        "STRING",              # positive_prompt
        "STRING",              # negative_prompt
        "STRING",              # parameters (tail)
        "STRING"               # help
    )
    RETURN_NAMES = (
        "image",
        "mask",
        "image_size_kb",
        "width",
        "height",
        "index",
        "total_images_qty_in_folder(s)",
        "images_qty_in_current_folder",
        "folder_path",
        "full_folder_path",
        "filename_text",
        "skipped_report",
        "positive_prompt",
        "negative_prompt",
        "parameters",
        "help"
    )

    FUNCTION = "load_image"
    CATEGORY = "👑FRED/image"

    def load_image(self, image, mode="no_folder", index=0, stop_at_max_index=999999999,
                   next_index_if_any_invalid=True, path="", include_subdirectories=False,
                   clear_cache=False, Get_metadata_from_file=True, show_image_from_path=False,
                   filename_text_extension="false", source_filename_hint: str = "", _skipped_list=None):

        if clear_cache:
            self._clear_invalid_log()
            _skipped_list = []
        else:
            if _skipped_list is None:
                _skipped_list = self._load_skipped_list()

        if index >= stop_at_max_index:
            raise RuntimeError(f"Reached stop_at_max_index limit: index={index}, stop_at_max_index={stop_at_max_index}")

        # ----------------------- NO FOLDER MODE -----------------------
        if mode == "no_folder":
            pos, neg, params_tail = ("", "", "")

            if isinstance(image, str):
                image_path = _resolve_image_path_from_ui(image)
                if not image_path or not os.path.exists(image_path):
                    raise FileNotFoundError(f"Cannot resolve input image path: {image!r}")

                # Use hint (if resolvable) for name/path/meta; else use actual chosen image path
                hint_path = _resolve_hint_path(source_filename_hint.strip()) if isinstance(source_filename_hint, str) else None
                info_path = hint_path or image_path

                img = node_helpers.pillow(Image.open, image_path)  # pixels from actual selected image

                # compute size in KB (based on the pixels we loaded)
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                image_size_kb = len(buf.getvalue()) / 1024.0

                if Get_metadata_from_file and info_path and os.path.exists(info_path):
                    try:
                        pos, neg, params_tail = _read_meta_triplet_from_path(info_path)
                    except Exception as e:
                        print(f"[FRED_ImageLoad] Meta parse warn (no_folder): {e}")

                filename = os.path.basename(info_path)
                full_folder_path = os.path.dirname(info_path)

            elif isinstance(image, Image.Image):
                img = image

                # Try to resolve hint or embedded filename for meta + names
                hint_path = _resolve_hint_path(source_filename_hint.strip()) if isinstance(source_filename_hint, str) else None
                pil_src = getattr(image, "filename", None) if hasattr(image, "filename") else None
                info_path = hint_path or (pil_src if isinstance(pil_src, str) and os.path.exists(pil_src) else None)

                # compute size in KB from pixels
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                image_size_kb = len(buf.getvalue()) / 1024.0

                if Get_metadata_from_file:
                    try:
                        if info_path:
                            pos, neg, params_tail = _read_meta_triplet_from_path(info_path)
                        else:
                            info = getattr(image, "info", {}) or {}
                            text = (info.get("parameters")
                                    or info.get("Description")
                                    or info.get("UserComment") or "")
                            if text:
                                pos, neg, params_tail = _extract_pos_neg_and_params(text)
                    except Exception as e:
                        print(f"[FRED_ImageLoad] Meta parse warn (no_folder PIL): {e}")

                # names/paths derived from hint if available; else placeholder
                if info_path:
                    filename = os.path.basename(info_path)
                    full_folder_path = os.path.dirname(info_path)
                else:
                    filename = "direct_image_input"
                    full_folder_path = ""
            else:
                raise ValueError("Invalid image input type.")

            # tensor convert & mask
            # if img.mode == 'RGBA':
                # rgb_image = img.convert('RGB')
                # alpha_channel = img.split()[3]
                # alpha_array = np.array(alpha_channel)
                # is_inverted = np.mean(alpha_array) > 127
                # image_t = np.array(rgb_image).astype(np.float32) / 255.0
                # image_t = torch.from_numpy(image_t)[None,]
                # mask_t = np.array(alpha_channel).astype(np.float32) / 255.0
                # if is_inverted:
                    # mask_t = 1. - mask_t
                # mask_t = torch.from_numpy(mask_t)
            if img.mode == 'RGBA':
                rgb_image = img.convert('RGB')
                alpha_channel = img.split()[3]
                alpha_array = np.array(alpha_channel)
                
                image_t = np.array(rgb_image).astype(np.float32) / 255.0
                image_t = torch.from_numpy(image_t)[None,]
                
                # Simply normalize alpha channel WITHOUT inverted logic
                mask_t = np.array(alpha_channel).astype(np.float32) / 255.0
                mask_t = torch.from_numpy(mask_t)
            else:
                image_t = np.array(img.convert("RGB")).astype(np.float32) / 255.0
                image_t = torch.from_numpy(image_t)[None,]
                mask_t = 1. - torch.ones((img.size[1], img.size[0]), dtype=torch.float32)

            output_image = image_t
            output_mask = mask_t.unsqueeze(0)
            width, height = img.size
            filename_text = filename if filename_text_extension == "true" else os.path.splitext(filename)[0]
            skipped_report = "None"

            # NOTE: return order includes image_size_kb at slot 2
            return (
                output_image, output_mask, float(image_size_kb), width, height, index,
                1, 1,
                full_folder_path,         # folder_path (for no_folder, identical to full)
                full_folder_path,         # full_folder_path
                filename_text,
                skipped_report,
                pos,
                neg,
                params_tail,
                HELP_MESSAGE
            )

        # ----------------------- FOLDER MODE -----------------------
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"Invalid path provided: {path}")

        fl = self.BatchImageLoader(path, include_subdirectories, clear_cache=clear_cache,
                                   enable_hash=Get_metadata_from_file)
        max_value = fl.get_total_image_count()
        if max_value == 0:
            raise RuntimeError("No valid images found in folder.")

        skipped_report = "None"

        try:
            image_path = fl.get_image_path_by_id(index)
            img = Image.open(image_path)
            img = ImageOps.exif_transpose(img)

            # compute size in KB for folder mode as well
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            image_size_kb = len(buf.getvalue()) / 1024.0

            pos, neg, params_tail = ("", "", "")
            if Get_metadata_from_file:
                try:
                    pos, neg, params_tail = _read_meta_triplet_from_path(image_path)
                except Exception as e:
                    print(f"[FRED_ImageLoad] metadata parse (folder) warning: {e}")
        except Exception as e:
            failed_path = os.path.abspath(image_path)

            if failed_path not in _skipped_list:
                _skipped_list.append(failed_path)
                self._log_invalid_path(failed_path)

            if next_index_if_any_invalid:
                index += 1
                result = self.load_image(
                    image, mode, index, stop_at_max_index,
                    next_index_if_any_invalid, path, include_subdirectories,
                    clear_cache, Get_metadata_from_file, show_image_from_path, filename_text_extension,
                    source_filename_hint=source_filename_hint, _skipped_list=_skipped_list
                )

                if isinstance(result, dict):
                    image_data = result["result"]
                    ui = result.get("ui", {})
                else:
                    image_data = result
                    ui = {}

                # Rebuild tuple, overriding folder_path and skipped_report
                final_result = (
                    image_data[0], image_data[1], image_data[2], image_data[3], image_data[4],
                    image_data[5], image_data[6], image_data[7],
                    path,                         # folder_path override (8)
                    image_data[9],                # full_folder_path
                    image_data[10],               # filename_text
                    "\n".join(_skipped_list),     # skipped_report
                    image_data[12],               # positive
                    image_data[13],               # negative
                    image_data[14],               # parameters
                    image_data[15],               # help
                )
                return {**ui, "result": final_result}
            else:
                raise RuntimeError(f"Error loading image: {failed_path} → {e}")

        filename = os.path.basename(image_path)
        current_folder_path = os.path.dirname(image_path)

        image_t = np.array(img.convert("RGB")).astype(np.float32) / 255.0
        image_t = torch.from_numpy(image_t)[None,]
        mask_t = torch.ones((1, img.size[1], img.size[0]), dtype=torch.float32)

        width, height = img.size
        filename_text = filename if filename_text_extension == "true" else os.path.splitext(filename)[0]

        skipped_report = "\n".join(sorted(set(_skipped_list))) if _skipped_list else "None"

        ui_result = {}
        if show_image_from_path:
            previewer = _FRED_PreviewHelper()
            ui_result = {"ui": previewer.save_image(image_t, "loaded_image_preview")}

        # NOTE: return order includes image_size_kb at slot 2
        return {
            **ui_result,
            "result": (
                image_t, mask_t, float(image_size_kb), width, height, index, max_value,
                self.count_images_in_current_folder(current_folder_path),
                path, current_folder_path, filename_text,
                skipped_report,
                pos,
                neg,
                params_tail,
                HELP_MESSAGE + ("[Cache] Rebuilt." if clear_cache else "")
            )
        }

    def count_images_in_current_folder(self, path):
        valid = 0
        for f in os.listdir(path):
            if f.lower().endswith(ALLOWED_EXT):
                full = os.path.join(path, f)
                if os.path.isfile(full):
                    try:
                        Image.open(full).verify()
                        valid += 1
                    except:
                        continue
        return valid

    def _log_invalid_path(self, path):
        try:
            log_dir = os.path.join(os.path.expanduser("~"), ".fred_logs")
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, "invalid_images_log.txt")
            if os.path.exists(log_path):
                with open(log_path, "r", encoding="utf-8") as f:
                    if path in f.read():
                        return
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(path + "\n")
        except Exception as e:
            print(f"[Invalid Image Log Error] {e}")

    def _load_skipped_list(self):
        try:
            log_dir = os.path.join(os.path.expanduser("~"), ".fred_logs")
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, "invalid_images_log.txt")
            if os.path.exists(log_path):
                with open(log_path, "r", encoding="utf-8") as f:
                    return list(set(line.strip() for line in f if line.strip()))
        except Exception as e:
            print(f"[Skipped List Load Error] {e}")
        return []

    def _clear_invalid_log(self):
        try:
            log_path = os.path.join(os.path.expanduser("~"), ".fred_logs", "invalid_images_log.txt")
            if os.path.exists(log_path):
                os.remove(log_path)
                print("[FRED] Cleared invalid image log.")
        except Exception as e:
            print(f"[FRED] Failed to clear invalid image log: {e}")

    class BatchImageLoader:
        def __init__(self, directory_path, include_subdirectories=False, help_log=None, clear_cache=False, enable_hash=True):
            self.directory_path = directory_path
            self.include_subdirectories = include_subdirectories
            self.image_paths = []

            if not enable_hash:
                self._build_index()
                return

            self.cache_file = self._get_cache_path()
            self.dir_hash = self._calculate_directory_hash()
            if clear_cache or not self._load_from_cache():
                self._build_index()
                self._save_to_cache()

        def _get_cache_path(self):
            temp_dir = os.path.join(os.path.expanduser("~"), ".fred_image_cache")
            os.makedirs(temp_dir, exist_ok=True)
            folder_name = os.path.basename(os.path.normpath(self.directory_path))
            full_identifier = f"{self.directory_path}|{folder_name}|{str(self.include_subdirectories)}"
            safe_name = hashlib.md5(full_identifier.encode()).hexdigest()
            return os.path.join(temp_dir, f"{safe_name}.json")

        def _calculate_directory_hash(self):
            try:
                folder_mtime = str(os.path.getmtime(self.directory_path))
                base = self.directory_path + str(self.include_subdirectories) + folder_mtime
            except:
                base = self.directory_path + str(self.include_subdirectories)
            return hashlib.md5(base.encode()).hexdigest()

        def _load_from_cache(self):
            if os.path.exists(self.cache_file):
                try:
                    with open(self.cache_file, 'r') as f:
                        data = json.load(f)
                        if data.get('dir_hash') == self.dir_hash:
                            self.image_paths = data['image_paths']
                            return True
                except:
                    return False
            return False

        def _save_to_cache(self):
            with open(self.cache_file, 'w') as f:
                json.dump({'dir_hash': self.dir_hash, 'image_paths': self.image_paths, 'timestamp': time.time()}, f)

        def _build_index(self):
            self.image_paths = []
            if self.include_subdirectories:
                for root, _, files in os.walk(self.directory_path):
                    for file in files:
                        if file.lower().endswith(ALLOWED_EXT):
                            full = os.path.abspath(os.path.join(root, file))
                            self.image_paths.append(full)
            else:
                files = os.listdir(self.directory_path)
                for file in files:
                    if file.lower().endswith(ALLOWED_EXT):
                        full = os.path.abspath(os.path.join(self.directory_path, file))
                        self.image_paths.append(full)
            self.image_paths.sort()

        def get_total_image_count(self):
            return len(self.image_paths)

        def get_image_path_by_id(self, image_id):
            if not self.image_paths:
                return None
            return self.image_paths[image_id % len(self.image_paths)]

    @staticmethod
    def IS_CHANGED(*args, **kwargs):
        return False

    @classmethod
    def VALIDATE_INPUTS(cls, image, mode="no_folder", path="", **kwargs):
        if mode == "no_folder":
            if not folder_paths.exists_annotated_filepath(image):
                return f"Image not found: {image}"
        elif mode == "image_from_folder":
            if not path or not os.path.exists(path):
                return f"Invalid folder path: {path}"
        return True


NODE_CLASS_MAPPINGS = {
    "FRED_ImageLoad": FRED_ImageLoad
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FRED_ImageLoad": "👑 FRED Image Load (avif support)"
}