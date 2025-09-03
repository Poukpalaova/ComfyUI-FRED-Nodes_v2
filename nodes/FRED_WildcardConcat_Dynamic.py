from __future__ import annotations
import os, random, json, time
from typing import List, Dict, Any
from comfy.comfy_types.node_typing import ComfyNodeABC, InputTypeDict, IO
from server import PromptServer
from aiohttp import web
from pathlib import Path

# Default fallback if widget not provided (kept for safety/back-compat)
SEP = " "

HELP_MESSAGE = """
👑 FRED_WildcardConcat_Dynamic

🔹 PURPOSE:
Build dynamic prompts by concatenating multiple wildcard blocks together.  
Each block can be toggled, weighted, and combined with prefix/suffix text, making this node 
similar to “Power LoRA” mixed with “Creaprompt” but for wildcards.

📥 INPUTS:
- wildcards_dir • base folder containing wildcard text files
- string_delimiter • delimiter placed between text parts (default = space)
- buttons
  • ➕ Add Wildcard → adds a row: toggle, file, weight, line combo, suffix, remove.
  • 🧹 Clear all → romove all added wildcards
- Toggle all
- prefix • optional string prepended before all blocks
- Each block:
  • on/off toggle
  • file (which wildcard file to pick lines from)
  • line (rule for selecting line: random, sequential, fixed, etc.)
  • weight (scaling factor applied to that block)
  • ❌ Remove button
- suffix • optional string appended after all blocks

⚙️ KEY OPTIONS:
- Blocks can be added/removed dynamically in the UI.
- Wildcard files are read from the selected directory.
- Wildcard are in .txt file and easy to edit.
- You can use your own list and adding them to the wildcards folder within this custom_nodes folder
- Each block expands independently; weighted blocks allow fine control.
- string_delimiter ensures clean spacing (can be customized, default is a space).
- Output is deterministic or random depending on “line” selection.
- The random line is added by the code, no need to add it in the .txt file.

📤 OUTPUTS:
- result_string • final concatenated string (prefix + blocks + suffix)
- help • this message

📝 Notes:
- Use consistent wildcard file structure (one entry per line).
- Weights are included numerically in the string only if block “on” is True.
- Changing delimiter also affects suffix join.
- Designed for flexible prompt building with many interchangeable parts.
"""


def _node_dir() -> str:
    return os.path.dirname(__file__)

def _wildcards_dir() -> str:
    # 1) User override (optional)
    override = os.getenv("FRED_WILDCARDS_DIR")
    if override and Path(override).is_dir():
        return override

    here = Path(__file__).resolve()

    # 2) Typical repo root wildcards: ComfyUI-FRED-Nodes/wildcards
    root_wildcards = here.parent.parent / "wildcards"
    if root_wildcards.is_dir():
        return str(root_wildcards)

    # 3) Fallback: a local folder next to the .py (nodes/wildcards)
    local_wildcards = here.parent / "wildcards"
    if local_wildcards.is_dir():
        return str(local_wildcards)

    # 4) Last resort: return the repo-root location even if missing (UI will show none)
    return str(root_wildcards)

def _read_lines(path: str) -> List[str]:
    out: List[str] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                out.append(s)
    except Exception as e:
        print(f"[FRED_WildcardConcat_Dynamic] Cannot read {path}: {e}")
    return out

# ---- tiny API for the UI ----
@PromptServer.instance.routes.get("/fred/wildcards/files")
async def fred_list_wildcard_files(request):
    folder = _wildcards_dir()
    files = []
    try:
        if os.path.isdir(folder):
            for fn in sorted(os.listdir(folder)):
                full = os.path.join(folder, fn)
                if os.path.isfile(full) and fn.lower().endswith(".txt"):
                    files.append(fn)
    except Exception as e:
        print(f"[FRED] files error: {e}")
    return web.json_response({"files": files})

@PromptServer.instance.routes.get("/fred/wildcards/lines")
async def fred_list_wildcard_lines(request):
    fname = request.rel_url.query.get("file", "") or ""
    fname = os.path.basename(fname)
    full = os.path.join(_wildcards_dir(), fname)
    lines = _read_lines(full) if os.path.exists(full) else []
    return web.json_response({"file": fname, "lines": ["random"] + lines})
# --------------------------------

class FRED_WildcardConcat_Dynamic(ComfyNodeABC):
    """
    UI packs rows into rows_json: list of dicts:
      { "on": bool, "file": "people.txt", "line": "random"|"<line>"|"", "weight": float, "suffix": str }

    Backend:
      - Skips empty lines.
      - "random" picks from non-empty file lines.
      - Weight != 1 → "(line:W)" with TWO decimals.
      - Appends per-row suffix after formatting, with delimiter when suffix is not empty.
      - Joins blocks with user 'string_delimiter'.
      - Optional 'prefix' is placed before the first block using the same delimiter (no auto-dot).
    """

    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                # keep order here; canvas JS can still re-order visually
                "wildcards_dir": (IO.STRING, {
                    "default": _wildcards_dir(),
                    "multiline": False,
                    "tooltip": "Folder with .txt files (defaults to this node's ./wildcards)."
                }),
                "string_delimiter": (IO.STRING, {
                    "default": " ",
                    "multiline": False,
                    "tooltip": "String placed between blocks and between text and suffix (e.g. space, comma+space, newline)."
                }),
                "prefix": (IO.STRING, {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Placed before the first block using the same delimiter."
                }),
                "rows_json": (IO.STRING, {
                    "default": "[]",
                    "multiline": True,
                    "tooltip": "Managed by UI. JSON array of rows."
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("result", "help")
    FUNCTION = "concat"
    CATEGORY = "👑FRED/text"

    def concat(self, wildcards_dir: str, string_delimiter: str, prefix: str, rows_json: str):
        base_dir = wildcards_dir or _wildcards_dir()
        delim = string_delimiter if string_delimiter is not None else SEP
        # normalize delimiter (allow things like "\n" to be typed)
        try:
            # safely decode common escapes without eval
            delim = bytes(delim, "utf-8").decode("unicode_escape")
        except Exception:
            pass

        pfx = (prefix or "").strip()

        try:
            rows = json.loads(rows_json or "[]")
        except Exception as e:
            print(f"[FRED_WildcardConcat_Dynamic] rows_json parse error: {e}")
            rows = []

        blocks: List[str] = []
        for row in rows if isinstance(rows, list) else []:
            try:
                if not row.get("on", True):
                    continue
                file_name = (row.get("file") or "").strip()
                chosen    = (row.get("line") or "").strip()
                weight    = float(row.get("weight") or 1.0)
                suffix    = str(row.get("suffix") or "").strip()

                if not file_name:
                    continue

                path = os.path.join(base_dir, os.path.basename(file_name))
                lines = _read_lines(path)
                if not lines:
                    continue

                if "random" in chosen.lower():
                    pool = [l for l in lines if l.strip()]
                    chosen = random.choice(pool) if pool else ""

                if not chosen:
                    continue

                # format weight
                formatted = f"({chosen}:{weight:.2f})" if abs(weight - 1.0) > 1e-9 else chosen

                # ensure delimiter between chosen text and suffix (if suffix present)
                if suffix:
                    formatted = f"{formatted}{delim}{suffix}"

                blocks.append(formatted)
            except Exception as e:
                print(f"[FRED_WildcardConcat_Dynamic] row error: {e}")

        # assemble final
        if not blocks:
            return ("", HELP_MESSAGE)

        if pfx:
            first = f"{pfx}{delim}{blocks[0]}"
            out = first if len(blocks) == 1 else f"{first}{delim}{delim.join(blocks[1:])}"
        else:
            out = delim.join(blocks)

        return (out, HELP_MESSAGE)

    @staticmethod
    def IS_CHANGED(*args, **kwargs):
        import json, time

        # Respect new argument order
        wildcards_dir    = kwargs.get("wildcards_dir",    args[0] if len(args) > 0 else "")
        string_delimiter = kwargs.get("string_delimiter", args[1] if len(args) > 1 else " ")
        prefix           = kwargs.get("prefix",           args[2] if len(args) > 2 else "")
        rows_json        = kwargs.get("rows_json",        args[3] if len(args) > 3 else "[]")

        try:
            rows = json.loads(rows_json or "[]")
        except Exception:
            rows = []

        cleaned = []
        random_present = False
        if isinstance(rows, list):
            for r in rows:
                on     = bool(r.get("on", True))
                file   = (r.get("file") or "").strip()
                line   = (r.get("line") or "").strip()
                weight = float(r.get("weight") or 1.0)
                suffix = str(r.get("suffix") or "")
                is_random = ("random" in line.lower())
                cleaned.append({
                    "on": on, "file": file,
                    "line": "random" if is_random else line,
                    "weight": round(weight, 2),
                    "suffix": suffix,
                })
                if on and is_random:
                    random_present = True

        state = {
            "wildcards_dir": wildcards_dir,
            "string_delimiter": string_delimiter,
            "prefix": prefix,
            "rows": cleaned,
        }
        if random_present:
            state["nonce"] = time.time_ns()

        return json.dumps(state, sort_keys=True, ensure_ascii=False)

NODE_CLASS_MAPPINGS = {
    "FRED_WildcardConcat_Dynamic": FRED_WildcardConcat_Dynamic
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FRED_WildcardConcat_Dynamic": "👑 FRED Wildcard Concat (Dynamic)"
}