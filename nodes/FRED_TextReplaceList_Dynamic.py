from __future__ import annotations
import os, json, re, time
from pathlib import Path
from typing import List
from comfy.comfy_types.node_typing import ComfyNodeABC, InputTypeDict, IO
from server import PromptServer
from aiohttp import web

HELP_MESSAGE = """
👑 FRED_TextReplaceList_Dynamic

🔹 PURPOSE:
Replace words or phrases in a prompt using external .txt lists. The node uses
one dynamic row per replacement rule, just like FRED_WildcardConcat_Dynamic.

📥 INPUTS:
- replacement_dir • shared folder containing .txt list files
- text • prompt or text to process
- rows_json • managed automatically by the JavaScript interface
- Each dynamic row:
  • on/off
  • list file
  • replacement text
  • exact or whole-word matching
  • case-sensitive toggle
- ➕ Add Replacement • adds a new replacement row.
- 🧹 Clear all • removes all rows.

⚙️ LIST FILES:
- Put one word or phrase per line in a .txt file.
- Blank lines and lines beginning with # are ignored.
- The node sorts entries by decreasing length before replacing them.
- Therefore "digital illustration" is handled before "illustration".
- Replacement rows themselves are processed from top to bottom.

📤 OUTPUTS:
- result • processed text
- help • this message

📝 MATCH MODES:
- exact: replaces the exact text anywhere (can match inside words).
- whole_word: only replaces when the text is a standalone word or phrase.
  Use this mode to avoid replacing "design" inside "designs" or "designed".
"""


def _replacement_dir() -> str:
    override = os.getenv("FRED_WILDCARDS_DIR")
    if override and Path(override).is_dir():
        return override
    here = Path(__file__).resolve()
    root_folder = here.parent.parent / "wildcards"
    if root_folder.is_dir():
        return str(root_folder)
    local_folder = here.parent / "wildcards"
    if local_folder.is_dir():
        return str(local_folder)
    return str(root_folder)


def _read_lines(path: str) -> List[str]:
    values = []
    try:
        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                value = line.strip()
                if value and not value.startswith("#"):
                    values.append(value)
    except Exception as error:
        print(f"[FRED_TextReplaceList_Dynamic] Cannot read {path}: {error}")
    return values


@PromptServer.instance.routes.get("/fred/text_replace/files")
async def fred_list_replace_files(request):
    folder = _replacement_dir()
    files = []
    try:
        if os.path.isdir(folder):
            for filename in sorted(os.listdir(folder)):
                full_path = os.path.join(folder, filename)
                if os.path.isfile(full_path) and filename.lower().endswith(".txt"):
                    files.append(filename)
    except Exception as error:
        print(f"[FRED_TextReplaceList_Dynamic] files error: {error}")
    return web.json_response({"files": files})


class FRED_TextReplaceList_Dynamic(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "replacement_dir": (IO.STRING, {
                    "default": _replacement_dir(),
                    "multiline": False,
                    "tooltip": "Folder with .txt lists; defaults to the shared ./wildcards folder."
                }),
                "text": (IO.STRING, {
                    "default": "",
                    "multiline": True,
                    "forceInput": True,
                    "tooltip": "Prompt or text to process."
                }),
                "rows_json": (IO.STRING, {
                    "default": "[]",
                    "multiline": True,
                    "tooltip": "Managed automatically by the dynamic JavaScript interface."
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("result", "help")
    FUNCTION = "replace_text"
    CATEGORY = "👑FRED/text"

    def replace_text(self, replacement_dir: str, text: str, rows_json: str):
        result = text or ""
        base_dir = replacement_dir or _replacement_dir()
        try:
            rows = json.loads(rows_json or "[]")
        except Exception as error:
            print(f"[FRED_TextReplaceList_Dynamic] rows_json parse error: {error}")
            rows = []

        for row in rows if isinstance(rows, list) else []:
            try:
                if not row.get("on", True):
                    continue
                filename = os.path.basename(str(row.get("file") or "").strip())
                replacement = str(row.get("replacement") or "")
                mode = str(row.get("mode") or "exact").lower()
                case_sensitive = bool(row.get("case_sensitive", False))
                if not filename:
                    continue

                entries = _read_lines(os.path.join(base_dir, filename))
                # Critical: longest phrases first, regardless of file order.
                entries.sort(key=lambda value: (-len(value), value.casefold()))
                flags = 0 if case_sensitive else re.IGNORECASE

                for entry in entries:
                    escaped = re.escape(entry)
                    # For whole-word mode, use word boundaries
                    if mode == "whole_word":
                        # \b matches word boundaries (between word and non-word characters)
                        pattern = rf"\b{escaped}\b"
                    else:
                        # Exact mode: replace the exact text, but still use word boundaries
                        # to avoid partial matches inside longer words
                        pattern = rf"(?<!\w){escaped}(?!\w)"
                    result = re.sub(pattern, lambda match: replacement, result, flags=flags)
            except Exception as error:
                print(f"[FRED_TextReplaceList_Dynamic] row error: {error}")

        return (result, HELP_MESSAGE)

    @staticmethod
    def IS_CHANGED(*args, **kwargs):
        return time.time_ns()


NODE_CLASS_MAPPINGS = {
    "FRED_TextReplaceList_Dynamic": FRED_TextReplaceList_Dynamic
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FRED_TextReplaceList_Dynamic": "👑 FRED Text Replace List (Dynamic)"
}