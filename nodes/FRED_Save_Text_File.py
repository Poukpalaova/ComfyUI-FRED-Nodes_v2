import os

HELP_MESSAGE = """
👑 FRED_Save_Text_File

🔹 PURPOSE:
Safely write or append text content to disk, with duplicate-skip and flexible file path options.
Useful for archiving results, logs, or annotations during a ComfyUI workflow.

📥 INPUTS:
- folder_path • destination directory (relative or absolute)
- file_name • the desired file name without extension
- file_extension • dropdown to select .txt, .csv, .json (defaults to .txt)
- mode • "append", "overwrite", "new only"
- terminator • in append mode, add a new line before writing new text
- skip_if_duplicate • in append mode, if the exact text is present in the file, skip writing
- text • the content to write

⚙️ MODES:
- append • content is added to the file (or created if missing)
- overwrite • file is replaced in full by the new content
- new only • only create the file if it does not already exist (otherwise do nothing)

📜 RETURNS (outputs):
- SUCCESS? • True if operation completed as intended (written, skipped as expected), False on error
- STATUS • verbose string ("file appended", "file overwritten", "file created", "text already exist", "file already exist", etc.)
- help • this message

📝 Tips:
- skip_if_duplicate helps avoid clutter in logs or tagged lists.
- Use new only for unintentional overwrite protection.
"""

class FRED_Save_Text_File:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "folder_path": ("STRING", {
                    "default": ".", 
                    "tooltip": "Target folder (relative or absolute). Will be created if missing."
                }),
                "file_name": ("STRING", {
                    "default": "out", 
                    "tooltip": "Output file name without extension"
                }),
                "file_extension": ([".txt", ".csv", ".json"], {
                    "default": ".txt",
                    "tooltip": "File extension to use"
                }),
                "mode": (["append", "overwrite", "new only"], {
                    "tooltip": "How to write: Append (add), Overwrite (replace), or New Only (create if missing)"
                }),
                "terminator": ("BOOLEAN", {
                    "default": True, "label_on": "new line", "label_off": "none",
                    "tooltip": "Add a line break in append mode before writing text",
                    "vykosx.binding": [{
                        "source": "mode",
                        "callback": [{
                            "type": "if",
                            "condition": [{
                                "left": "$source.value",
                                "op": "eq",
                                "right": '"append"'
                            }],
                            "true": [{
                                "type": "set",
                                "target": "$this.disabled",
                                "value": False
                            }],
                            "false": [{
                                "type": "set",
                                "target": "$this.disabled",
                                "value": True
                            }],
                        }]
                    }]
                }),
                "skip_if_duplicate": ("BOOLEAN", {
                    "default": True,
                    "label_on": "skip if already present",
                    "label_off": "always write",
                    "tooltip": "If enabled (append mode), do not write if this text already exists in the target file.",
                    "vykosx.binding": [{
                        "source": "mode",
                        "callback": [{
                            "type": "if",
                            "condition": [{
                                "left": "$source.value",
                                "op": "eq",
                                "right": '"append"'
                            }],
                            "true": [{
                                "type": "set",
                                "target": "$this.disabled",
                                "value": False
                            }],
                            "false": [{
                                "type": "set",
                                "target": "$this.disabled",
                                "value": True
                            }],
                        }]
                    }]
                }),
                "text": ("STRING", {
                    "multiline": True,
                    "tooltip": "Text to write to the file. Right click and convert to input for dynamic content."
                }),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "STRING", "STRING")
    RETURN_NAMES = ("SUCCESS?", "STATUS", "help")
    OUTPUT_TOOLTIPS = (
        "True if file written/skipped as intended, False on error",
        "Operation result: file appended, file overwritten, file created, text already exist, file already exist, error",
        "This message (detailed help/instructions)"
    )
    CATEGORY = "👑FRED/utils"
    FUNCTION = "save_text_file"

    def save_text_file(self, folder_path, file_name, file_extension, mode, terminator, skip_if_duplicate, text):
        try:
            # Append extension if not present
            if not file_name.endswith(file_extension):
                file_name += file_extension

            file_path = os.path.join(folder_path, file_name)
            dir_path = os.path.dirname(file_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)

            file_exists_before = os.path.exists(file_path)

            if mode == "new only":
                if file_exists_before:
                    return (True, "file already exist", HELP_MESSAGE)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(text)
                return (True, "file created", HELP_MESSAGE)

            if mode == "overwrite":
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(text)
                if file_exists_before:
                    return (True, "file overwritten", HELP_MESSAGE)
                else:
                    return (True, "file created", HELP_MESSAGE)

            if mode == "append":
                if skip_if_duplicate and file_exists_before:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    if text in content:
                        return (True, "text already exist", HELP_MESSAGE)
                with open(file_path, "a+", encoding="utf-8") as f:
                    is_append = file_exists_before and os.path.getsize(file_path) > 0
                    if is_append and terminator:
                        f.write("\n")
                    f.write(text)
                if file_exists_before:
                    return (True, "file appended", HELP_MESSAGE)
                else:
                    return (True, "file created", HELP_MESSAGE)

            return (False, "error or invalid mode", HELP_MESSAGE)

        except Exception as e:
            return (False, f"error: {str(e)}", HELP_MESSAGE)

# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {
    "FRED_Save_Text_File": FRED_Save_Text_File,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FRED_Save_Text_File": "👑 FRED_Save_Text_File"
}


# import os

# HELP_MESSAGE = """
# 👑 FRED_Save_Text_File

# 🔹 PURPOSE:
# Safely write or append text content to disk, with duplicate-skip and flexible file path options.
# Useful for archiving results, logs, or annotations during a ComfyUI workflow.

# 📥 INPUTS:
# - folder_path • destination directory (relative or absolute)
# - file_name • the desired file name (with extension)
# - mode • "append", "overwrite", "new only"
# - terminator • in append mode, add a new line before writing new text
# - skip_if_duplicate • in append mode, if the exact text is present in the file, skip writing
# - text • the content to write

# ⚙️ MODES:
# - append • content is added to the file (or created if missing)
# - overwrite • file is replaced in full by the new content
# - new only • only create the file if it does not already exist (otherwise do nothing)

# 📜 RETURNS (outputs):
# - SUCCESS? • True if operation completed as intended (written, skipped as expected), False on error
# - STATUS • verbose string ("file appended", "file overwritten", "file created", "text already exist", "file already exist", etc.)
# - help • this message

# 📝 Tips:
# - skip_if_duplicate helps avoid clutter in logs or tagged lists.
# - Use new only for unintentional overwrite protection.
# """

# class FRED_Save_Text_File:
    # @classmethod
    # def INPUT_TYPES(s):
        # return {
            # "required": {
                # "folder_path": ("STRING", {
                    # "default": ".", 
                    # "tooltip": "Target folder (relative or absolute). Will be created if missing."
                # }),
                # "file_name": ("STRING", {
                    # "default": "out.txt", 
                    # "tooltip": "Output file name including extension (e.g. results.txt)"
                # }),
                # "mode": (["append", "overwrite", "new only"], {
                    # "tooltip": "How to write: Append (add), Overwrite (replace), or New Only (create if missing)"
                # }),
                # "terminator": ("BOOLEAN", {
                    # "default": True, "label_on": "new line", "label_off": "none",
                    # "tooltip": "Add a line break in append mode before writing text",
                    # "vykosx.binding": [{
                        # "source": "mode",
                        # "callback": [{
                            # "type": "if",
                            # "condition": [{
                                # "left": "$source.value",
                                # "op": "eq",
                                # "right": '"append"'
                            # }],
                            # "true": [{
                                # "type": "set",
                                # "target": "$this.disabled",
                                # "value": False
                            # }],
                            # "false": [{
                                # "type": "set",
                                # "target": "$this.disabled",
                                # "value": True
                            # }],
                        # }]
                    # }]
                # }),
                # "skip_if_duplicate": ("BOOLEAN", {
                    # "default": True,
                    # "label_on": "skip if already present",
                    # "label_off": "always write",
                    # "tooltip": "If enabled (append mode), do not write if this text already exists in the target file.",
                    # "vykosx.binding": [{
                        # "source": "mode",
                        # "callback": [{
                            # "type": "if",
                            # "condition": [{
                                # "left": "$source.value",
                                # "op": "eq",
                                # "right": '"append"'
                            # }],
                            # "true": [{
                                # "type": "set",
                                # "target": "$this.disabled",
                                # "value": False
                            # }],
                            # "false": [{
                                # "type": "set",
                                # "target": "$this.disabled",
                                # "value": True
                            # }],
                        # }]
                    # }]
                # }),
                # "text": ("STRING", {
                    # "multiline": True,
                    # "tooltip": "Text to write to the file. Right click and convert to input for dynamic content."
                # }),
            # }
        # }

    # RETURN_TYPES = ("BOOLEAN", "STRING", "STRING")
    # RETURN_NAMES = ("SUCCESS?", "STATUS", "help")
    # OUTPUT_TOOLTIPS = (
        # "True if file written/skipped as intended, False on error",
        # "Operation result: file appended, file overwritten, file created, text already exist, file already exist, error",
        # "This message (detailed help/instructions)"
    # )
    # CATEGORY = "👑FRED/utils"
    # FUNCTION = "save_text_file"

    # def save_text_file(self, folder_path, file_name, mode, terminator, skip_if_duplicate, text):
        # try:
            # file_path = os.path.join(folder_path, file_name)
            # dir_path = os.path.dirname(file_path)
            # if dir_path:
                # os.makedirs(dir_path, exist_ok=True)

            # # new only
            # if mode == "new only":
                # if os.path.exists(file_path):
                    # return (True, "file already exist", HELP_MESSAGE)
                # with open(file_path, "w", encoding="utf-8") as f:
                    # f.write(text)
                # return (True, "file created", HELP_MESSAGE)

            # # overwrite
            # if mode == "overwrite":
                # with open(file_path, "w", encoding="utf-8") as f:
                    # f.write(text)
                # return (True, "file overwritten", HELP_MESSAGE)

            # # append
            # if mode == "append":
                # exists = os.path.exists(file_path)
                # if skip_if_duplicate and exists:
                    # with open(file_path, "r", encoding="utf-8") as f:
                        # content = f.read()
                    # if text in content:
                        # return (True, "text already exist", HELP_MESSAGE)
                # with open(file_path, "a+", encoding="utf-8") as f:
                    # is_append = exists and os.path.getsize(file_path) > 0
                    # if is_append and terminator:
                        # f.write("\n")
                    # f.write(text)
                # return (True, "file appended", HELP_MESSAGE)

            # return (False, "error or invalid mode", HELP_MESSAGE)

        # except Exception as e:
            # return (False, f"error: {str(e)}", HELP_MESSAGE)

# # Node registration for ComfyUI
# NODE_CLASS_MAPPINGS = {
    # "FRED_Save_Text_File": FRED_Save_Text_File,
# }

# NODE_DISPLAY_NAME_MAPPINGS = {
    # "FRED_Save_Text_File": "👑 FRED_Save_Text_File"
# }
