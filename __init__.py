# __init__.py (repo root)

import importlib
import os

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Exact filenames in /nodes
NODE_MODULES = [
    "FRED_AutoCropImage_Native_Ratio",
    "FRED_AutoImageTile_from_Mask",
    "FRED_CropFace",
    "FRED_ImageLoad",
    "FRED_ImageQualityInspector",
    "FRED_ImageSaver",
    "FRED_Simplified_Parameters_Panel",
    "FRED_Text_to_XMP",
    "FRED_WildcardConcat_Dynamic",
    "FRED_Save_Text_File",
    "FRED_ImageUncropFromBBox",
    "FRED_ModelLoaderWithName",
    "FRED_ExtractKSampler_parameter",
    "FRED_Image_Sharpening_Blur_Level",
    "FRED_AutoLoraLoader_Dynamic",
    "FRED_AutoLoraLoader_5_stacks_simple",
    "FRED_ExtractLora_parameter",
    "FRED_ImageSaver_v2",
]

for module_name in NODE_MODULES:
    try:
        mod = importlib.import_module(f".nodes.{module_name}", __name__)
        print(f"[FRED] loaded {module_name} → nodes:",
              list(getattr(mod, "NODE_CLASS_MAPPINGS", {}).keys()))
        NODE_CLASS_MAPPINGS.update(getattr(mod, "NODE_CLASS_MAPPINGS", {}))
        NODE_DISPLAY_NAME_MAPPINGS.update(getattr(mod, "NODE_DISPLAY_NAME_MAPPINGS", {}))
    except Exception as e:
        print(f"[FRED] ERROR loading {module_name}: {e}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

# Expose web assets only if the folder exists (prevents Comfy from mounting a missing dir)
_web_dir = os.path.join(os.path.dirname(__file__), "web")
if os.path.isdir(_web_dir):
    WEB_DIRECTORY = "./web"
    __all__.append("WEB_DIRECTORY")