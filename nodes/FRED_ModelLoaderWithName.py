import os
import torch
import comfy.sd
import folder_paths

class FRED_CheckpointLoaderWithName:
    """Loads Model, CLIP, VAE models with checkpoint name output, optional path/ext removal"""

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "VAE", "model_name")
    FUNCTION = "load_checkpoint"
    CATEGORY = "👑FRED/Model"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), {"tooltip": "Checkpoint"}),
                "strip_path_and_extension": ("BOOLEAN", {"default": True, "tooltip": "Remove path and extension from checkpoint name output"},),
            }
        }

    def load_checkpoint(self, ckpt_name, strip_path_and_extension=True):
        # Get full checkpoint path
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)

        # Load checkpoint models (U-Net, CLIP, VAE)
        model_tuple = comfy.sd.load_checkpoint_guess_config(
            ckpt_path,
            output_vae=True,
            output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )

        # Prepare model_name output with or without path and extension
        model_name_out = ckpt_name
        if strip_path_and_extension:
            model_name_out = os.path.splitext(os.path.basename(ckpt_name))[0]

        # Return models plus model_name as string
        return (*model_tuple[:3], model_name_out)

    @staticmethod
    def IS_CHANGED(*args, **kwargs):
        return False

class FRED_UNETLoaderWithName:
    """Loads U-Net model and outputs it's filename"""
    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "filename")
    OUTPUT_TOOLTIPS = ("U-Net model (denoising latents)", "model filename")
    FUNCTION = "load_unet"

    CATEGORY = "👑FRED/Model"
    DESCRIPTION = "Loads U-Net model and outputs it's filename"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],),
                "strip_path_and_extension": ("BOOLEAN", {"default": True, "tooltip": "Remove path and extension from checkpoint name output"},),
            }
        }

    def load_unet(self, unet_name, weight_dtype, strip_path_and_extension=True):
        model_options = {}
        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2

        # Prepare model_name output with or without path and extension
        model_name_out = unet_name
        if strip_path_and_extension:
            model_name_out = os.path.splitext(os.path.basename(unet_name))[0]

        unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
        return (model, model_name_out)

NODE_CLASS_MAPPINGS = {
    "FRED_CheckpointLoaderWithName": FRED_CheckpointLoaderWithName,
    "FRED_UNETLoaderWithName": FRED_UNETLoaderWithName
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FRED_CheckpointLoaderWithName": "👑 FRED_CheckpointLoaderWithName",
    "FRED_UNETLoaderWithName": "👑 FRED_UNETLoaderWithName"
}