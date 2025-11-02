import cv2
from PIL import Image
import numpy as np
import torch
from torchvision.transforms import InterpolationMode
import torch.nn.functional as F
import comfy.utils
from comfy_extras.nodes_mask import ImageCompositeMasked

ASPECT_RATIOS = [
    {"name": "9:21 portrait 640x1536", "width": 640, "height": 1536},
    {"name": "1:2 portrait 768x1536", "width": 768, "height": 1536},
    {"name": "9:16 portrait 768x1344", "width": 768, "height": 1344},
    {"name": "2:3 portrait 1024x1536", "width": 832, "height": 1216},
    {"name": "5:8 portrait 832x1216", "width": 832, "height": 1216},
    {"name": "5:7 portrait 896x1254", "width": 896, "height": 1254},
    {"name": "3:4 portrait 896x1152", "width": 896, "height": 1152},
    {"name": "4:5 portrait 1024x1280", "width": 1024, "height": 1280},
    {"name": "5:6 portrait 1066x1280", "width": 1066, "height": 1280},
    {"name": "9:10 portrait 1152x1280", "width": 1152, "height": 1280},
    {"name": "1:1 square 1024x1024", "width": 1024, "height": 1024},
    {"name": "10:9 landscape 1280x1152", "width": 1280, "height": 1152},
    {"name": "6:5 landscape 1280x1066", "width": 1280, "height": 1066},
    {"name": "5:4 landscape 1280x1024", "width": 1280, "height": 1024},
    {"name": "8:5 landscape 1280x800", "width": 1280, "height": 800},
    {"name": "7:5 landscape 1120x800", "width": 1120, "height": 800},
    {"name": "4:3 landscape 1152x896", "width": 1152, "height": 896},
    {"name": "3:2 landscape 1216x832", "width": 1216, "height": 832},
    {"name": "16:9 wide landscape 1344x768", "width": 1344, "height": 768},
    {"name": "2:1 panorama 1536x768", "width": 1536, "height": 768},
    {"name": "21:9 ultra-wide 1536x640", "width": 1536, "height": 640}
]

HELP_MESSAGE = """
👑 FRED_AutoCropImage_Native_Ratio

🔹 PURPOSE:
Automatically crop and resize images to fit Stable Diffusion standard aspect ratios or custom sizes, while preserving masks if needed.

📥 INPUTS:
- image • main input image
- mask_optional • optional mask used to preserve that portion of the image during crop or to precrop from it
- aspect_ratio • predefined SD ratios, "Auto_find_resolution", "custom", or "no_crop_to_ratio"
- Precrop_from_input_mask • crop image to mask boundaries before ratio crop
- mask_preserve • ensures mask area is not cut off for the choosed resolution
- custom_width / custom_height • manual target dimensions
- crop_from_center / crop_x_in_Percent / crop_y_in_Percent • crop positioning
- resize_image / resize_mode_if_upscale / resize_mode_if_downscale • resize modes
- prescale_factor / include_prescale_if_resize • apply pre-scaling
- multiple_of • round dimensions to multiple of N
- preview_mask_color / preview_mask_color_intensity • preview overlay settings

⚙️ KEY OPTIONS:
- Auto-find: detect closest SDXL ratio
- Custom: force manual width/height
- no_crop_to_ratio: skip aspect ratio crop
- Resize: upscale/downscale modes (bicubic, bilinear, area…)
- Overlay: preview crop + mask in color

📤 OUTPUTS:
- modified_image • cropped/resized result
- preview • preview image with overlay
- modified_mask • resized and/or cropped mask like the image
- scale_factor • wanted value to be applyed to the output image. If include scale_factor is true, the scale_factor output will be 1
- output_width / output_height • final dimensions
- native_width / native_height • target SD ratio dimensions
- sd_aspect_ratios • name of matched ratio
"""


class FRED_AutoCropImage_Native_Ratio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "Precrop_from_input_mask": ("BOOLEAN", {"default": False}),
                "aspect_ratio": (
                    ["custom", "Auto_find_resolution"] +
                    [ar["name"] for ar in ASPECT_RATIOS] +
                    ["no_crop_to_ratio"], {"default": "Auto_find_resolution"}
                ),
                "mask_preserve": ("BOOLEAN", {"default": False}),
                "custom_width": ("INT", {"default": 1024, "min": 64, "max": 8192}),
                "custom_height": ("INT", {"default": 1024, "min": 64, "max": 8192}),
                "crop_from_center": ("BOOLEAN", {"default": True}),
                "crop_x_in_Percent": ("INT", {"default": 0, "min": 0, "max": 100}),
                "crop_y_in_Percent": ("INT", {"default": 0, "min": 0, "max": 100}),
                "resize_image": ("BOOLEAN", {"default": False}),
                "resize_mode_if_upscale": (["bicubic", "bilinear", "nearest", "nearest-exact", "area"], {"default": "bilinear"}),
                "resize_mode_if_downscale": (["bicubic", "bilinear", "nearest", "nearest-exact", "area"], {"default": "area"}),
                "prescale_factor": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 8.0, "step": 0.1}),
                "include_prescale_if_resize": ("BOOLEAN", {"default": False}),
                "multiple_of": (["1", "2", "4", "8", "16", "32", "64"], {"default": "1"}),
                "preview_mask_color_intensity": ("FLOAT", {"default": 0.4, "min": 0.1, "max": 1.0, "step": 0.1}),
                "preview_mask_color": ("COLOR", {"default": "#503555", "widgetType": "MTB_COLOR"},),
            },
            "optional": {
                "mask_optional": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "FLOAT", "INT", "INT", "INT", "INT", "STRING", "STRING")
    RETURN_NAMES = ("modified_image", "preview", "modified_mask", "scale_factor",
                    "output_width", "output_height", "native_width", "native_height",
                    "sd_aspect_ratios", "help")
    FUNCTION = "run"
    CATEGORY = "👑FRED/image/postprocessing"
    OUTPUT_NODE = True

    def run(self, image, Precrop_from_input_mask, aspect_ratio, mask_preserve,
            custom_width, custom_height, crop_from_center, crop_x_in_Percent, crop_y_in_Percent,
            resize_image, resize_mode_if_upscale, resize_mode_if_downscale,
            prescale_factor, include_prescale_if_resize,
            preview_mask_color_intensity, preview_mask_color, multiple_of,
            mask_optional=None):

        _, original_height, original_width, _ = image.shape

        # Precrop from mask
        if Precrop_from_input_mask and mask_optional is not None:
            x_min, y_min, x_max, y_max = self.find_mask_boundaries(mask_optional)
            if x_min is not None:
                # Crop image : on suppose image en format (B, H, W, C)
                image = image[:, y_min:y_max+1, x_min:x_max+1, :]
                # Crop mask : format (B, H, W)
                mask = mask_optional[:, y_min:y_max+1, x_min:x_max+1]
                # Mettre à jour taille pour la suite
                original_height, original_width = (y_max - y_min + 1), (x_max - x_min + 1)
            else:
                # On conserve taille originale si pas de bounding box valide
                original_height, original_width = image.shape[1], image.shape[2]
                mask = mask_optional
        else:
            # Pas de crop depuis mask, taille originale conservée
            original_height, original_width = image.shape[1], image.shape[2]
            if mask_optional is None:
                mask = torch.zeros(1, original_height, original_width, dtype=torch.float32, device=image.device)
            else:
                mask = mask_optional

        # Si le masque a une taille différente de l'image, on redimensionne
        if mask.shape[1] != original_height or mask.shape[2] != original_width:
            mask = torch.nn.functional.interpolate(
                mask.unsqueeze(0).unsqueeze(0),  # (1, 1, H, W)
                size=(original_height, original_width),
                mode="bicubic",
                align_corners=False
            ).squeeze(0).squeeze(0).clamp(0.0, 1.0)

        # Native resolution selection
        if aspect_ratio == "no_crop_to_ratio":
            native_width, native_height = original_width, original_height
            sd_aspect_ratios = "no_crop"
        elif aspect_ratio == "Auto_find_resolution":
            native_width, native_height, sd_aspect_ratios = self.find_closest_sd_resolution(original_width, original_height)
        elif aspect_ratio == "custom":
            native_width, native_height = custom_width, custom_height
            sd_aspect_ratios = f"{custom_width}x{custom_height}"
        else:
            ratio = next(a for a in ASPECT_RATIOS if a["name"] == aspect_ratio)
            native_width, native_height = ratio["width"], ratio["height"]
            sd_aspect_ratios = aspect_ratio

        # Multiple-of adjustment
        m = int(multiple_of)
        native_width = (native_width // m) * m
        native_height = (native_height // m) * m

        # Crop
        if aspect_ratio != "no_crop_to_ratio" and mask_preserve and mask is not None and mask.sum() > 0:
            cropped_image, preview, cropped_mask = self.crop_preserve_mask(
                image, mask, native_width, native_height,
                crop_from_center, crop_x_in_Percent, crop_y_in_Percent,
                preview_mask_color_intensity, preview_mask_color
            )
        else:
            cropped_image, preview = self.crop_image_to_ratio(
                image, native_width, native_height,
                crop_from_center, crop_x_in_Percent, crop_y_in_Percent,
                False, preview_mask_color_intensity, preview_mask_color
            )
            cropped_mask = self.crop_image_to_ratio(
                mask, native_width, native_height,
                crop_from_center, crop_x_in_Percent, crop_y_in_Percent,
                True, preview_mask_color_intensity, preview_mask_color
            )

        # Resize
        _, cropped_height, cropped_width, _ = cropped_image.shape
        if resize_image:
            if include_prescale_if_resize:
                final_width  = int(((native_width  * prescale_factor) // m) * m)
                final_height = int(((native_height * prescale_factor) // m) * m)
                scale_factor = 1.0
            else:
                final_width  = (native_width  // m) * m
                final_height = (native_height // m) * m
                scale_factor = prescale_factor
            resize_mode = resize_mode_if_downscale if final_width < cropped_width or final_height < cropped_height else resize_mode_if_upscale
            modified_image = self.resize_image(cropped_image, resize_mode, final_width, final_height, "center")
            modified_mask = comfy.utils.common_upscale(cropped_mask.unsqueeze(1), final_width, final_height, resize_mode, "center").squeeze(1)
        else:
            desired_w = native_width  * prescale_factor
            desired_h = native_height * prescale_factor
            scale_factor = min(desired_w / cropped_width, desired_h / cropped_height)
            modified_image = cropped_image
            modified_mask = cropped_mask

        # Overlay preview
        if mask is not None and mask.sum() > 0 and mask.sum() != mask.numel():
            preview = self.apply_mask_overlay(preview, mask, preview_mask_color, preview_mask_color_intensity)

        _, output_height, output_width, _ = modified_image.shape

        # return (modified_image, preview, modified_mask, scale_factor,
                # output_width, output_height, native_width, native_height,
                # sd_aspect_ratios, HELP_MESSAGE)
        output = modified_image
        output = output.to(dtype=torch.float32, device=image.device).clamp(0, 1).contiguous()
        return (output, preview, modified_mask, scale_factor,
                output_width, output_height, native_width, native_height,
                sd_aspect_ratios, HELP_MESSAGE)

    # --- New helper ---
    def crop_preserve_mask(self, image, mask, native_width, native_height,
                           crop_from_center, crop_x_percent, crop_y_percent,
                           preview_intensity, preview_color):
        x_min, y_min, x_max, y_max = self.find_mask_boundaries(mask)
        target_aspect_ratio = native_width / native_height

        if (image.shape[2] / image.shape[1]) > target_aspect_ratio:
            new_height = image.shape[1]
            new_width = int(new_height * target_aspect_ratio)
        else:
            new_width = image.shape[2]
            new_height = int(new_width / target_aspect_ratio)

        mask_w = x_max - x_min + 1
        mask_h = y_max - y_min + 1
        cannot_contain_mask = (mask_w > new_width) or (mask_h > new_height)

        # --- CAS 1 : le crop ne peut pas contenir le masque ---
        if cannot_contain_mask:
            print(f"[FRED_AutoCropImage] Fallback: le masque ({mask_w}x{mask_h}) "
                  f"est plus grand que la zone de crop ({new_width}x{new_height}). "
                  f"Utilisation d’un crop standard sans 'mask preserve'.")
            _, img_h, img_w, _ = image.shape
            if crop_from_center:
                x_start = max(0, (img_w  - new_width)  // 2)
                y_start = max(0, (img_h - new_height) // 2)
            else:
                x_start = int((crop_x_percent / 100) * (img_w  - new_width))
                y_start = int((crop_y_percent / 100) * (img_h - new_height))
                x_start = min(max(0, x_start), img_w  - new_width)
                y_start = min(max(0, y_start), img_h - new_height)

            new_crop_x_percent = int(100 * x_start / max(1, img_w  - new_width))
            new_crop_y_percent = int(100 * y_start / max(1, img_h - new_height))

            cropped_image, preview = self.crop_image_to_ratio(
                image, native_width, native_height,
                False, new_crop_x_percent, new_crop_y_percent,
                False, preview_intensity, preview_color
            )
            cropped_mask = self.crop_image_to_ratio(
                mask, native_width, native_height,
                False, new_crop_x_percent, new_crop_y_percent,
                True, preview_intensity, preview_color
            )
            return cropped_image, preview, cropped_mask

        # --- CAS 2 : crop “normal” avec snap au masque ---
        if crop_from_center:
            mask_cx = (x_min + x_max) // 2
            mask_cy = (y_min + y_max) // 2
            x_start = mask_cx - new_width // 2
            y_start = mask_cy - new_height // 2
        else:
            x_start = int((crop_x_percent / 100) * (image.shape[2] - new_width))
            y_start = int((crop_y_percent / 100) * (image.shape[1] - new_height))

        # Snap forcé si le masque touche les bords
        if y_min == 0 or y_max == image.shape[1] - 1 or \
           x_min == 0 or x_max == image.shape[2] - 1:
            print("[FRED_AutoCropImage] Ajustement: le masque touche un ou plusieurs bords, "
                  "le crop est déplacé pour rester dans l’image.")

        x_start = max(0, min(x_start, image.shape[2] - new_width))
        y_start = max(0, min(y_start, image.shape[1] - new_height))
        x_start = min(x_start, x_min)
        x_start = max(x_start, x_max + 1 - new_width)
        y_start = min(y_start, y_min)
        y_start = max(y_start, y_max + 1 - new_height)

        new_crop_x_percent = int(100 * x_start / max(1, image.shape[2] - new_width))
        new_crop_y_percent = int(100 * y_start / max(1, image.shape[1] - new_height))

        cropped_image, preview = self.crop_image_to_ratio(
            image, native_width, native_height,
            False, new_crop_x_percent, new_crop_y_percent,
            False, preview_intensity, preview_color
        )
        cropped_mask = self.crop_image_to_ratio(
            mask, native_width, native_height,
            False, new_crop_x_percent, new_crop_y_percent,
            True, preview_intensity, preview_color
        )
        return cropped_image, preview, cropped_mask

    # --- Utilities ---
    # def apply_mask_overlay(self, preview, mask, mask_color, alpha=0.6):
    def apply_mask_overlay(self, preview, mask, mask_color, alpha):
        rgb = torch.tensor(self.Hex_to_RGB(mask_color), dtype=preview.dtype, device=preview.device) / 255.0
        B, H, W, C = preview.shape
        if mask.ndim == 3:
            mask = mask.unsqueeze(-1)
        if mask.shape[1:3] != (H, W):
            mask = mask.permute(0, 3, 1, 2)
            mask = torch.nn.functional.interpolate(mask, size=(H, W), mode="nearest")
            mask = mask.permute(0, 2, 3, 1)
        mask = mask.clamp(0.0, 1.0)
        overlay = rgb.view(1, 1, 1, 3).expand(B, H, W, 3)
        blended = preview * (1.0 - mask * alpha) + overlay * (mask * alpha)
        return blended.clamp(0.0, 1.0)

    def find_mask_boundaries(self, mask):
        if mask is None: return None, None, None, None
        mask_np = mask.squeeze().cpu().numpy()
        rows = np.any(mask_np, axis=1)
        cols = np.any(mask_np, axis=0)
        if not np.any(rows) or not np.any(cols):
            return 0, 0, mask_np.shape[1] - 1, mask_np.shape[0] - 1
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        return x_min, y_min, x_max, y_max

    def resize_image(self, cropped_image, mode, width, height, crop_from_center):
        samples = cropped_image.movedim(-1, 1)
        resized_image = comfy.utils.common_upscale(samples, width, height, mode, crop_from_center)
        return resized_image.movedim(1, -1)

    def find_closest_sd_resolution(self, original_width, original_height):
        sd_ratios = [(a["name"], a["width"], a["height"]) for a in ASPECT_RATIOS]
        original_ratio = original_width / original_height
        for name, w, h in sd_ratios:
            if abs((w / h) - original_ratio) < 0.001:
                # return w, h, f"{name} - ({w}x{h})"
                return w, h, f"{name}"
        closest = min(sd_ratios, key=lambda ar: abs(original_ratio - (ar[1] / ar[2])))
        return closest[1], closest[2], f"{closest[0]}"

    def crop_image_to_ratio(self, image, native_width, native_height, crop_from_center,
                            crop_x_percent, crop_y_percent, is_mask,
                            preview_intensity, preview_color):
        _, original_height, original_width, *_ = image.shape
        target_ratio = native_width / native_height
        if (original_width / original_height) > target_ratio:
            new_height = original_height
            new_width = int(new_height * target_ratio)
        else:
            new_width = original_width
            new_height = int(new_width / target_ratio)

        if crop_from_center:
            x_start = max(0, (original_width - new_width) // 2)
            y_start = max(0, (original_height - new_height) // 2)
        else:
            x_start = int((crop_x_percent / 100) * (original_width - new_width))
            y_start = int((crop_y_percent / 100) * (original_height - new_height))
            x_start = min(max(0, x_start), original_width - new_width)
            y_start = min(max(0, y_start), original_height - new_height)

        if is_mask:
            return image[:, y_start:y_start+new_height, x_start:x_start+new_width]
        # else:
            # preview_color_tensor = torch.tensor(self.Hex_to_RGB(preview_color), dtype=torch.uint8, device=image.device)
            # preview = image.clone()
            # overlay_image = torch.full((1, original_height, original_width, 3), 255, dtype=torch.uint8, device=image.device)
            # if x_start > 0:
                # overlay_image[:, :, :x_start, :] = preview_color_tensor
            # if x_start + new_width < original_width:
                # overlay_image[:, :, x_start+new_width:, :] = preview_color_tensor
            # if y_start > 0:
                # overlay_image[:, :y_start, x_start:x_start+new_width, :] = preview_color_tensor
            # if y_start + new_height < original_height:
                # overlay_image[:, y_start+new_height:, x_start:x_start+new_width, :] = preview_color_tensor
            # overlay_float = overlay_image.float() / 255.0
            # preview_float = preview.float()
            # blend_preview = self.blend_images(preview_float, overlay_float, preview_intensity)
            # blend_np = (blend_preview[0].cpu().numpy() * 255).astype(np.uint8)
            # blend_np = np.ascontiguousarray(blend_np)
            # cv2.rectangle(blend_np, (x_start, y_start), (x_start+new_width, y_start+new_height),
                          # (int(preview_color_tensor[0]), int(preview_color_tensor[1]), int(preview_color_tensor[2])), 4)
            # blend_preview = torch.from_numpy(blend_np).unsqueeze(0).float() / 255.0
            # cropped_image = image[:, y_start:y_start+new_height, x_start:x_start+new_width, :]
            # return cropped_image, blend_preview
        else:
            preview = image.clone()

            mask_overlay = torch.ones((1, original_height, original_width), dtype=torch.float32, device=image.device)
            mask_overlay[:, y_start:y_start+new_height, x_start:x_start+new_width] = 0.0

            # Passer la couleur Jaune sous forme de string hexadécimale ici, comme preview_color
            preview = self.apply_mask_overlay(preview, mask_overlay, "#DCDC32", preview_intensity)

            # Conversion en numpy pour le dessin du rectangle contour
            blend_np = (preview[0].cpu().numpy() * 255).astype(np.uint8)
            blend_np = np.ascontiguousarray(blend_np)

            # Dessin du rectangle contour avec OpenCV (épaisseur 4)
            preview_color_tensor = torch.tensor(self.Hex_to_RGB("#DCDC32"), dtype=torch.uint8, device=image.device)
            cv2.rectangle(
                blend_np,
                (x_start, y_start),
                (x_start + new_width, y_start + new_height),
                (int(preview_color_tensor[0]), int(preview_color_tensor[1]), int(preview_color_tensor[2])),
                thickness=4
            )

            # Conversion inverse en tensor float normalisé
            blend_preview = torch.from_numpy(blend_np).unsqueeze(0).float() / 255.0

            # Crop de l'image originale à retourner
            cropped_image = image[:, y_start:y_start + new_height, x_start:x_start + new_width, :]

            return cropped_image, blend_preview

    def blend_images(self, image1, image2, blend_factor):
        if image1.shape != image2.shape:
            image2 = self.crop_and_resize(image2, image1.shape)
        blended = image1 * (1 - blend_factor) + (image1 * image2) * blend_factor
        return torch.clamp(blended, 0, 1)

    def crop_and_resize(self, img, target_shape):
        _, img_h, img_w, _ = img.shape
        _, target_h, target_w, _ = target_shape
        img_aspect = img_w / img_h
        target_aspect = target_w / target_h
        if img_aspect > target_aspect:
            new_width = int(img_h * target_aspect)
            left = (img_w - new_width) // 2
            img = img[:, :, left:left+new_width, :]
        else:
            new_height = int(img_w / target_aspect)
            top = (img_h - new_height) // 2
            img = img[:, top:top+new_height, :, :]
        img = img.permute(0, 3, 1, 2)
        img = F.interpolate(img, size=(target_h, target_w), mode='bilinear', align_corners=False)
        return img.permute(0, 2, 3, 1)

    def Hex_to_RGB(self, inhex: str) -> tuple:
        if not inhex.startswith('#'):
            raise ValueError(f'Invalid Hex Code in {inhex}')
        return (int(inhex[1:3], 16), int(inhex[3:5], 16), int(inhex[5:], 16))

    @staticmethod
    def IS_CHANGED(*args, **kwargs):
        return False

NODE_CLASS_MAPPINGS = {
    "FRED_AutoCropImage_Native_Ratio": FRED_AutoCropImage_Native_Ratio
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FRED_AutoCropImage_Native_Ratio": "👑 FRED AutoCropImage Native Ratio"
}
