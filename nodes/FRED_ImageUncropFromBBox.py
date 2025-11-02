import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from ..utils import tensor2pil, pil2tensor
from PIL import ImageOps

HELP_MESSAGE = """
👑️ FRED_ImageUncropFromBBox

🔹 PURPOSE:
Allows placing a cropped image (with or without resizing) back into its original context using a bounding box (bbox),
with smooth progressive blending using a mask and border blending factor.

📥 INPUTS:
- original_image • Full original image (IMAGE)
- cropped_image • Cropped and possibly modified image (IMAGE)
- bbox • Bounding box (x0, y0, x1, y1) in the original image where the crop should be placed (BBOX)
- border_blending • Strength of blend/fade on edges (FLOAT, 0=no blend, 1=strong)
- use_square_mask • Mask shape choice: square (True) or ellipse (False) (BOOLEAN)
- resize_mode • Resizing method to apply for the optional mask (STRING among bilinear, bicubic, nearest, area)
- optional_mask • Optional mask to blend the cropped image (MASK)

⚙️ HOW IT WORKS:
- The cropped image is automatically resized to match the bbox before blending.
- If a mask is given, its size is detected versus original image, cropped image, or bbox.
- The mask is then resized using the chosen resize_mode to fit the bbox.
- The mask controls the blending area between the original image and the crop.
- Makes it easy to reintegrate modified cropped images into their original context.

📤 OUTPUT:
- image • Resulting image after blending the crop into the original according to bbox and mask.
"""

class FRED_ImageUncropFromBBox:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "cropped_image": ("IMAGE",),
                "bbox": ("BBOX",),
                "border_blending": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "use_square_mask": ("BOOLEAN", {"default": True}),
                "resize_mode": (["bilinear", "bicubic", "nearest", "area"], {"default": "bicubic"}),
            },
            "optional": {
                "optional_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "help")
    FUNCTION = "uncrop"
    CATEGORY = "👑FRED/image/postprocessing"

    def bbox_to_xywh(self, bbox):
        if len(bbox) != 4:
            raise ValueError(f"Invalid bbox length: {len(bbox)}")
        x0, y0, x1, y1 = bbox
        if x1 < x0 or y1 < y0:
            # Probably already xywh format
            return (x0, y0, x1, y1)
        return (x0, y0, x1 - x0, y1 - y0)

    def uncrop(self, original_image, cropped_image, bbox, border_blending, use_square_mask, resize_mode="bicubic", optional_mask=None):
        def inset_border(image, border_width=20, border_color=(0)):
            width, height = image.size
            bordered_image = Image.new(image.mode, (width, height), border_color)
            bordered_image.paste(image, (0, 0))
            draw = ImageDraw.Draw(bordered_image)
            draw.rectangle((0, 0, width - 1, height - 1), outline=border_color, width=border_width)
            return bordered_image

        if isinstance(bbox[0], (list, tuple)):
            bboxes = bbox
        else:
            bboxes = [bbox]

        if len(cropped_image) != len(bboxes):
            raise ValueError(f"Count mismatch: {len(cropped_image)} cropped images vs {len(bboxes)} bboxes")
        if optional_mask is not None:
            if (torch.is_tensor(optional_mask) and len(optional_mask) != len(cropped_image)) \
                    or (isinstance(optional_mask, list) and len(optional_mask) != len(cropped_image)):
                raise ValueError(f"Count mismatch: {len(cropped_image)} cropped images vs {len(optional_mask)} optional masks")

        img = tensor2pil(original_image[0]).convert("RGBA")
        blend_img = img.copy()
        
        border_blending = border_blending * 0.2

        for i in range(len(cropped_image)):
            crop = tensor2pil(cropped_image[i])
            _bbox = bboxes[i]

            x, y, w, h = self.bbox_to_xywh(_bbox)
            paste_region = (x, y)

            bbox_size = (round(w), round(h))
            # crop = crop.resize(bbox_size).convert("RGBA")
            # taille cible (bbox)
            target_w, target_h = bbox_size

            # resize crop en conservant le ratio, avec remplissage transparent
            crop = crop.convert("RGBA")
            crop = ImageOps.contain(crop, (target_w, target_h))  # preserve ratio
            padded = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))  # transparent

            # centrer le crop dans le bbox
            offset = ((target_w - crop.width) // 2, (target_h - crop.height) // 2)
            padded.paste(crop, offset)
            crop = padded

            blend_ratio = (max(bbox_size) / 2) * float(border_blending)
            

            mask_bbox = Image.new("L", bbox_size, 0)

            if use_square_mask:
                mask_block = Image.new("L", bbox_size, 255)
                mask_block = inset_border(mask_block, round((max(bbox_size) / 2) * border_blending / 2), 0)
                mask_bbox.paste(mask_block, (0, 0))

                # Flou appliqué ici après collage direct
                if border_blending > 0:
                    # blur_radius = int(min(mask_bbox.size) * border_blending)
                    blur_radius = int(blend_ratio / 4)
                    if blur_radius > 0:
                        mask_bbox = mask_bbox.filter(ImageFilter.BoxBlur(blur_radius))
                        mask_bbox = mask_bbox.filter(ImageFilter.GaussianBlur(blur_radius))

            else:
                if optional_mask is None:
                    raise ValueError("optional_mask is required when use_square_mask is False")

                if isinstance(optional_mask, list):
                    mask_tensor = optional_mask[i]
                elif torch.is_tensor(optional_mask):
                    mask_tensor = optional_mask[i]
                else:
                    mask_tensor = optional_mask

                if torch.is_tensor(mask_tensor):
                    if mask_tensor.dim() == 3 and mask_tensor.shape[0] == 1:
                        mask_tensor = mask_tensor.squeeze(0)
                    elif mask_tensor.dim() == 3 and mask_tensor.shape[0] in [3, 4]:
                        mask_tensor = mask_tensor.permute(1, 2, 0)
                    mask_pil = tensor2pil(mask_tensor)
                elif isinstance(mask_tensor, Image.Image):
                    mask_pil = mask_tensor
                else:
                    raise ValueError(f"Unsupported mask_tensor type: {type(mask_tensor)}")

                mask_width, mask_height = mask_pil.size
                orig_crop = tensor2pil(cropped_image[i])
                orig_crop_w, orig_crop_h = orig_crop.size

                if (mask_width, mask_height) == bbox_size:
                    resized_mask = mask_pil
                elif (mask_width, mask_height) == (orig_crop_w, orig_crop_h):
                    resized_mask = mask_pil.resize(bbox_size, resample=Image.Resampling[resize_mode.upper()])
                elif (mask_width, mask_height) == img.size:
                    resized_mask = mask_pil.crop((x, y, x + w, y + h))
                else:
                    resized_mask = mask_pil.resize(bbox_size, resample=Image.Resampling[resize_mode.upper()])

                mask_bbox.paste(resized_mask, (0, 0))

                # Flou appliqué ici après collage du masque optionnel
                if border_blending > 0:
                    blur_radius = int(min(mask_bbox.size) * border_blending)
                    if blur_radius > 0:
                        mask_bbox = mask_bbox.filter(ImageFilter.BoxBlur(blur_radius))
                        mask_bbox = mask_bbox.filter(ImageFilter.GaussianBlur(blur_radius))

            blend_img.paste(crop, paste_region, mask_bbox)

        result = blend_img.convert("RGB")
        output_images = torch.cat([pil2tensor(result)], dim=0)

        return output_images, HELP_MESSAGE

    # def uncrop(self, original_image, cropped_image, bbox, border_blending, use_square_mask, resize_mode="bicubic", optional_mask=None):
        # def inset_border(image, border_width=20, border_color=(0)):
            # width, height = image.size
            # bordered_image = Image.new(image.mode, (width, height), border_color)
            # bordered_image.paste(image, (0, 0))
            # draw = ImageDraw.Draw(bordered_image)
            # draw.rectangle((0, 0, width - 1, height - 1), outline=border_color, width=border_width)
            # return bordered_image

        # if isinstance(bbox[0], (list, tuple)):
            # bboxes = bbox
        # else:
            # bboxes = [bbox]

        # if len(cropped_image) != len(bboxes):
            # raise ValueError(f"Count mismatch: {len(cropped_image)} cropped images vs {len(bboxes)} bboxes")
        # if optional_mask is not None:
            # if optional_mask.dim() == 3 and len(optional_mask) != len(cropped_image):
                # raise ValueError(f"Count mismatch: {len(cropped_image)} cropped images vs {len(optional_mask)} optional masks")
        # # if len(original_image) != len(cropped_image):
            # # raise ValueError(f"Image count mismatch: {len(original_image)} vs {len(cropped_image)}")
        # # if len(bboxes) < len(original_image):
            # # raise ValueError(f"Not enough bboxes: {len(bboxes)} for {len(original_image)} images")
        # # if len(bboxes) > len(original_image):
            # # bboxes = bboxes[:len(original_image)]
            # # print(f"Warning: Dropping excess bboxes. Keeping {len(bboxes)}")

        # out_images = []
        # img = tensor2pil(original_image[0])

        # # for i in range(len(original_image)):
        # for i in range(len(cropped_image)):
            # # img = tensor2pil(original_image[i])
            # crop = tensor2pil(cropped_image[i])
            # _bbox = bboxes[i]

            # x, y, w, h = self.bbox_to_xywh(_bbox)
            # paste_region = (x, y, x + w, y + h)

            # crop = crop.resize((round(w), round(h)))
            # crop_img = crop.convert("RGB")

            # blend_ratio = (max(crop_img.size) / 2) * float(border_blending)
            # blend = img.convert("RGBA")
            # mask = Image.new("L", img.size, 0)

            # # crée un masque local uniquement à la taille du bbox (w,h)
            # mask_bbox = Image.new("L", (w, h), 0)

            # if use_square_mask:
                # mask_block = Image.new("L", (round(w), round(h)), 255)
                # mask_block = inset_border(mask_block, round(blend_ratio / 2), (0))
                # mask.paste(mask_block, paste_region)
            # else:
                # if optional_mask is None:
                    # raise ValueError("optional_mask is required when use_square_mask is False")

                # # Gestion flexible du type d'optional_mask
                # if isinstance(optional_mask, list):
                    # # Liste Python de masques PIL ou tensors
                    # mask_tensor = optional_mask[i]
                # elif torch.is_tensor(optional_mask):
                    # # Tensor batch: indexation directe
                    # mask_tensor = optional_mask[i]
                # else:
                    # # Single mask (PIL image ou tensor) à réutiliser pour tous
                    # mask_tensor = optional_mask

                # # Conversion tensor vers PIL Image adaptée
                # if torch.is_tensor(mask_tensor):
                    # if mask_tensor.dim() == 3 and mask_tensor.shape[0] == 1:
                        # mask_tensor = mask_tensor.squeeze(0)
                    # elif mask_tensor.dim() == 3 and mask_tensor.shape[0] in [3, 4]:
                        # mask_tensor = mask_tensor.permute(1, 2, 0)
                    # mask_pil = tensor2pil(mask_tensor)
                # elif isinstance(mask_tensor, Image.Image):
                    # mask_pil = mask_tensor
                # else:
                    # raise ValueError(f"Unsupported mask_tensor type: {type(mask_tensor)}")

                # # mask_tensor = optional_mask[i] if optional_mask.dim() == 3 else optional_mask

                # # img_width, img_height = img.size
                # mask_width, mask_height = mask_tensor.shape[-1], mask_tensor.shape[-2]
                # img_width, img_height = img.size[0], img.size[1]

                # # Get original crop size BEFORE resize
                # orig_crop = tensor2pil(cropped_image[i])
                # orig_crop_w, orig_crop_h = orig_crop.size

                # crop_width, crop_height = crop_img.size
                # bbox_size = (round(w), round(h))

                # # Determine mask resize and paste strategy
                # if (mask_width, mask_height) == (img_width, img_height):
                    # print(f"Mask detected as original_image size: {mask_width}x{mask_height}")
                    # # Mask is same size as original image
                    # mask_pil = tensor2pil(mask_tensor)
                    # # Paste mask on entire image (0,0), no resize
                    # mask.paste(mask_pil, (0, 0))
                # elif (mask_width, mask_height) == (orig_crop_w, orig_crop_h):
                    # print(f"Mask detected as cropped_image size (before resize): {mask_width}x{mask_height}")
                    # # Mask corresponds to cropped_image size (before resize) — resize to bbox
                    # mask_pil = tensor2pil(mask_tensor)
                    # mask_pil = mask_pil.resize(bbox_size, resample=Image.Resampling[resize_mode.upper()])
                    # mask.paste(mask_pil, paste_region)
                # elif (mask_width, mask_height) == bbox_size:
                    # print(f"Mask detected as BBOX size: {mask_width}x{mask_height}")
                    # # Mask corresponds exactly to bbox size — paste directly
                    # mask_pil = tensor2pil(mask_tensor)
                    # mask.paste(mask_pil, paste_region)
                # elif True:
                    # # Try resizing mask to original image size
                    # mask_pil_test = tensor2pil(mask_tensor).resize((img_width, img_height), resample=Image.Resampling[resize_mode.upper()])
                    # if mask_pil_test.size == (img_width, img_height):
                        # print("Mask detected as original_image when resized")
                        # mask_pil = mask_pil_test
                        # mask.paste(mask_pil, (0, 0))
                    # else:
                        # # Try resizing mask to cropped image size (before resize)
                        # mask_pil_test = tensor2pil(mask_tensor).resize((orig_crop_w, orig_crop_h), resample=Image.Resampling[resize_mode.upper()])
                        # if mask_pil_test.size == (orig_crop_w, orig_crop_h):
                            # print("Mask detected as cropped_image when resized")
                            # mask_pil = mask_pil_test.resize(bbox_size, resample=Image.Resampling[resize_mode.upper()])
                            # mask.paste(mask_pil, paste_region)
                        # else:
                            # # Try resizing mask to bbox size
                            # mask_pil_test = tensor2pil(mask_tensor).resize(bbox_size, resample=Image.Resampling[resize_mode.upper()])
                            # if mask_pil_test.size == bbox_size:
                                # print("Mask detected as BBOX when resized")
                                # mask_pil = mask_pil_test
                                # mask.paste(mask_pil, paste_region)
                            # else:
                                # print(f"Mask size unknown ({mask_width}x{mask_height}), resizing to bbox {bbox_size}")
                                # # Fallback resize to bbox
                                # mask_pil = tensor2pil(mask_tensor)
                                # mask_pil = mask_pil.resize(bbox_size, resample=Image.Resampling[resize_mode.upper()])
                                # mask.paste(mask_pil, paste_region)

            # mask = mask.filter(ImageFilter.BoxBlur(radius=blend_ratio / 4))
            # mask = mask.filter(ImageFilter.GaussianBlur(radius=blend_ratio / 4))

            # blend.paste(crop_img, paste_region)
            # blend.putalpha(mask)

            # result = Image.alpha_composite(img.convert("RGBA"), blend)
            # out_images.append(result.convert("RGB"))

        # output_images = torch.cat([pil2tensor(img) for img in out_images], dim=0)

        # return output_images, HELP_MESSAGE

    @staticmethod
    def IS_CHANGED(*args, **kwargs):
        return False


NODE_CLASS_MAPPINGS = {
    "FRED_ImageUncropFromBBox": FRED_ImageUncropFromBBox
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FRED_ImageUncropFromBBox": "👑 FRED_ImageUncropFromBBox"
}