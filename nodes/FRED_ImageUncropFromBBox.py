import torch
import numpy as np
import logging
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from ..utils import tensor2pil, pil2tensor

# Import for gaussian blur
try:
    import torchvision.transforms.functional as TF
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
    print("[FRED_ImageUncropFromBBox] Warning: torchvision not available, using PIL for blur")

# Try to import scipy for erosion, fallback to simple method if not available
try:
    import scipy.ndimage
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[FRED_ImageUncropFromBBox] Warning: scipy not available, using fallback erosion method")

HELP_MESSAGE = """
👑️ FRED_ImageUncropFromBBox

🔹 PURPOSE:
Allows placing a cropped image (with or without resizing) back into its original context using a bounding box (bbox),
with smooth progressive blending using border blending and optional mask support.

📥 INPUTS:
- original_image • Full original image (IMAGE)
- cropped_image • Cropped and possibly modified image (IMAGE)
- bbox • Bounding box coordinates in the original image where the crop should be placed (BBOX)
- border_blending • Strength of blend/fade on edges (FLOAT, 0=no blur, 1=maximum blur)
- erode_size • Pixels to erode mask inward before blur (INT, 0-100, creates inward gradient)
- use_mask • Enable mask-based blending: False=rectangular blend, True=use optional_mask shape (BOOLEAN)
- resize_mode • Resizing method for images and masks (STRING: bilinear, bicubic, nearest, area)
- bbox_mode • Bounding box format: x0y0x1y1 (default) or xywh (STRING)
- optional_mask • Optional mask to control blending shape and intensity (MASK)

⚙️ HOW IT WORKS:
- Cropped image is automatically resized to match the bbox dimensions while preserving aspect ratio
- When use_mask=False: rectangular blending with gradient borders controlled by border_blending
- When use_mask=True: 
  1. Adds black padding to mask edges that touch bbox boundaries (automatic, based on erode_size)
  2. Erodes the mask inward to shrink white area
  3. Blurs the eroded mask to create smooth gradient
  4. Crops back to bbox size, keeping the inward gradient
- This technique ensures gradient only goes toward mask interior, not exterior
- Handles multiple images, bboxes, and masks automatically (duplicates single values to match batch size)
- Robust dimension detection for masks (original image size, cropped image size, or bbox size)

📤 OUTPUT:
- image • Resulting image after blending the crop into the original according to bbox and mask
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
                "erode_size": ("INT", {"default": 20, "min": 0, "max": 100, "step": 1}),
                "use_mask": ("BOOLEAN", {"default": False}),
                "resize_mode": (["bilinear", "bicubic", "nearest", "area"], {"default": "bicubic"}),
                "bbox_mode": (["x0y0x1y1", "xywh"], {"default": "x0y0x1y1"}),
            },
            "optional": {
                "optional_mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "help")
    FUNCTION = "uncrop"
    CATEGORY = "👑FRED/image/postprocessing"

    def uncrop(
        self,
        original_image,
        cropped_image,
        bbox,
        border_blending,
        erode_size,
        use_mask,
        resize_mode="bicubic",
        bbox_mode="x0y0x1y1",
        optional_mask=None
    ):
        """
        Uncrop images using bounding boxes and optional masks.
        Handles batch processing, multiple bboxes, and robust mask dimension matching.
        """

        # Ensure tensors are in proper format: (B, H, W, C) for images
        if original_image.dim() == 3:
            original_image = original_image.unsqueeze(0)

        B, H_orig, W_orig, C = original_image.shape

        # Handle bbox format
        if not isinstance(bbox, list):
            bboxes = [bbox]
        elif len(bbox) > 0 and not isinstance(bbox[0], (list, tuple)):
            bboxes = [bbox]
        else:
            bboxes = bbox

        # Determine batch size
        num_images = len(cropped_image)
        num_bboxes = len(bboxes)
        num_masks = len(optional_mask) if optional_mask is not None else 0

        # Find maximum batch size
        max_batch = max(num_images, num_bboxes, num_masks if optional_mask is not None else 0)

        # Duplicate single items to match batch
        if num_images == 1 and max_batch > 1:
            cropped_image = cropped_image.repeat(max_batch, 1, 1, 1)
            num_images = max_batch

        if num_bboxes == 1 and max_batch > 1:
            bboxes = bboxes * max_batch
            num_bboxes = max_batch

        if optional_mask is not None and num_masks == 1 and max_batch > 1:
            optional_mask = optional_mask.repeat(max_batch, 1, 1)
            num_masks = max_batch

        # Validate counts match
        if not (num_images == num_bboxes == (num_masks if optional_mask is not None else num_bboxes)):
            raise ValueError(
                f"Batch size mismatch: {num_images} images, {num_bboxes} bboxes, "
                f"{num_masks if optional_mask is not None else 'no'} masks"
            )

        # Convert PIL resampling mode
        resample_mode_map = {
            "bilinear": Image.Resampling.BILINEAR,
            "bicubic": Image.Resampling.BICUBIC,
            "nearest": Image.Resampling.NEAREST,
            "area": Image.Resampling.BOX,
        }
        pil_resample = resample_mode_map.get(resize_mode, Image.Resampling.BICUBIC)

        # Process each image
        result_images = []

        for i in range(num_images):
            # Get current items
            orig_tensor = original_image[0] if len(original_image) == 1 else original_image[i % len(original_image)]
            crop_tensor = cropped_image[i]
            current_bbox = bboxes[i]

            # Parse bbox
            x0, y0, x1, y1 = self.parse_bbox(current_bbox, bbox_mode)
            bbox_w = x1 - x0
            bbox_h = y1 - y0

            # Convert tensors to PIL
            orig_pil = tensor2pil(orig_tensor).convert("RGB")
            crop_pil = tensor2pil(crop_tensor).convert("RGB")

            # Resize cropped image to bbox size while preserving aspect ratio
            crop_resized = ImageOps.fit(crop_pil, (bbox_w, bbox_h), method=pil_resample)

            # Create or process mask
            if not use_mask:
                # Rectangular mask with border blending
                mask_pil = self.create_rectangular_mask(
                    bbox_w, bbox_h, border_blending
                )
            else:
                # Use optional mask
                if optional_mask is None:
                    raise ValueError("optional_mask is required when use_mask=True")

                mask_tensor = optional_mask[i]

                # Check if bbox touches image borders
                bbox_at_edges = {
                    'left': x0 == 0,
                    'right': x1 >= W_orig,
                    'top': y0 == 0,
                    'bottom': y1 >= H_orig
                }

                mask_pil = self.process_mask(
                    mask_tensor, 
                    orig_pil.size, 
                    crop_pil.size,
                    (bbox_w, bbox_h),
                    (x0, y0, x1, y1),
                    pil_resample,
                    border_blending,
                    erode_size,
                    bbox_at_edges
                )

            # Blend images
            result_pil = self.blend_with_mask(
                orig_pil, crop_resized, mask_pil, (x0, y0)
            )

            result_images.append(pil2tensor(result_pil))

        # Stack results
        output_tensor = torch.cat(result_images, dim=0)

        return (output_tensor, HELP_MESSAGE)

    def parse_bbox(self, bbox, mode):
        """Parse bbox based on format mode"""
        if mode == "xywh":
            x, y, w, h = [int(v) for v in bbox]
            return x, y, x + w, y + h
        else:  # x0y0x1y1
            x0, y0, x1, y1 = [int(v) for v in bbox]
            return x0, y0, x1, y1

    def create_rectangular_mask(self, width, height, border_blending):
        """Create rectangular mask with gradient borders"""
        mask = Image.new("L", (width, height), 255)

        if border_blending > 0:
            # Calculate border width based on border_blending percentage
            max_border = min(width, height) // 4
            border_width = int(max_border * border_blending)

            if border_width > 0:
                # Create gradient mask
                mask_array = np.ones((height, width), dtype=np.float32)

                # Apply gradient on all sides
                for i in range(border_width):
                    alpha = i / border_width
                    # Top
                    if i < height:
                        mask_array[i, :] *= alpha
                    # Bottom
                    if height - 1 - i >= 0:
                        mask_array[height - 1 - i, :] *= alpha
                    # Left
                    if i < width:
                        mask_array[:, i] *= alpha
                    # Right
                    if width - 1 - i >= 0:
                        mask_array[:, width - 1 - i] *= alpha

                mask = Image.fromarray((mask_array * 255).astype(np.uint8), mode="L")

                # Apply gaussian blur for smoother transition
                blur_radius = max(1, border_width // 4)
                mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        return mask

    def process_mask(self, mask_tensor, orig_size, crop_size, bbox_size, bbox_coords, resample, border_blending, erode_size, bbox_at_edges):
        """
        Process mask tensor with Padding + Erosion + Blur technique.
        This creates an inward-only gradient even when mask touches bbox edges.
        """
        # Normalize mask tensor to (H, W) format
        if mask_tensor.dim() == 4:
            if mask_tensor.shape[1] == 1:
                mask_tensor = mask_tensor.squeeze(0).squeeze(0)
            elif mask_tensor.shape[-1] == 1:
                mask_tensor = mask_tensor.squeeze(0).squeeze(-1)
            else:
                mask_tensor = mask_tensor[0, :, :, 0] if mask_tensor.shape[-1] < mask_tensor.shape[1] else mask_tensor[0, 0, :, :]
        elif mask_tensor.dim() == 3:
            if mask_tensor.shape[0] == 1:
                mask_tensor = mask_tensor.squeeze(0)
            elif mask_tensor.shape[-1] == 1:
                mask_tensor = mask_tensor.squeeze(-1)
            elif mask_tensor.shape[0] in [3, 4]:
                mask_tensor = mask_tensor[0, :, :]

        # Get dimensions
        mask_h, mask_w = mask_tensor.shape[-2:]
        orig_w, orig_h = orig_size
        crop_w, crop_h = crop_size
        bbox_w, bbox_h = bbox_size
        x0, y0, x1, y1 = bbox_coords

        # Convert to numpy
        mask_np = mask_tensor.cpu().numpy()
        if mask_np.max() <= 1.0:
            mask_np = (mask_np * 255.0)
        mask_np = np.clip(mask_np, 0, 255).astype(np.uint8)
        mask_pil = Image.fromarray(mask_np, mode="L")

        # Detect mask context and resize
        if (mask_w, mask_h) == (bbox_w, bbox_h):
            print(f"[FRED_ImageUncropFromBBox] Mask matches bbox size: {mask_w}x{mask_h}")
            resized_mask = mask_pil
        elif (mask_w, mask_h) == (crop_w, crop_h):
            print(f"[FRED_ImageUncropFromBBox] Mask matches crop size: {mask_w}x{mask_h}, resizing to bbox")
            resized_mask = mask_pil.resize((bbox_w, bbox_h), resample=resample)
        elif (mask_w, mask_h) == (orig_w, orig_h):
            print(f"[FRED_ImageUncropFromBBox] Mask matches original image size: {mask_w}x{mask_h}, cropping to bbox")
            resized_mask = mask_pil.crop((x0, y0, x1, y1))
        else:
            print(f"[FRED_ImageUncropFromBBox] Mask size {mask_w}x{mask_h} unknown, resizing to bbox {bbox_w}x{bbox_h}")
            resized_mask = mask_pil.resize((bbox_w, bbox_h), resample=resample)

        # Apply Padding + Erosion + Blur technique
        if border_blending > 0 or erode_size > 0:
            MAX_BLUR_RADIUS = 150
            blur_radius = int(MAX_BLUR_RADIUS * border_blending)
            blur_radius = max(1, blur_radius)

            # Calculate padding: needs to be at least erode_size for erosion to work
            # Add extra blur_radius for smooth transition
            pad_pixels = erode_size + blur_radius

            # Determine which sides need padding (only if NOT at image edge)
            pad_left = 0 if bbox_at_edges['left'] else pad_pixels
            pad_right = 0 if bbox_at_edges['right'] else pad_pixels
            pad_top = 0 if bbox_at_edges['top'] else pad_pixels
            pad_bottom = 0 if bbox_at_edges['bottom'] else pad_pixels

            # Convert to numpy array
            mask_array = np.array(resized_mask).astype(np.float32) / 255.0

            # Step 1: Add black padding if needed
            if any([pad_left, pad_right, pad_top, pad_bottom]):
                mask_array = np.pad(
                    mask_array,
                    ((pad_top, pad_bottom), (pad_left, pad_right)),
                    mode='constant',
                    constant_values=0
                )
                print(f"[FRED_ImageUncropFromBBox] Added padding: {pad_pixels}px (L:{pad_left}, R:{pad_right}, T:{pad_top}, B:{pad_bottom})")

            # Convert to torch for processing
            mask_torch = torch.from_numpy(mask_array)

            # Step 2: Erode the mask inward
            if erode_size > 0:
                if HAS_SCIPY:
                    # Use scipy for better erosion
                    mask_array = scipy.ndimage.grey_erosion(mask_array, size=(erode_size, erode_size))
                    mask_torch = torch.from_numpy(mask_array)
                else:
                    # Fallback: simple min pooling erosion
                    kernel_size = erode_size * 2 + 1
                    if kernel_size > 1:
                        mask_torch = mask_torch.unsqueeze(0).unsqueeze(0)
                        mask_torch = -F.max_pool2d(-mask_torch, kernel_size, stride=1, padding=erode_size)
                        mask_torch = mask_torch.squeeze(0).squeeze(0)

                print(f"[FRED_ImageUncropFromBBox] Applied erosion: {erode_size}px")

            # Step 3: Blur the eroded mask
            if blur_radius > 0:
                # Ensure blur kernel size is odd
                blur_kernel = blur_radius * 2 + 1

                # Apply Gaussian blur
                if HAS_TORCHVISION:
                    mask_torch = mask_torch.unsqueeze(0).unsqueeze(0)
                    mask_torch = TF.gaussian_blur(mask_torch, blur_kernel)
                    mask_torch = mask_torch.squeeze(0).squeeze(0)
                else:
                    # Fallback to PIL
                    mask_array = mask_torch.cpu().numpy()
                    mask_pil_temp = Image.fromarray((mask_array * 255).astype(np.uint8), mode="L")
                    mask_pil_temp = mask_pil_temp.filter(ImageFilter.GaussianBlur(radius=blur_radius))
                    mask_torch = torch.from_numpy(np.array(mask_pil_temp).astype(np.float32) / 255.0)

                print(f"[FRED_ImageUncropFromBBox] Applied blur: radius {blur_radius}px")

            # Step 4: Crop back to bbox size (removing padding, keeping gradient)
            if any([pad_left, pad_right, pad_top, pad_bottom]):
                mask_array = mask_torch.cpu().numpy()
                # Crop to remove padding
                mask_array = mask_array[pad_top:pad_top+bbox_h, pad_left:pad_left+bbox_w]
                mask_torch = torch.from_numpy(mask_array)
                print(f"[FRED_ImageUncropFromBBox] Cropped back to bbox size: {bbox_w}x{bbox_h}")

            # Convert back to PIL
            mask_array = mask_torch.cpu().numpy()
            mask_array = np.clip(mask_array * 255, 0, 255).astype(np.uint8)
            resized_mask = Image.fromarray(mask_array, mode="L")

        return resized_mask

    def blend_with_mask(self, orig_pil, crop_pil, mask_pil, paste_coords):
        """
        Blend cropped image into original using mask at specified coordinates.
        """
        x0, y0 = paste_coords

        # Ensure all images are RGB
        orig_pil = orig_pil.convert("RGB")
        crop_pil = crop_pil.convert("RGB")

        # Get dimensions (PIL uses W, H)
        crop_w, crop_h = crop_pil.size
        orig_w, orig_h = orig_pil.size
        mask_w, mask_h = mask_pil.size

        # Verify mask matches crop size
        if (mask_w, mask_h) != (crop_w, crop_h):
            print(f"[FRED_ImageUncropFromBBox] WARNING: Mask size {mask_w}x{mask_h} != crop size {crop_w}x{crop_h}, resizing mask")
            mask_pil = mask_pil.resize((crop_w, crop_h), Image.Resampling.BICUBIC)
            mask_w, mask_h = crop_w, crop_h

        # Convert to numpy for blending (H, W, C format)
        orig_array = np.array(orig_pil).astype(np.float32)
        crop_array = np.array(crop_pil).astype(np.float32)
        mask_array = np.array(mask_pil).astype(np.float32) / 255.0

        # Verify dimensions
        assert crop_array.shape[0] == crop_h and crop_array.shape[1] == crop_w,             f"Crop array shape mismatch: expected ({crop_h}, {crop_w}, 3), got {crop_array.shape}"
        assert mask_array.shape[0] == mask_h and mask_array.shape[1] == mask_w,             f"Mask array shape mismatch: expected ({mask_h}, {mask_w}), got {mask_array.shape}"

        # Expand mask to 3 channels
        if mask_array.ndim == 2:
            mask_array = np.stack([mask_array] * 3, axis=-1)

        # Calculate paste region with bounds checking
        x1 = min(x0 + crop_w, orig_w)
        y1 = min(y0 + crop_h, orig_h)

        actual_w = x1 - x0
        actual_h = y1 - y0

        if actual_w != crop_w or actual_h != crop_h:
            print(f"[FRED_ImageUncropFromBBox] WARNING: Crop extends beyond bounds, clipping to {actual_w}x{actual_h}")

        # Extract regions and blend
        orig_region = orig_array[y0:y1, x0:x1]
        crop_region = crop_array[:actual_h, :actual_w]
        mask_region = mask_array[:actual_h, :actual_w]

        blended_region = orig_region * (1.0 - mask_region) + crop_region * mask_region

        # Copy result back
        result_array = orig_array.copy()
        result_array[y0:y1, x0:x1] = blended_region

        # Convert to PIL
        result_pil = Image.fromarray(np.clip(result_array, 0, 255).astype(np.uint8), mode="RGB")

        return result_pil

    @staticmethod
    def IS_CHANGED(*args, **kwargs):
        return False


NODE_CLASS_MAPPINGS = {
    "FRED_ImageUncropFromBBox": FRED_ImageUncropFromBBox
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FRED_ImageUncropFromBBox": "👑 FRED_ImageUncropFromBBox"
}

# import torch
# import numpy as np
# import logging
# import torch.nn.functional as F
# from PIL import Image, ImageDraw, ImageFilter, ImageOps
# from ..utils import tensor2pil, pil2tensor

# HELP_MESSAGE = """
# 👑️ FRED_ImageUncropFromBBox

# 🔹 PURPOSE:
# Allows placing a cropped image (with or without resizing) back into its original context using a bounding box (bbox),
# with smooth progressive blending using border blending and optional mask support.

# 📥 INPUTS:
# - original_image • Full original image (IMAGE)
# - cropped_image • Cropped and possibly modified image (IMAGE)
# - bbox • Bounding box coordinates in the original image where the crop should be placed (BBOX)
# - border_blending • Strength of blend/fade on edges (FLOAT, 0=no blend, 1=strong fade)
# - use_mask • Enable mask-based blending: False=rectangular blend, True=use optional_mask shape (BOOLEAN)
# - resize_mode • Resizing method for images and masks (STRING: bilinear, bicubic, nearest, area)
# - bbox_mode • Bounding box format: x0y0x1y1 (default) or xywh (STRING)
# - optional_mask • Optional mask to control blending shape and intensity (MASK)

# ⚙️ HOW IT WORKS:
# - Cropped image is automatically resized to match the bbox dimensions while preserving aspect ratio
# - When use_mask=False: rectangular blending with gradient borders controlled by border_blending
# - When use_mask=True: blending follows the mask shape with gradient around mask edges
# - Handles multiple images, bboxes, and masks automatically (duplicates single values to match batch size)
# - Robust dimension detection for masks (original image size, cropped image size, or bbox size)

# 📤 OUTPUT:
# - image • Resulting image after blending the crop into the original according to bbox and mask
# """

# class FRED_ImageUncropFromBBox:
    # @classmethod
    # def INPUT_TYPES(cls):
        # return {
            # "required": {
                # "original_image": ("IMAGE",),
                # "cropped_image": ("IMAGE",),
                # "bbox": ("BBOX",),
                # "border_blending": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                # "use_mask": ("BOOLEAN", {"default": False}),
                # "resize_mode": (["bilinear", "bicubic", "nearest", "area"], {"default": "bicubic"}),
                # "bbox_mode": (["x0y0x1y1", "xywh"], {"default": "x0y0x1y1"}),
            # },
            # "optional": {
                # "optional_mask": ("MASK",),
            # },
        # }

    # RETURN_TYPES = ("IMAGE", "STRING")
    # RETURN_NAMES = ("image", "help")
    # FUNCTION = "uncrop"
    # CATEGORY = "👑FRED/image/postprocessing"

    # def uncrop(
        # self,
        # original_image,
        # cropped_image,
        # bbox,
        # border_blending,
        # use_mask,
        # resize_mode="bicubic",
        # bbox_mode="x0y0x1y1",
        # optional_mask=None
    # ):
        # """
        # Uncrop images using bounding boxes and optional masks.
        # Handles batch processing, multiple bboxes, and robust mask dimension matching.
        # """

        # # Ensure tensors are in proper format: (B, H, W, C) for images
        # if original_image.dim() == 3:
            # original_image = original_image.unsqueeze(0)

        # B, H_orig, W_orig, C = original_image.shape

        # # Handle bbox format
        # if not isinstance(bbox, list):
            # bboxes = [bbox]
        # elif len(bbox) > 0 and not isinstance(bbox[0], (list, tuple)):
            # bboxes = [bbox]
        # else:
            # bboxes = bbox

        # # Determine batch size
        # num_images = len(cropped_image)
        # num_bboxes = len(bboxes)
        # num_masks = len(optional_mask) if optional_mask is not None else 0

        # # Find maximum batch size
        # max_batch = max(num_images, num_bboxes, num_masks if optional_mask is not None else 0)

        # # Duplicate single items to match batch
        # if num_images == 1 and max_batch > 1:
            # cropped_image = cropped_image.repeat(max_batch, 1, 1, 1)
            # num_images = max_batch

        # if num_bboxes == 1 and max_batch > 1:
            # bboxes = bboxes * max_batch
            # num_bboxes = max_batch

        # if optional_mask is not None and num_masks == 1 and max_batch > 1:
            # optional_mask = optional_mask.repeat(max_batch, 1, 1)
            # num_masks = max_batch

        # # Validate counts match
        # if not (num_images == num_bboxes == (num_masks if optional_mask is not None else num_bboxes)):
            # raise ValueError(
                # f"Batch size mismatch: {num_images} images, {num_bboxes} bboxes, "
                # f"{num_masks if optional_mask is not None else 'no'} masks"
            # )

        # # Convert PIL resampling mode
        # resample_mode_map = {
            # "bilinear": Image.Resampling.BILINEAR,
            # "bicubic": Image.Resampling.BICUBIC,
            # "nearest": Image.Resampling.NEAREST,
            # "area": Image.Resampling.BOX,
        # }
        # pil_resample = resample_mode_map.get(resize_mode, Image.Resampling.BICUBIC)

        # # Process each image
        # result_images = []

        # for i in range(num_images):
            # # Get current items
            # orig_tensor = original_image[0] if len(original_image) == 1 else original_image[i % len(original_image)]
            # crop_tensor = cropped_image[i]
            # current_bbox = bboxes[i]

            # # Parse bbox
            # x0, y0, x1, y1 = self.parse_bbox(current_bbox, bbox_mode)
            # bbox_w = x1 - x0
            # bbox_h = y1 - y0

            # # Convert tensors to PIL
            # orig_pil = tensor2pil(orig_tensor).convert("RGB")
            # crop_pil = tensor2pil(crop_tensor).convert("RGB")

            # # Resize cropped image to bbox size while preserving aspect ratio
            # crop_resized = ImageOps.fit(crop_pil, (bbox_w, bbox_h), method=pil_resample)

            # # Create or process mask
            # if not use_mask:
                # # Rectangular mask with border blending
                # mask_pil = self.create_rectangular_mask(
                    # bbox_w, bbox_h, border_blending
                # )
            # else:
                # # Use optional mask
                # if optional_mask is None:
                    # raise ValueError("optional_mask is required when use_mask=True")

                # mask_tensor = optional_mask[i]
                # mask_pil = self.process_mask(
                    # mask_tensor, 
                    # orig_pil.size, 
                    # crop_pil.size,
                    # (bbox_w, bbox_h),
                    # (x0, y0, x1, y1),
                    # pil_resample,
                    # border_blending
                # )

            # # Blend images
            # result_pil = self.blend_with_mask(
                # orig_pil, crop_resized, mask_pil, (x0, y0)
            # )

            # result_images.append(pil2tensor(result_pil))

        # # Stack results
        # output_tensor = torch.cat(result_images, dim=0)

        # return (output_tensor, HELP_MESSAGE)

    # def parse_bbox(self, bbox, mode):
        # """Parse bbox based on format mode"""
        # if mode == "xywh":
            # x, y, w, h = [int(v) for v in bbox]
            # return x, y, x + w, y + h
        # else:  # x0y0x1y1
            # x0, y0, x1, y1 = [int(v) for v in bbox]
            # return x0, y0, x1, y1

    # def create_rectangular_mask(self, width, height, border_blending):
        # """Create rectangular mask with gradient borders"""
        # mask = Image.new("L", (width, height), 255)

        # if border_blending > 0:
            # # Calculate border width
            # border_width = int(min(width, height) * border_blending * 0.5)

            # if border_width > 0:
                # # Create gradient mask
                # mask_array = np.ones((height, width), dtype=np.float32)

                # # Apply gradient on all sides
                # for i in range(border_width):
                    # alpha = i / border_width
                    # # Top
                    # if i < height:
                        # mask_array[i, :] *= alpha
                    # # Bottom
                    # if height - 1 - i >= 0:
                        # mask_array[height - 1 - i, :] *= alpha
                    # # Left
                    # if i < width:
                        # mask_array[:, i] *= alpha
                    # # Right
                    # if width - 1 - i >= 0:
                        # mask_array[:, width - 1 - i] *= alpha

                # mask = Image.fromarray((mask_array * 255).astype(np.uint8), mode="L")

                # # Apply gaussian blur for smoother transition
                # blur_radius = max(1, border_width // 4)
                # mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        # return mask

    # def process_mask(self, mask_tensor, orig_size, crop_size, bbox_size, bbox_coords, resample, border_blending):
        # """
        # Process mask tensor and resize to bbox size with robust dimension detection.
        # Detects whether mask matches original image, cropped image, or bbox dimensions.
        # """
        # # Normalize mask tensor to (H, W) format
        # if mask_tensor.dim() == 4:  # (B, C, H, W) or (B, H, W, C)
            # if mask_tensor.shape[1] == 1:  # (B, 1, H, W)
                # mask_tensor = mask_tensor.squeeze(0).squeeze(0)
            # elif mask_tensor.shape[-1] == 1:  # (B, H, W, 1)
                # mask_tensor = mask_tensor.squeeze(0).squeeze(-1)
            # else:  # (B, H, W, C) take first channel
                # mask_tensor = mask_tensor[0, :, :, 0]
        # elif mask_tensor.dim() == 3:
            # if mask_tensor.shape[0] == 1:  # (1, H, W)
                # mask_tensor = mask_tensor.squeeze(0)
            # elif mask_tensor.shape[-1] == 1:  # (H, W, 1)
                # mask_tensor = mask_tensor.squeeze(-1)
            # elif mask_tensor.shape[0] in [3, 4]:  # (C, H, W)
                # mask_tensor = mask_tensor[0, :, :]

        # # Get mask dimensions
        # mask_h, mask_w = mask_tensor.shape[-2:]
        # orig_w, orig_h = orig_size
        # crop_w, crop_h = crop_size
        # bbox_w, bbox_h = bbox_size
        # x0, y0, x1, y1 = bbox_coords

        # # Convert to PIL
        # mask_np = mask_tensor.cpu().numpy()
        # if mask_np.max() <= 1.0:
            # mask_np = (mask_np * 255).astype(np.uint8)
        # mask_pil = Image.fromarray(mask_np, mode="L")

        # # Detect mask context and resize accordingly
        # if (mask_w, mask_h) == (bbox_w, bbox_h):
            # # Mask already matches bbox size
            # print(f"[FRED_ImageUncropFromBBox] Mask matches bbox size: {mask_w}x{mask_h}")
            # resized_mask = mask_pil
        # elif (mask_w, mask_h) == (crop_w, crop_h):
            # # Mask matches original cropped image size
            # print(f"[FRED_ImageUncropFromBBox] Mask matches crop size: {mask_w}x{mask_h}, resizing to bbox")
            # resized_mask = mask_pil.resize((bbox_w, bbox_h), resample=resample)
        # elif (mask_w, mask_h) == (orig_w, orig_h):
            # # Mask matches original full image size - crop to bbox
            # print(f"[FRED_ImageUncropFromBBox] Mask matches original image size: {mask_w}x{mask_h}, cropping to bbox")
            # resized_mask = mask_pil.crop((x0, y0, x1, y1))
        # else:
            # # Unknown size - resize to bbox
            # print(f"[FRED_ImageUncropFromBBox] Mask size {mask_w}x{mask_h} unknown, resizing to bbox {bbox_w}x{bbox_h}")
            # resized_mask = mask_pil.resize((bbox_w, bbox_h), resample=resample)

        # # Apply border blending to mask edges
        # if border_blending > 0:
            # mask_array = np.array(resized_mask).astype(np.float32) / 255.0

            # # Create distance transform for smooth edges
            # from scipy.ndimage import distance_transform_edt

            # # Invert mask for distance calculation
            # binary_mask = (mask_array > 0.5).astype(np.uint8)

            # if np.any(binary_mask):
                # # Calculate distance from edge
                # dist = distance_transform_edt(binary_mask)

                # # Apply gradient based on border_blending
                # blend_distance = max(1, int(min(bbox_w, bbox_h) * border_blending * 0.3))

                # # Create gradient
                # gradient = np.clip(dist / blend_distance, 0, 1)

                # # Combine with original mask
                # mask_array = mask_array * gradient

            # resized_mask = Image.fromarray((mask_array * 255).astype(np.uint8), mode="L")

            # # Apply slight blur for smoothness
            # blur_radius = max(1, int(min(bbox_w, bbox_h) * border_blending * 0.1))
            # resized_mask = resized_mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        # return resized_mask

    # def blend_with_mask(self, orig_pil, crop_pil, mask_pil, paste_coords):
        # """
        # Blend cropped image into original using mask at specified coordinates.
        # """
        # x0, y0 = paste_coords

        # # Ensure all images are RGB
        # orig_pil = orig_pil.convert("RGB")
        # crop_pil = crop_pil.convert("RGB")

        # # Convert to numpy for blending
        # orig_array = np.array(orig_pil).astype(np.float32)
        # crop_array = np.array(crop_pil).astype(np.float32)
        # mask_array = np.array(mask_pil).astype(np.float32) / 255.0

        # # Expand mask to 3 channels
        # if mask_array.ndim == 2:
            # mask_array = np.stack([mask_array] * 3, axis=-1)

        # # Get dimensions
        # crop_h, crop_w = crop_array.shape[:2]
        # orig_h, orig_w = orig_array.shape[:2]

        # # Calculate paste region with bounds checking
        # x1 = min(x0 + crop_w, orig_w)
        # y1 = min(y0 + crop_h, orig_h)

        # # Adjust crop if needed
        # actual_w = x1 - x0
        # actual_h = y1 - y0

        # # Blend in the region
        # orig_array[y0:y1, x0:x1] = (
            # orig_array[y0:y1, x0:x1] * (1 - mask_array[:actual_h, :actual_w]) +
            # crop_array[:actual_h, :actual_w] * mask_array[:actual_h, :actual_w]
        # )

        # # Convert back to PIL
        # result_pil = Image.fromarray(orig_array.astype(np.uint8), mode="RGB")

        # return result_pil

    # @staticmethod
    # def IS_CHANGED(*args, **kwargs):
        # return False


# NODE_CLASS_MAPPINGS = {
    # "FRED_ImageUncropFromBBox": FRED_ImageUncropFromBBox
# }

# NODE_DISPLAY_NAME_MAPPINGS = {
    # "FRED_ImageUncropFromBBox": "👑 FRED_ImageUncropFromBBox"
# }