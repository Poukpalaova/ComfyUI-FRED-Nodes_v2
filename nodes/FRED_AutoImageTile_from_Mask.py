import torch
import numpy as np
import cv2
from nodes import MAX_RESOLUTION

HELP_MESSAGE = """
👑 FRED_AutoImageTile_from_Mask

🔹 PURPOSE:
Split an image into tiles (grid) automatically or manually, based on mask/BBOX, with optional overlap and preview.

📥 INPUTS:
- image • image to tile
- MASK / BBOX • optional input for bounding box
- auto_from_mask_bbox • auto-calculate rows/cols from mask or bbox
- bbox_mode • bbox format: x0y0x1y1 or xywh
- input_bbox_management • how to handle multiple bboxes
- Auto_grid_strategy • choose max/min/balanced rows/cols
- max_divisions • limit max number of rows/cols in auto mode
- confidence • shrink or expand bbox
- manual_rows / manual_cols • manual tiling if auto disabled
- overlap / overlap_x / overlap_y • overlap between tiles
- preview_bbox_color • color for preview overlay

⚙️ KEY OPTIONS:
- Auto grid selection strategies (min/max/balanced)
- Max divisions limiter to prevent excessive grids
- BBOX mode selector (x0y0x1y1 or xywh)
- BBOX management (none_if_multi/first/last/all)
- Overlap as ratio or absolute pixels
- Confidence scaling of mask/bbox
- Preview overlay shows grid lines + bbox
- 1x1 grid forbidden (at least one dimension must be > 1)

📤 OUTPUTS:
- IMAGES • generated tiles
- preview • preview overlay image
- tile_width / tile_height • tile dimensions
- overlap_x / overlap_y • applied overlap
- rows / columns • final grid size
"""

class FRED_AutoImageTile_from_Mask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "auto_from_mask_bbox": ("BOOLEAN", { "default": True }),
                "bbox_mode": (["x0y0x1y1", "xywh"], {"default": "x0y0x1y1"}),
                "input_bbox_management": (["none_if_multi", "first", "last", "all"], {"default": "all"}),
                "Auto_grid_strategy": (["max_rows_cols", "min_rows_cols", "max_rows_min_cols", "min_rows_max_cols", "min_balanced", "max_balanced"], {"default": "min_balanced"}),
                "max_divisions": ("INT", {"default": 5, "min": 2, "max": 256, "step": 1}),
                "confidence": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.0, "step": 0.1}),
                "manual_rows": ("INT", { "default": 3, "min": 1, "max": 256, "step": 1 }),
                "manual_cols": ("INT", { "default": 3, "min": 1, "max": 256, "step": 1 }),
                "overlap": ("FLOAT", { "default": 0, "min": 0, "max": 0.5, "step": 0.01 }),
                "overlap_x": ("INT", { "default": 56, "min": 0, "max": MAX_RESOLUTION//2, "step": 1 }),
                "overlap_y": ("INT", { "default": 56, "min": 0, "max": MAX_RESOLUTION//2, "step": 1 }),
                "preview_bbox_color": ("COLOR", {"default": "#FFC800", "widgetType": "MTB_COLOR"},),
            },
            "optional": {
                "MASK": ("MASK",),
                "BBOX": ("BBOX",),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "INT", "INT", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("IMAGES", "preview", "tile_width", "tile_height", "overlap_x", "overlap_y", "rows", "columns", "help")
    FUNCTION = "execute"
    CATEGORY = "👑FRED/image/postprocessing"

    def execute(self, image, manual_rows, manual_cols, overlap, overlap_x, overlap_y, preview_bbox_color, Auto_grid_strategy, max_divisions=32, bbox_mode="x0y0x1y1", input_bbox_management="all", auto_from_mask_bbox=False, confidence=1.0, MASK=None, BBOX=None):
        rows = manual_rows
        cols = manual_cols

        if len(image.shape) == 4:
            _, h, w, _ = image.shape
        elif len(image.shape) == 3:
            _, h, w = image.shape
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")

        # --- Helper functions ---
        def convert_to_xywh(bbox_array, mode):
            """Convert bbox to xywh format based on input mode"""
            if mode == "x0y0x1y1":
                if bbox_array.ndim == 1:
                    x0, y0, x1, y1 = bbox_array
                    return np.array([x0, y0, x1 - x0, y1 - y0])
                else:  # 2D array
                    x0, y0, x1, y1 = bbox_array[:, 0], bbox_array[:, 1], bbox_array[:, 2], bbox_array[:, 3]
                    return np.column_stack([x0, y0, x1 - x0, y1 - y0])
            else:  # xywh
                return bbox_array

        def find_all_divisions(total, start, length, min_grid=1):
            """Find all valid divisions where bbox fits in one tile"""
            results = []
            for div in range(256, min_grid - 1, -1):
                step = total // div
                if step <= 0:
                    continue
                if (start % step + length) <= step:
                    results.append(div)
            return results

        # --- Auto rows/cols logic ---
        original_bbox = None  # Track original bbox for preview
        if auto_from_mask_bbox:
            mask_bbox = None
            if MASK is not None:
                mask_np = MASK.cpu().numpy() if hasattr(MASK, 'cpu') else MASK
                mask_np = np.squeeze(mask_np)
                if mask_np.ndim == 3:
                    mask_np = mask_np[0]
                elif mask_np.ndim == 4:
                    mask_np = mask_np[0, 0]
                ys, xs = np.where(mask_np > 0)
                if len(xs) > 0 and len(ys) > 0:
                    x_min, x_max = xs.min(), xs.max()
                    y_min, y_max = ys.min(), ys.max()
                    mask_bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
                    original_bbox = mask_bbox
            elif BBOX is not None:
                # Handle BBOX input with format detection
                if hasattr(BBOX, "cpu"):
                    bbox_vals = BBOX.cpu().numpy()
                else:
                    bbox_vals = np.array(BBOX)
                
                # Check if we have multiple bboxes (2D array with multiple rows)
                if bbox_vals.ndim == 2 and bbox_vals.shape[0] > 1:
                    num_bboxes = bbox_vals.shape[0]
                    print(f"⚠️ Multiple bboxes detected ({num_bboxes} bboxes).")
                    
                    if input_bbox_management == "none_if_multi":
                        print(f"⚠️ input_bbox_management='none_if_multi': Falling back to manual rows/cols due to multiple bboxes.")
                        mask_bbox = None
                    elif input_bbox_management == "first":
                        bbox_xywh = convert_to_xywh(bbox_vals[0], bbox_mode)
                        mask_bbox = tuple(float(v) for v in bbox_xywh)
                        original_bbox = mask_bbox
                        print(f"📦 Using FIRST bbox (converted to xywh): {mask_bbox}")
                    elif input_bbox_management == "last":
                        bbox_xywh = convert_to_xywh(bbox_vals[-1], bbox_mode)
                        mask_bbox = tuple(float(v) for v in bbox_xywh)
                        original_bbox = mask_bbox
                        print(f"📦 Using LAST bbox (converted to xywh): {mask_bbox}")
                    elif input_bbox_management == "all":
                        # Convert ALL bboxes to xywh first
                        bboxes_xywh = convert_to_xywh(bbox_vals, bbox_mode)
                        
                        # Now calculate bounding box that encompasses all (in xywh space)
                        x0s = bboxes_xywh[:, 0]
                        y0s = bboxes_xywh[:, 1]
                        x1s = bboxes_xywh[:, 0] + bboxes_xywh[:, 2]
                        y1s = bboxes_xywh[:, 1] + bboxes_xywh[:, 3]
                        
                        x0_min = x0s.min()
                        y0_min = y0s.min()
                        x1_max = x1s.max()
                        y1_max = y1s.max()
                        
                        mask_bbox = (float(x0_min), float(y0_min), float(x1_max - x0_min), float(y1_max - y0_min))
                        original_bbox = mask_bbox
                        print(f"📦 Using ALL bboxes: combined bbox xywh={mask_bbox}")
                else:
                    # Single bbox - convert to xywh
                    bbox_xywh = convert_to_xywh(bbox_vals.flatten(), bbox_mode)
                    
                    if bbox_xywh.shape[0] == 4:
                        mask_bbox = tuple(float(v) for v in bbox_xywh)
                        original_bbox = mask_bbox
                        print(f"📦 BBOX converted to xywh: {mask_bbox}")
                    else:
                        print(f"❌ Invalid BBOX format: expected 4 values, got {bbox_xywh.shape[0]}")
                        mask_bbox = None
                    
            if mask_bbox is not None:
                x, y, bw, bh = mask_bbox
                
                # confidence: 1.0 = original bbox, <1.0 = shrink bbox
                scaling = max(min(confidence, 1.0), 0.1)
                cx, cy = x + bw / 2, y + bh / 2
                new_bw = max(int(bw * scaling), 1)
                new_bh = max(int(bh * scaling), 1)
                new_x = max(0, int(cx - new_bw / 2))
                new_y = max(0, int(cy - new_bh / 2))
                new_bw = min(w - new_x, new_bw)
                new_bh = min(h - new_y, new_bh)
                x, y, bw, bh = new_x, new_y, new_bw, new_bh
                
                row_divs = find_all_divisions(h, y, bh, min_grid=1)
                col_divs = find_all_divisions(w, x, bw, min_grid=1)
                
                # Apply max_divisions limit
                row_divs = [d for d in row_divs if d <= max_divisions]
                col_divs = [d for d in col_divs if d <= max_divisions]
                
                # Build valid pairs excluding 1x1 grid
                grid_pairs = []
                for r in row_divs:
                    for c in col_divs:
                        if r == 1 and c == 1:
                            continue  # Forbid 1x1 grid
                        grid_pairs.append((r, c))

                if grid_pairs:
                    if Auto_grid_strategy == "max_rows_cols":
                        selected_rows = max(row_divs) if row_divs else rows
                        selected_cols = max(col_divs) if col_divs else cols
                        if selected_rows == 1 and selected_cols == 1:
                            # Force at least one dimension > 1
                            selected_cols = 2 if 2 in col_divs else selected_cols
                            if selected_rows == 1 and selected_cols == 1:
                                selected_rows = 2
                        rows, cols = selected_rows, selected_cols
                    elif Auto_grid_strategy == "min_rows_cols":
                        selected_rows = min(row_divs) if row_divs else rows
                        selected_cols = min(col_divs) if col_divs else cols
                        if selected_rows == 1 and selected_cols == 1:
                            selected_cols = 2 if 2 in col_divs else selected_cols
                            if selected_rows == 1 and selected_cols == 1:
                                selected_rows = 2
                        rows, cols = selected_rows, selected_cols
                    elif Auto_grid_strategy == "max_rows_min_cols":
                        rows = max(row_divs)
                        cols = min(col_divs)
                        if rows == 1 and cols == 1:
                            rows = 2 if 2 in row_divs else rows
                    elif Auto_grid_strategy == "min_rows_max_cols":
                        rows = min(row_divs)
                        cols = max(col_divs)
                        if rows == 1 and cols == 1:
                            cols = 2 if 2 in col_divs else cols
                    elif Auto_grid_strategy in ["min_balanced", "max_balanced"]:
                        sorted_pairs = sorted(
                            grid_pairs,
                            key=lambda pair: (abs(pair[0] - pair[1]), -pair[0]*pair[1] if Auto_grid_strategy == "max_balanced" else pair[0]*pair[1])
                        )
                        rows, cols = sorted_pairs[0]
                else:
                    print(f"⚠️ No valid grid found (excluding 1x1). Falling back to manual rows/cols.")
                    preview_bbox_color = "#FF0000"  # Fallback color for no match
        # --- End auto logic ---

        tile_h = h // rows
        tile_w = w // cols
        h_adj = tile_h * rows
        w_adj = tile_w * cols
        overlap_h = int(tile_h * overlap) + overlap_y
        overlap_w = int(tile_w * overlap) + overlap_x

        # max overlap is half of the tile size
        overlap_h = min(tile_h // 2, overlap_h)
        overlap_w = min(tile_w // 2, overlap_w)

        if rows == 1:
            overlap_h = 0
        if cols == 1:
            overlap_w = 0

        tiles = []
        for i in range(rows):
            for j in range(cols):
                y1 = i * tile_h
                x1 = j * tile_w

                if i > 0:
                    y1 -= overlap_h
                if j > 0:
                    x1 -= overlap_w

                y2 = y1 + tile_h + overlap_h
                x2 = x1 + tile_w + overlap_w

                if y2 > h:
                    y2 = h
                    y1 = y2 - tile_h - overlap_h
                if x2 > w:
                    x2 = w
                    x1 = x2 - tile_w - overlap_w

                tiles.append(image[:, y1:y2, x1:x2, :])
        tiles = torch.cat(tiles, dim=0)

        # 1. Create a white overlay image (uint8)
        overlay_image = torch.full((1, h, w, 3), 255, dtype=torch.uint8, device=image.device)

        # 2. Draw the yellow grid on the overlay (still uint8)
        overlay_image = self.draw_grid_overlay(overlay_image, rows, cols, tile_h, tile_w, overlap_h, overlap_w, preview_bbox_color)

        # 3. Convert overlay to float [0,1]
        overlay_image_float = overlay_image.float() / 255.0

        # 4. Make sure preview is a float tensor [0,1]
        preview = image.clone() if len(image.shape) == 4 else image.clone().unsqueeze(0)
        preview_float = preview.float() / 255.0 if preview.max() > 1.0 else preview.float()

        # 5. Blend preview and overlay (0.6 = 60% overlay, 30% original)
        blend_preview = self.blend_images(preview_float, overlay_image_float, 0.6)

        # 6. Convert to numpy for any further OpenCV drawing
        blend_preview_np = (blend_preview[0].cpu().numpy() * 255).astype(np.uint8)
        blend_preview_np = np.ascontiguousarray(blend_preview_np)

        # 7. Draw rectangles for original and confidence-adjusted bboxes
        preview_color = torch.tensor(self.Hex_to_RGB(preview_bbox_color), dtype=torch.uint8, device=image.device)
        
        if auto_from_mask_bbox and original_bbox is not None:
            # Draw original bbox in purple (original_bbox is always in xywh format now)
            orig_x, orig_y, orig_bw, orig_bh = original_bbox
            orig_pt1 = (int(orig_x), int(orig_y))
            orig_pt2 = (int(orig_x + orig_bw), int(orig_y + orig_bh))
            cv2.rectangle(blend_preview_np, orig_pt1, orig_pt2, self.Hex_to_RGB("#710193"), 2)
            
            # Draw confidence-adjusted bbox in yellow (if confidence != 1.0)
            if confidence != 1.0 and mask_bbox is not None:
                x_int = int(x)
                y_int = int(y)
                bw_int = int(bw)
                bh_int = int(bh)
                pt1 = (x_int, y_int)
                pt2 = (x_int + bw_int, y_int + bh_int)
                cv2.rectangle(blend_preview_np, pt1, pt2, (int(preview_color[0]), int(preview_color[1]), int(preview_color[2])), 4)

        # 8. Convert back to tensor for output
        preview = torch.from_numpy(blend_preview_np).unsqueeze(0).float() / 255.0

        # 9. Return as ComfyUI expects (tensor, float, [0,1])
        return (tiles, preview, tile_w+overlap_w, tile_h+overlap_h, overlap_w, overlap_h, rows, cols, HELP_MESSAGE)

    def draw_grid_overlay(self, image, rows, cols, tile_h, tile_w, overlap_h, overlap_w, color_hex):
        """
        Draw grid overlay with FIXED grid lines and overlap zones centered around them.
        Grid lines stay at c*tile_w and r*tile_h regardless of overlap values.
        """
        preview = image.clone()
        color = torch.tensor(self.Hex_to_RGB(color_hex), dtype=preview.dtype, device=preview.device)
        black = torch.tensor([0, 0, 0], dtype=preview.dtype, device=preview.device)
        B, H, W, C = preview.shape

        # --- First pass: Draw colored overlap zones centered on FIXED grid positions ---
        for c in range(1, cols):
            # FIXED grid line position (never changes with overlap)
            x = c * tile_w
            
            # Draw overlap zone centered on this fixed position
            x_start = max(0, x - overlap_w // 2)
            x_end = min(W, x + (overlap_w + 1) // 2)
            
            if x_start < x_end:
                preview[:, :, x_start:x_end, :3] = color

        for r in range(1, rows):
            # FIXED grid line position (never changes with overlap)
            y = r * tile_h
            
            # Draw overlap zone centered on this fixed position
            y_start = max(0, y - overlap_h // 2)
            y_end = min(H, y + (overlap_h + 1) // 2)
            
            if y_start < y_end:
                preview[:, y_start:y_end, :, :3] = color

        # --- Second pass: Draw black center lines at FIXED grid positions ---
        for c in range(1, cols):
            x = c * tile_w  # FIXED position
            
            # Draw 2-pixel black line at center
            bx_start = max(0, x - 1)
            bx_end = min(W, x + 1)
            
            if bx_start < bx_end:
                preview[:, :, bx_start:bx_end, :3] = black

        for r in range(1, rows):
            y = r * tile_h  # FIXED position
            
            # Draw 2-pixel black line at center
            by_start = max(0, y - 1)
            by_end = min(H, y + 1)
            
            if by_start < by_end:
                preview[:, by_start:by_end, :, :3] = black

        return preview

    def blend_images(self, image1: torch.Tensor, image2: torch.Tensor, blend_factor: float):
        """Multiplicative blend that preserves brightness (original logic)"""
        if image1.shape != image2.shape:
            image2 = self.crop_and_resize(image2, image1.shape)

        blended_image = image1 * image2
        blended_image = image1 * (1 - blend_factor) + blended_image * blend_factor
        blended_image = torch.clamp(blended_image, 0, 1)
        return blended_image

    def crop_and_resize(self, image: torch.Tensor, target_shape: tuple):
        """Helper function to match image dimensions"""
        B, H, W, C = target_shape
        return torch.nn.functional.interpolate(
            image.permute(0, 3, 1, 2),
            size=(H, W),
            mode='bilinear',
            align_corners=False
        ).permute(0, 2, 3, 1)

    def Hex_to_RGB(self, inhex: str) -> tuple:
        if not inhex.startswith('#'):
            raise ValueError(f'Invalid Hex Code in {inhex}')
        rval = inhex[1:3]
        gval = inhex[3:5]
        bval = inhex[5:]
        return (int(rval, 16), int(gval, 16), int(bval, 16))

# Dictionary mapping node names to their corresponding classes
NODE_CLASS_MAPPINGS = {
    "FRED_AutoImageTile_from_Mask": FRED_AutoImageTile_from_Mask
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FRED_AutoImageTile_from_Mask": "👑 FRED_AutoImageTile_from_Mask"
}