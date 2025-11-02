# FRED_CropFace — face cropper with RetinaFace (facexlib), robust after moving nodes/ subpkg
# Outputs:
#   0 face_image  (IMAGE)   — cropped face (with margins), batch-preserved (B=1) or list if face_id=-1
#   1 preview     (IMAGE)   — original image with detections + margins overlay (unique image)
#   2 bbox        (BBOX)    — selected bbox(es) as list(s) of 4 elements
#   3 face_pixel_ratio            (FLOAT) — % of image area (face only)
#   4 face_w_margin_pixel_ratio   (FLOAT) — % of image area (face + margins)
#   5 help        (STRING)  — inline help / usage notes

import os
import sys
import cv2
import torch
import numpy as np

from ..utils import tensor2cv, cv2tensor, hex2bgr, models_dir as MODELS_ROOT

HELP_MESSAGE = """
👑 FRED_CropFace

🔹 PURPOSE:
Detect faces using RetinaFace (via facexlib) and crop one of them with customizable margins.
Useful for isolating faces while keeping context.

📥 INPUTS:
- image • input image (IMAGE)
- confidence • detection confidence threshold (default 0.8)
- left_margin_factor / right_margin_factor / top_margin_factor / bottom_margin_factor
  • expand bounding box by proportional margins
- face_id • which detected face to crop (sorted left→right, default 0, -1 for all)
- max_size • resize image before detection for performance (default 1536)
- bbox_mode • output bbox format: "x0y0x1y1" (default) or "xywh"

⚙️ KEY OPTIONS:
- If no face detected → returns original image and empty bbox
- Margins are applied relative to detected face size and clamped to image bounds
- Face detection weights are stored under <models>/facexlib and downloaded on first run
- Works with either pip-installed facexlib or bundled version

📤 OUTPUTS:
- face_image • cropped face(s) (IMAGE)
- preview • original image with detection + margin overlay (IMAGE)
- bbox • bounding box(es) as list(s)
- face_pixel_ratio • % of total pixels occupied by the detected face
- face_w_margin_pixel_ratio • % including applied margins
"""

def _normalize_boxes(ret):
    if ret is None:
        return []
    if isinstance(ret, tuple) and len(ret) == 2:
        boxes = ret[0]
    else:
        boxes = ret
    try:
        import numpy as _np
        if isinstance(boxes, _np.ndarray):
            return boxes.tolist() if boxes.size else []
    except Exception:
        pass
    if isinstance(boxes, (list, tuple)):
        return list(boxes)
    return [boxes]

def _is_empty(seq):
    try:
        return len(seq) == 0
    except Exception:
        return True

def _ensure_facexlib_on_path():
    try:
        import facexlib
        return
    except Exception:
        here = os.path.dirname(__file__)
        local_container = os.path.abspath(os.path.join(here, "..", "thirdparty", "facexlib"))
        if os.path.isdir(local_container) and local_container not in sys.path:
            sys.path.insert(0, local_container)
        import facexlib

class FRED_CropFace:
    RETURN_TYPES = ("IMAGE", "IMAGE", "BBOX", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = (
        "face_image",
        "preview",
        "bbox",
        "face_pixel_ratio",
        "face_w_margin_pixel_ratio",
        "help",
    )
    FUNCTION = "crop"
    CATEGORY = "👑FRED/image/postprocessing"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "confidence": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0}),
                "left_margin_factor": ("FLOAT", {"default": 0.6, "min": 0.0}),
                "right_margin_factor": ("FLOAT", {"default": 0.6, "min": 0.0}),
                "top_margin_factor": ("FLOAT", {"default": 0.4, "min": 0.0}),
                "bottom_margin_factor": ("FLOAT", {"default": 1.0, "min": 0.0}),
                "face_id": ("INT", {"default": 0, "min": -1}),
                "min_face_ratio": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 100.0}),
                "max_size": ("INT", {"default": 1536, "min": 256}),
                "bbox_mode": (["x0y0x1y1", "xywh"], {"default": "x0y0x1y1"}),
            }
        }

    def __init__(self):
        _ensure_facexlib_on_path()
        try:
            from facexlib.detection import init_detection_model
        except Exception as e:
            raise RuntimeError(
                "[FRED_CropFace] facexlib not found. Install `pip install facexlib` "
                "or place vendored package at <repo>/thirdparty/facexlib"
            ) from e

        self.models_dir = os.path.join(MODELS_ROOT, "facexlib")
        os.makedirs(self.models_dir, exist_ok=True)

        self.model = init_detection_model("retinaface_resnet50", model_rootpath=self.models_dir)

    @staticmethod
    def _visualize_detection(img_bgr: np.ndarray, bboxes_and_landmarks):
        img = np.copy(img_bgr)
        for b in bboxes_and_landmarks:
            cv2.putText(
                img, f"{b[4]:.4f}", (int(b[0]), int(b[1] + 12)),
                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255)
            )
            bi = list(map(int, b))
            cv2.rectangle(img, (bi[0], bi[1]), (bi[2], bi[3]), (0, 0, 255), 2)
            for i in range(5):
                cv2.circle(img, (bi[5 + i * 2], bi[6 + i * 2]), 1, (0, 0, 255), 4)
        return img

    @staticmethod
    def _visualize_margin(img_bgr: np.ndarray, bboxes_xywh):
        img = np.copy(img_bgr)
        color = hex2bgr("#710193")
        for x, y, w, h in bboxes_xywh:
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        return img

    @staticmethod
    def _add_margin(bbox_xywh, l, r, t, b, W, H):
        x, y, w, h = bbox_xywh
        left = int(l * w)
        right = int(r * w)
        top = int(t * h)
        bottom = int(b * h)

        x = max(0, x - left)
        y = max(0, y - top)
        w = min(W - x, w + left + right)
        h = min(H - y, h + top + bottom)

        return int(x), int(y), int(w), int(h)

    def crop(
        self,
        image: torch.Tensor,
        confidence: float,
        left_margin_factor: float,
        right_margin_factor: float,
        top_margin_factor: float,
        bottom_margin_factor: float,
        face_id: int,
        min_face_ratio: float,
        max_size: int,
        bbox_mode="x0y0x1y1",
    ):
        img_cv = tensor2cv(image)
        H, W = img_cv.shape[:2]

        scale = 1.0
        if max(W, H) > max_size:
            scale = max_size / float(max(W, H))
            newW = max(1, int(W * scale))
            newH = max(1, int(H * scale))
            img_resized = cv2.resize(img_cv, (newW, newH), interpolation=cv2.INTER_AREA)
        else:
            img_resized = img_cv

        with torch.no_grad():
            raw = self.model.detect_faces(img_resized, confidence)

        bboxes = _normalize_boxes(raw)

        if _is_empty(bboxes):
            print("[FRED_CropFace] WARNING: No face detected. Passing original image.")
            empty_bbox = [0, 0, 0, 0]
            return image, image, empty_bbox, 0.0, 0.0, HELP_MESSAGE

        inv = 1.0 / scale
        bboxes = [
            (
                x0 * inv, y0 * inv, x1 * inv, y1 * inv, score,
                *[p * inv for p in points]
            )
            for (x0, y0, x1, y1, score, *points) in bboxes
        ]

        bboxes.sort(key=lambda b: b[0])
        if face_id >= len(bboxes):
            print(f"[FRED_CropFace] face_id {face_id} out of range; using 0 (found {len(bboxes)})")
            face_id = 0

        preview = self._visualize_detection(img_cv, bboxes)

        xywh_list = []
        for (x0, y0, x1, y1, *_) in bboxes:
            x0, x1 = sorted([int(x0), int(x1)])
            y0, y1 = sorted([int(y0), int(y1)])
            w0 = x1 - x0
            h0 = y1 - y0
            x0i, y0i, x1i, y1i = map(int, (x0, y0, x1, y1))
            w0 = abs(x1i - x0i)
            h0 = abs(y1i - y0i)
            x = min(x0i, x1i)
            y = min(y0i, y1i)
            xywh_list.append(
                self._add_margin(
                    (x, y, w0, h0),
                    left_margin_factor,
                    right_margin_factor,
                    top_margin_factor,
                    bottom_margin_factor,
                    W, H
                )
            )

        preview = self._visualize_margin(preview, xywh_list)

        output_list = []

        total_pixels = H * W if H and W else 0

        for i, (bbox_vals, margin) in enumerate(zip(bboxes, xywh_list)):
            x0, y0, x1, y1 = map(int, bbox_vals[:4])
            mx, my, mw, mh = margin

            # Calcul des pixels de la face sans marges
            face_pixels = max(0, x1 - x0) * max(0, y1 - y0)
            face_pixel_ratio = (face_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0

            if face_pixel_ratio < min_face_ratio:
                continue

            # Découpe de l'image selon bbox_mode
            cropped_face = image[0, my:my+mh, mx:mx+mw, :].squeeze(0)
            if bbox_mode == "x0y0x1y1":
                bbox_arr = [mx, my, mx + mw, my + mh]  # bbox marginée convertie en coins
            else:  # "xywh"
                bbox_arr = [mx, my, mw, mh]

            # Calcul ratio avec marges
            face_w_margin_pixel_ratio = ((mw) * (mh) / total_pixels) * 100 if total_pixels > 0 else 0.0

            output_list.append(
                (
                    cropped_face,
                    cv2tensor(preview),
                    bbox_arr,
                    float(face_pixel_ratio),
                    float(face_w_margin_pixel_ratio),
                    HELP_MESSAGE,
                )
            )
        if face_id == -1:
            if not output_list:
                empty_bbox = [0, 0, 0, 0]
                return image, image, empty_bbox, 0.0, 0.0, HELP_MESSAGE

            batch = list(zip(*output_list))

            if bbox_mode == "x0y0x1y1":
                batch_bbox = [[x, y, x + w, y + h] for (x, y, w, h) in batch[2]]
            else:
                batch_bbox = batch[2]

            return (
                list(batch[0]),
                cv2tensor(preview),
                batch_bbox,
                list(batch[3]),
                list(batch[4]),
                HELP_MESSAGE,
            )
        else:
            # ton code pour face_id unique avec bbox_mode appliqué et découpe marginée
            mx, my, mw, mh = xywh_list[face_id]
            x0, y0, x1, y1 = map(int, bboxes[face_id][:4])

            face_pixels = max(0, x1 - x0) * max(0, y1 - y0)
            total_pixels = H * W if H and W else 0
            face_pixel_ratio = (face_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0
            face_w_margin_pixel_ratio = ((mw) * (mh) / total_pixels) * 100.0 if total_pixels > 0 else 0.0

            cropped_face = image[0, my:my + mh, mx:mx + mw, :].unsqueeze(0)

            if bbox_mode == "x0y0x1y1":
                bboxarr = [mx, my, mx + mw, my + mh]
            else:
                bboxarr = [mx, my, mw, mh]

            return cropped_face, cv2tensor(preview), bboxarr, face_pixel_ratio, face_w_margin_pixel_ratio, HELP_MESSAGE

        # Sinon retour standard valeur unique au face_id sélectionné :
        if len(output_list) == 0:
            empty_bbox = [0, 0, 0, 0]
            return image, image, empty_bbox, 0.0, 0.0, HELP_MESSAGE

        if face_id >= len(output_list):
            face_id = 0

        selected_vals = tuple(val[face_id] for val in zip(*output_list))
        
        for i, ((x, y, w, h), bbox_vals) in enumerate(zip(xywh_list, bboxes)):
            print(f"Face {i} crop box: x={x}, y={y}, w={w}, h={h}")
            print(f"Image crop expected dim = (h, w): ({y+h - y}, {x+w - x})")
            cropped_face = image[0, y:y+h, x:x+w, :].squeeze(0)
            print(f"Cropped face shape: {cropped_face.shape}")

        if bbox_mode == "x0y0x1y1":
            bbox_x = selected_vals[2][0]
            bbox_y = selected_vals[2][1]
            bbox_w = selected_vals[2][2]
            bbox_h = selected_vals[2][3]
            bbox_out = [bbox_x, bbox_y, bbox_x + bbox_w, bbox_y + bbox_h]
            return (
                selected_vals[0],
                selected_vals[1],
                bbox_out,
                selected_vals[3],
                selected_vals[4],
                HELP_MESSAGE,
            )
        else:
            return (
                selected_vals[0],
                selected_vals[1],
                selected_vals[2],
                selected_vals[3],
                selected_vals[4],
                HELP_MESSAGE,
            )

NODE_CLASS_MAPPINGS = {"FRED_CropFace": FRED_CropFace}
NODE_DISPLAY_NAME_MAPPINGS = {"FRED_CropFace": "👑 FRED_CropFace"}