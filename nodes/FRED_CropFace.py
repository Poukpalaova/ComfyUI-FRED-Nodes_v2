# FRED_CropFace — face cropper with RetinaFace (facexlib), robust after moving nodes/ subpkg
# Outputs:
#   0 face_image  (IMAGE)   — cropped face (with margins), batch-preserved (B=1)
#   1 preview     (IMAGE)   — original image with detections + margins overlay
#   2 bbox        (BBOX)    — selected (x,y,w,h) as a 4-element array
#   3 face_pixel_ratio            (FLOAT) — % of image area (face only)
#   4 face_w_margin_pixel_ratio   (FLOAT) — % of image area (face + margins)
#   5 help        (STRING)  — inline help / usage notes

import os
import sys
import cv2
import torch
import numpy as np

# IMPORTANT: this file is inside package "ComfyUI-FRED-Nodes.nodes"
# Ensure nodes/ and repo root both have __init__.py so relative imports work.
from ..utils import tensor2cv, cv2tensor, hex2bgr, models_dir as MODELS_ROOT  # from repo-root utils.py

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
- face_id • which detected face to crop (sorted left→right, default 0)
- max_size • resize image before detection for performance (default 1536)

⚙️ KEY OPTIONS:
- If no face detected → returns original image and empty bbox
- Margins are applied relative to detected face size and clamped to image bounds
- Face detection weights are stored under `<models>/facexlib` and downloaded on first run
- Works with either pip-installed facexlib or bundled version in `thirdparty/facexlib`

📤 OUTPUTS:
- face_image • cropped face (IMAGE)
- preview • original image with detection + margin overlay (IMAGE)
- bbox • bounding box (x,y,w,h) as array
- face_pixel_ratio • % of total pixels occupied by the detected face
- face_w_margin_pixel_ratio • % including applied margins
"""

def _normalize_boxes(ret):
    """
    facexlib.detect_faces can return:
      - list of boxes
      - numpy array of shape (N, …)
      - (boxes, kps) tuple
    Return a plain Python list of boxes (each box iterable).
    """
    if ret is None:
        return []
    # If it's a tuple like (boxes, landmarks)
    if isinstance(ret, tuple) and len(ret) == 2:
        boxes = ret[0]
    else:
        boxes = ret

    # numpy array → list
    try:
        import numpy as _np
        if isinstance(boxes, _np.ndarray):
            return boxes.tolist() if boxes.size else []
    except Exception:
        pass

    # already list/tuple
    if isinstance(boxes, (list, tuple)):
        return list(boxes)

    # single object fallback
    return [boxes]

def _is_empty(seq):
    try:
        return len(seq) == 0
    except Exception:
        return True

def _ensure_facexlib_on_path():
    """Make sure we can import facexlib. Fall back to bundled thirdparty/facexlib."""
    try:
        import facexlib  # noqa: F401
        return
    except Exception:
        here = os.path.dirname(__file__)  # .../ComfyUI-FRED-Nodes/nodes
        local_container = os.path.abspath(os.path.join(here, "..", "thirdparty", "facexlib"))
        # Expect: <repo_root>/thirdparty/facexlib/facexlib/...
        if os.path.isdir(local_container) and local_container not in sys.path:
            sys.path.insert(0, local_container)
        import facexlib  # noqa: F401  # let ImportError bubble if truly missing


class FRED_CropFace:
    """
    Crop a detected face with margin controls.
    Loads RetinaFace at init; robust to repo being moved into /nodes.
    """

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
                "left_margin_factor":   ("FLOAT", {"default": 0.6, "min": 0.0}),
                "right_margin_factor":  ("FLOAT", {"default": 0.6, "min": 0.0}),
                "top_margin_factor":    ("FLOAT", {"default": 0.4, "min": 0.0}),
                "bottom_margin_factor": ("FLOAT", {"default": 1.0, "min": 0.0}),
                "face_id": ("INT", {"default": 0, "min": 0}),
                "max_size": ("INT", {"default": 1536, "min": 256}),
            }
        }

    def __init__(self):
        """
        Load RetinaFace model during node initialization.
        Finds facexlib via pip or bundled thirdparty/facexlib,
        and stores weights under <models>/facexlib.
        """
        _ensure_facexlib_on_path()
        try:
            from facexlib.detection import init_detection_model
        except Exception as e:
            raise RuntimeError(
                "[FRED_CropFace] facexlib not found. Install `pip install facexlib` "
                "or place the vendored package at <repo>/thirdparty/facexlib"
            ) from e

        self.models_dir = os.path.join(MODELS_ROOT, "facexlib")
        os.makedirs(self.models_dir, exist_ok=True)

        # retinaface_resnet50 is accurate; switch to 'retinaface_mnet025' for lighter model if needed.
        self.model = init_detection_model("retinaface_resnet50", model_rootpath=self.models_dir)

    @staticmethod
    def _visualize_detection(img_bgr: np.ndarray, bboxes_and_landmarks):
        """Draw detection rectangles + confidence + 5 landmarks."""
        img = np.copy(img_bgr)
        for b in bboxes_and_landmarks:
            # confidence
            cv2.putText(
                img, f"{b[4]:.4f}", (int(b[0]), int(b[1] + 12)),
                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255)
            )
            bi = list(map(int, b))
            cv2.rectangle(img, (bi[0], bi[1]), (bi[2], bi[3]), (0, 0, 255), 2)
            # landmarks
            for i in range(5):
                cv2.circle(img, (bi[5 + i * 2], bi[6 + i * 2]), 1, (0, 0, 255), 4)
        return img

    @staticmethod
    def _visualize_margin(img_bgr: np.ndarray, bboxes_xywh):
        """Draw margin-augmented boxes in purple."""
        img = np.copy(img_bgr)
        color = hex2bgr("#710193")
        for x, y, w, h in bboxes_xywh:
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        return img

    @staticmethod
    def _add_margin(bbox_xywh, l, r, t, b, W, H):
        """Expand bbox by given factors, clamp inside image."""
        x, y, w, h = bbox_xywh
        left   = int(l * w)
        right  = int(r * w)
        top    = int(t * h)
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
        max_size: int,
    ):
        """
        Detect faces and crop selected face with margins.
        """
        # Convert to OpenCV (BGR)
        img_cv = tensor2cv(image)  # HxWxC, uint8
        H, W = img_cv.shape[:2]

        # Resize for faster detection if too large
        scale = 1.0
        if max(W, H) > max_size:
            scale = max_size / float(max(W, H))
            newW = max(1, int(W * scale))
            newH = max(1, int(H * scale))
            img_resized = cv2.resize(img_cv, (newW, newH), interpolation=cv2.INTER_AREA)
        else:
            img_resized = img_cv

        # Detect
        with torch.no_grad():
            raw = self.model.detect_faces(img_resized, confidence)

        bboxes = _normalize_boxes(raw)

        if _is_empty(bboxes):
            print("[FRED_CropFace] WARNING: No face detected. Adjust confidence / picture. Passing through.")
            empty_bbox = np.zeros((4,), dtype=np.int32)
            return image, image, empty_bbox, 0.0, 0.0, HELP_TEXT

        # Rescale boxes/landmarks back to original
        inv = 1.0 / scale
        bboxes = [
            (
                x0 * inv, y0 * inv, x1 * inv, y1 * inv, score,
                *[p * inv for p in points]
            )
            for (x0, y0, x1, y1, score, *points) in bboxes
        ]

        # Sort left→right, pick by face_id
        bboxes.sort(key=lambda b: b[0])
        if face_id >= len(bboxes):
            print(f"[FRED_CropFace] face_id {face_id} out of range; using 0 (found {len(bboxes)})")
            face_id = 0

        # Preview (raw detections)
        preview = self._visualize_detection(img_cv, bboxes)

        # Build margin-augmented xywh for each detected face
        xywh_list = []
        for (x0, y0, x1, y1, *_) in bboxes:
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

        # Overlay margins too
        preview = self._visualize_margin(preview, xywh_list)

        # Selected face
        x, y, w, h = xywh_list[face_id]
        # Clamp to image
        x1 = min(x + w, W)
        y1 = min(y + h, H)

        # Areas / ratios
        # (Original face without margin)
        ox0, oy0, ox1, oy1 = map(int, bboxes[face_id][:4])
        ofw = abs(ox1 - ox0)
        ofh = abs(oy1 - oy0)
        face_pixels = max(0, ofw) * max(0, ofh)

        total_pixels = H * W if H and W else 0
        face_pixel_ratio = (face_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0
        face_w_margin_pixel_ratio = ((x1 - x) * (y1 - y) / total_pixels) * 100.0 if total_pixels > 0 else 0.0

        # Crop from input tensor: [B,H,W,C]
        cropped_face = image[0, y:y1, x:x1, :].unsqueeze(0)

        # Pack bbox as a 4-vector (x,y,w,h)
        bbox_arr = np.array([x, y, x1 - x, y1 - y], dtype=np.int32)

        return (
            cropped_face,
            cv2tensor(preview),
            bbox_arr,
            float(face_pixel_ratio),
            float(face_w_margin_pixel_ratio),
            HELP_MESSAGE,
        )


# Registration
NODE_CLASS_MAPPINGS = {
    "FRED_CropFace": FRED_CropFace
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FRED_CropFace": "👑 FRED_CropFace"
}
