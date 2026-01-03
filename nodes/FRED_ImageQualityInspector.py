import torch
import numpy as np
import cv2
from PIL import Image
from piq import brisque
from skimage import img_as_float32

HELP_MESSAGE = """
👑 FRED_ImageQualityInspector

🔹 PURPOSE:
Analyze image quality from four perspectives:
• BRISQUE (perceptual quality, no reference, via piq)
• Blur (variance of Laplacian, optionally restricted by mask)
• SNR (signal-to-noise ratio, dB, optionally restricted by mask)
• Compression proxy (Raw RGB size vs. JPEG@95 size)

📥 INPUTS:
- image • input image (required)
- mask • optional mask (restricts blur/SNR computation to region)
- scale_output_0_100 • if ON, metrics are normalized to scores (0–100, 100 = best)

⚙️ KEY OPTIONS:
- BRISQUE always computed on the full image.
- Blur/SNR can be restricted by mask region if provided.
- Compression proxy is heuristic: re-encodes JPEG@95 to estimate compressibility.
- scale_output_0_100 allows direct comparison and automation (thresholding, scoring).

📤 OUTPUTS:
- BRISQUE_SCORE • raw or scaled value
- BLUR_VALUE • raw or scaled value
- SNR_VALUE • raw or scaled value
- COMPRESSION_RATIO • raw or scaled value
- help • this message

───────────────────────────────────────────────
WHEN “scale_output_0–100” = ON → normalized SCORES (100 = best)
───────────────────────────────────────────────
• brisque_score: 100 = excellent, 0 = poor.
  Mapping: 100 − clamp(raw_brisque, 0..100).

• blur_score: 100 = sharp, 0 = very blurry.
  Mapping: saturating Laplacian variance map.

• snr_score: 100 = no visible noise, 0 = very noisy.
  Mapping: scaled 0–40 dB (≥40 = 100, ∞ = 100).

• compression_score: 100 = uncompressed/high intrinsic quality,
  0 = oversmoothed or heavily compressible.
  Mapping: raw_RGB_size / JPEG95_size with saturation.

───────────────────────────────────────────────
WHEN “scale_output_0–100” = OFF → RAW METRICS
───────────────────────────────────────────────
• BRISQUE (lower is better; ~0–100+)
  <20 excellent • 20–30 very good • 30–40 OK • >40 questionable.

• BLUR (variance of Laplacian; higher = sharper, unbounded)
  >300 very sharp • 50–300 moderately sharp • <50 likely blurry.

• SNR (dB; higher = cleaner; can be negative or ∞)
  >25 low noise • 15–25 moderate • <15 noisy.

• COMPRESSION_RATIO (raw_size / JPEG95_size; heuristic proxy)
  ~1.0 normal • <0.7 heavily compressed • >1.2 high quality/simple content.

📝 Tips:
- Read metrics together (e.g., sharp + noisy vs. smooth + clean).
- Use mask for subject-focused evaluation (e.g., face area).
- For thresholds/automation, prefer scaled 0–100 scores.
"""

class FRED_ImageQualityInspector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "scale_output_0_100": ("BOOLEAN", {"default": True}),  # 100 = best when True
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("FLOAT", "FLOAT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("BRISQUE", "BLUR", "SNR", "COMPRESSION_RATIO", "help")
    FUNCTION = "analyze_image"
    CATEGORY = "👑FRED/postprocessing"

    # ------------------------- helpers (scaling) -------------------------
    @staticmethod
    def _scale_brisque(raw: float) -> float:
        # BRISQUE lower is better. Map to score where 100=best.
        # Clamp raw into [0,100] then invert.
        if raw is None or np.isnan(raw):
            return 0.0
        return float(np.clip(100.0 - np.clip(raw, 0.0, 100.0), 0.0, 100.0))

    @staticmethod
    def _scale_blur(lap_var: float, baseline: float = 150.0) -> float:
        # Laplacian variance: higher -> sharper. Convert to 0..100 score (100=sharp).
        # Saturating map: score = 100 * var / (var + baseline)
        lap_var = max(0.0, float(lap_var))
        return float(100.0 * lap_var / (lap_var + baseline + 1e-9))

    @staticmethod
    def _scale_snr(snr_db: float, lo: float = 0.0, hi: float = 40.0) -> float:
        # Map SNR dB into [0..100]: <=lo → 0, >=hi → 100. ∞ → 100.
        if snr_db == float("inf"):
            return 100.0
        if snr_db is None or np.isnan(snr_db):
            return 0.0
        return float(np.clip((snr_db - lo) * (100.0 / max(1e-6, hi - lo)), 0.0, 100.0))

    @staticmethod
    def _scale_compression(ratio: float, hi: float = 80.0) -> float:
        # ratio = raw_size / JPEG95_size. Larger → more “uncompressed” / more compressible.
        # Map to score where 100 is “no compression / highly compressible” (saturates).
        if ratio is None or ratio <= 0 or np.isnan(ratio):
            return 0.0
        return float(np.clip((ratio - 1.0) * (100.0 / max(1e-6, hi - 1.0)), 0.0, 100.0))

    # ------------------------------ core -------------------------------
    def analyze_image(self, image, scale_output_0_100=True, mask=None):
        # Convert ComfyUI image (1, H, W, C) or (B, H, W, C) to numpy (H, W, C) in [0,1]
        if image.ndim == 4:
            image_tensor = image[0]  # (H, W, C)
        else:
            image_tensor = image  # fallback

        image_np = image_tensor.cpu().numpy()
        if image_np.shape[-1] != 3 and image_np.shape[0] == 3:
            # (C, H, W) -> (H, W, C)
            image_np = np.transpose(image_np, (1, 2, 0))
        image_np = np.clip(image_np, 0.0, 1.0).astype(np.float32)
        h, w = image_np.shape[:2]

        # ---------- BRISQUE (piq) – full image ----------
        # expects (N,3,H,W), float32, [0,1]
        image_brisque = img_as_float32(image_np)
        image_brisque = torch.from_numpy(image_brisque).permute(2, 0, 1).unsqueeze(0).float()
        try:
            brisque_raw = float(brisque(image_brisque, data_range=1.0).item())
        except Exception:
            brisque_raw = float("nan")

        # ---------- Mask for blur/SNR ----------
        if mask is not None:
            mask_np = mask[0].cpu().numpy()
            if mask_np.ndim == 3:
                mask_np = mask_np.squeeze()
            mask_bin = (mask_np > 0.5).astype(np.uint8)
            if mask_bin.shape != (h, w):
                # conservative fallback to full area if sizes mismatch
                mask_bin = np.ones((h, w), dtype=np.uint8)
        else:
            mask_bin = np.ones((h, w), dtype=np.uint8)

        # ---------- Blur (variance of Laplacian) ----------
        gray_u8 = (image_np * 255.0).astype(np.uint8)
        gray_u8 = cv2.cvtColor(gray_u8, cv2.COLOR_RGB2GRAY)
        lap = cv2.Laplacian(gray_u8, cv2.CV_64F)
        blur_raw = float(np.var(lap[mask_bin == 1]))

        # ---------- SNR (dB) ----------
        mean = cv2.blur(gray_u8.astype(np.float32), (3, 3))
        diff = gray_u8.astype(np.float32) - mean
        signal_power = float(np.mean((gray_u8[mask_bin == 1]) ** 2))
        noise_power = float(np.mean((diff[mask_bin == 1]) ** 2))
        snr_raw = float("inf") if noise_power == 0 else float(10.0 * np.log10(max(1e-9, signal_power) / max(1e-9, noise_power)))

        # ---------- Compression proxy: raw_size / JPEG95_size ----------
        try:
            pil_img = Image.fromarray((image_np * 255.0).astype(np.uint8))
            from io import BytesIO
            buf = BytesIO()
            pil_img.save(buf, format="JPEG", quality=95, optimize=True)
            jpeg_kb = max(1e-9, len(buf.getvalue()) / 1024.0)
            raw_kb = (h * w * 3) / 1024.0  # RGB bytes
            comp_ratio_raw = float(raw_kb / jpeg_kb) if jpeg_kb > 0 else 0.0
        except Exception:
            comp_ratio_raw = 1.0

        # ---------- Scale (optional) ----------
        if scale_output_0_100:
            brisque_out = round(self._scale_brisque(brisque_raw), 2)
            blur_out = round(self._scale_blur(blur_raw), 2)
            snr_out = round(self._scale_snr(snr_raw), 2)
            comp_out = round(self._scale_compression(comp_ratio_raw), 2)
        else:
            brisque_out = round(float(brisque_raw), 2) if np.isfinite(brisque_raw) else float("nan")
            blur_out = round(float(blur_raw), 2)
            snr_out = float("inf") if snr_raw == float("inf") else round(float(snr_raw), 2)
            comp_out = round(float(comp_ratio_raw), 2)

        return (brisque_out, blur_out, snr_out, comp_out, HELP_MESSAGE)


NODE_CLASS_MAPPINGS = {
    "FRED_ImageQualityInspector": FRED_ImageQualityInspector
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FRED_ImageQualityInspector": "👑 FRED Image Quality Inspector (piq)"
}
