import torch
import numpy as np
import cv2

class FRED_Image_Sharpening_Blur_Level:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "minimum_deblur_output": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 100.0, "step": 5.0}),
            },
            "optional": {
                "mask_optional": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "👑FRED/image/postprocessing"

    @staticmethod
    def print_tensor_info(tensor, name="Tensor"):
        print(f"[DEBUG] {name}: type={type(tensor)}, dtype={tensor.dtype}, shape={tensor.shape}, "
              f"min={tensor.min().item():.4f}, max={tensor.max().item():.4f}, device={tensor.device}")

    @staticmethod
    def compute_raw_blur(image_np, mask_np=None):
        h, w, c = image_np.shape
        gray = cv2.cvtColor((image_np * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        if mask_np is not None:
            mask_bin = (mask_np.squeeze() > 0.5).astype(np.uint8)
            if mask_bin.shape != (h, w):
                mask_bin = np.ones((h, w), dtype=np.uint8)
            if np.any(mask_bin):
                blur_var = float(np.var(laplacian[mask_bin == 1]))
            else:
                blur_var = float(np.var(laplacian))
        else:
            blur_var = float(np.var(laplacian))
        return blur_var

    @staticmethod
    def scale_blur_raw_to_100(blur_raw, baseline=150.0):
        blur_raw = max(0.0, float(blur_raw))
        return 100.0 * blur_raw / (blur_raw + baseline + 1e-9)

    @staticmethod
    def sharpen_torch(image_torch, amount):
        epsilon = 1e-5
        img = torch.nn.functional.pad(image_torch.permute([0, 3, 1, 2]), pad=(1, 1, 1, 1))
        a = img[..., :-2, :-2]
        b = img[..., :-2, 1:-1]
        c = img[..., :-2, 2:]
        d = img[..., 1:-1, :-2]
        e = img[..., 1:-1, 1:-1]
        f = img[..., 1:-1, 2:]
        g = img[..., 2:, :-2]
        h = img[..., 2:, 1:-1]
        i = img[..., 2:, 2:]

        cross = (b, d, e, f, h)
        mn = torch.min(torch.stack(cross), dim=0).values
        mx = torch.max(torch.stack(cross), dim=0).values

        diag = (a, c, g, i)
        mn2 = torch.min(torch.stack(diag), dim=0).values
        mx2 = torch.max(torch.stack(diag), dim=0).values
        mx = mx + mx2
        mn = mn + mn2

        inv_mx = torch.reciprocal(mx + epsilon)
        amp = inv_mx * torch.minimum(mn, (2 - mx))
        amp = torch.sqrt(amp)
        w = -amp * (amount * (1 / 5 - 1 / 8) + 1 / 8)
        div = torch.reciprocal(1 + 4 * w)
        output = ((b + d + f + h) * w + e) * div
        output = output.clamp(0, 1)
        output = output.permute([0, 2, 3, 1])
        return output.contiguous()

    def fix_output(self, output, reference):
        # Ensures output is float32, correct device, batch shape and contiguous
        if output.dtype != torch.float32:
            output = output.to(dtype=torch.float32)
        if output.device != reference.device:
            output = output.to(reference.device)
        if output.shape != reference.shape:
            output = output.reshape(reference.shape)
        output = output.clamp(0, 1).contiguous()
        return output

    def execute(self, image, mask_optional=None, minimum_deblur_output=50.0):
        device = image.device
        dtype = image.dtype
        image_np = image[0].cpu().numpy()
        h, w, c = image_np.shape

        mask_np = None
        if mask_optional is not None:
            mask_np = mask_optional[0].cpu().numpy()
            if mask_np.shape != (h, w):
                mask_np = None

        raw_blur = self.compute_raw_blur(image_np, mask_np)
        scaled_blur = self.scale_blur_raw_to_100(raw_blur)

        print(f"[FRED_Image_Sharpening_Blur_Level] Initial raw blur: {raw_blur:.4f}")
        print(f"[FRED_Image_Sharpening_Blur_Level] Initial scaled blur: {scaled_blur:.2f} (threshold: {minimum_deblur_output:.2f})")

        if scaled_blur >= minimum_deblur_output:
            print("[FRED_Image_Sharpening_Blur_Level] Image meets deblur threshold, passing image unchanged.")
            self.print_tensor_info(image, "Original Input Image")
            output = self.fix_output(image, image)
            return (output,)

        image_torch = image.clone()

        for amount in np.arange(0.1, 1.01, 0.1):
            sharpened_img = self.sharpen_torch(image_torch, amount)

            # Convert to numpy for blur calculation only
            sharpened_np = sharpened_img[0].cpu().numpy()
            raw_blur_sharp = self.compute_raw_blur(sharpened_np, mask_np)
            scaled_blur_sharp = self.scale_blur_raw_to_100(raw_blur_sharp)

            print(f"[FRED_Image_Sharpening_Blur_Level] Sharpen {amount:.2f} - raw blur: {raw_blur_sharp:.4f} scaled blur: {scaled_blur_sharp:.2f}")

            if scaled_blur_sharp >= minimum_deblur_output:
                print(f"[FRED_Image_Sharpening_Blur_Level] Adaptive sharpen amount {amount:.2f} applied to reach threshold: scaled blur {scaled_blur_sharp:.2f}")
                self.print_tensor_info(sharpened_img, "Sharpened Output Image")
                sharpened_img = self.sharpen_torch(image_torch, amount)
                sharpened_img = sharpened_img.to(device=device, dtype=dtype).clamp(0, 1).contiguous()
                output = self.fix_output(sharpened_img, image)
                return (output,)

        print("[FRED_Image_Sharpening_Blur_Level] Could not reach minimum_deblur_output; applying fallback sharpen amount=1.00.")
        fallback_img = self.sharpen_torch(image_torch, 1.0)
        self.print_tensor_info(fallback_img, "Fallback Sharpened (amount=1.0)")
        fallback_img = fallback_img.to(device=device, dtype=dtype).clamp(0, 1).contiguous()
        output = self.fix_output(fallback_img, image)
        return (output,)

NODE_CLASS_MAPPINGS = {
    "FRED_Image_Sharpening_Blur_Level": FRED_Image_Sharpening_Blur_Level,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FRED_Image_Sharpening_Blur_Level": "👑 FRED Image Sharpening Blur Level",
}