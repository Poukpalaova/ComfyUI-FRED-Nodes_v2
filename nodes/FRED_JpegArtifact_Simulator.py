import torch
import numpy as np
from PIL import Image
import io
from comfy.comfy_types.node_typing import ComfyNodeABC, InputTypeDict, IO

HELP_MESSAGE = """

👑 FRED_JpegArtifact_Simulator

🔹 PURPOSE

- Simulates JPEG compression artifacts by compressing and reloading the image in memory.
- Useful for testing robustness, generating aesthetic lo-fi effects, or dataset augmentation.

📥 INPUTS

- images: The input image batch (IMAGE).
- quality: JPEG quality factor (1-100). Lower values = more artifacts. (Default: 50)

⚙️ KEY BEHAVIOR

- Converts Tensor to PIL Image.
- Saves to an in-memory byte buffer as JPEG with specified quality.
- Reloads from buffer and converts back to Tensor.
- No files are saved to disk.

📤 OUTPUTS

- IMAGE: The degraded image batch with JPEG artifacts.
- help: Help documentation string.

📝 NOTES & TIPS

- Quality < 20 produces strong blocking and ringing.
- Quality > 90 is barely noticeable.

"""

class FRED_JpegArtifact_Simulator(ComfyNodeABC):
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", ),
                "quality": ("INT", {"default": 50, "min": 1, "max": 100, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "help")
    FUNCTION = "simulate_jpeg"
    CATEGORY = "👑FRED/image"
    DESCRIPTION = "Simulates JPEG compression artifacts by compressing and reloading the image in memory."

    def simulate_jpeg(self, images, quality):
        
        output_images = []
        
        # Iterate over batch
        for image in images:
            # 1. Convert ComfyUI Tensor to Numpy (Height, Width, Channels)
            # Tensor is usually 0-1 float, so scale to 0-255
            i = 255. * image.cpu().numpy()
            
            # 2. Convert to PIL Image
            img_pil = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            
            # Handle Alpha channel by converting to RGB (JPEG doesn't support Alpha)
            if img_pil.mode == 'RGBA':
                img_pil = img_pil.convert('RGB')
                
            # 3. Save to memory buffer
            buffer = io.BytesIO()
            img_pil.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            
            # 4. Load back from buffer (Artifacts are now baked in)
            img_degraded = Image.open(buffer)
            
            # 5. Convert back to Tensor (Normalize to 0-1)
            img_degraded_np = np.array(img_degraded).astype(np.float32) / 255.0
            output_images.append(torch.from_numpy(img_degraded_np))

        # Stack list of tensors back into a batch tensor (Batch, Height, Width, Channels)
        result_batch = torch.stack(output_images)
        
        return (result_batch, HELP_MESSAGE)

# Node Mappings
NODE_CLASS_MAPPINGS = {
    "FRED_JpegArtifact_Simulator": FRED_JpegArtifact_Simulator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FRED_JpegArtifact_Simulator": "👑 FRED Jpeg Artifact Simulator",
}
