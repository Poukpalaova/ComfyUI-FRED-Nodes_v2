"""
FRED_RestoreFaceWithMask - Custom ComfyUI Node
Based on ReActor's restore_face with custom mask support
Version: 4.1 (Fixed model_loading import)
"""

import os
import glob
import sys
import torch
import numpy as np
import cv2
from torchvision.transforms.functional import normalize

# ComfyUI imports
import comfy.model_management as model_management
import comfy.utils
import folder_paths

# Find ReActor installation
REACTOR_PATH = None
for node_dir in os.listdir(os.path.join(folder_paths.base_path, "custom_nodes")):
    if "reactor" in node_dir.lower():
        potential_path = os.path.join(folder_paths.base_path, "custom_nodes", node_dir)
        if os.path.isdir(potential_path) and os.path.exists(os.path.join(potential_path, "nodes.py")):
            REACTOR_PATH = potential_path
            break

if REACTOR_PATH is None:
    print("👑FRED: ❌ ReActor not found! Install from: https://github.com/Gourieff/comfyui-reactor-node")
    raise ImportError("ReActor is required for this node")

print(f"👑FRED: ✓ ReActor found at: {REACTOR_PATH}")

# Add ReActor to path
if REACTOR_PATH not in sys.path:
    sys.path.insert(0, REACTOR_PATH)

# Import ALL ReActor dependencies
try:
    from reactor_utils import img2tensor, tensor2img, set_ort_session, prepare_cropped_face, normalize_cropped_face
    print("👑FRED: ✓ reactor_utils imported")
except ImportError as e:
    print(f"👑FRED: ❌ Failed to import reactor_utils: {e}")
    raise

try:
    from r_facexlib.utils.face_restoration_helper import FaceRestoreHelper
    print("👑FRED: ✓ r_facexlib.FaceRestoreHelper imported")
except ImportError:
    try:
        from rfacexlib.utils.face_restoration_helper import FaceRestoreHelper
        print("👑FRED: ✓ rfacexlib.FaceRestoreHelper imported")
    except ImportError as e:
        print(f"👑FRED: ❌ Failed to import FaceRestoreHelper: {e}")
        FaceRestoreHelper = None

try:
    from rbasicsr.utils.registry import ARCH_REGISTRY
    print("👑FRED: ✓ rbasicsr.ARCH_REGISTRY imported")
except ImportError as e:
    print(f"👑FRED: ❌ Failed to import ARCH_REGISTRY: {e}")
    ARCH_REGISTRY = None

# Import model_loading - CRITICAL FIX!
model_loading = None
try:
    from rchainner import model_loading
    print("👑FRED: ✓ rchainner.model_loading imported")
except ImportError:
    print("👑FRED: ⚠️ rchainner not found, trying comfy_extras...")
    try:
        from comfy_extras.chainner_models import model_loading
        print("👑FRED: ✓ comfy_extras.chainner_models.model_loading imported")
    except ImportError:
        print("👑FRED: ⚠️ comfy_extras.chainner_models not found, trying direct import...")
        try:
            # Try importing from ReActor's rchainner directory
            rchainner_path = os.path.join(REACTOR_PATH, "rchainner")
            if os.path.exists(rchainner_path) and rchainner_path not in sys.path:
                sys.path.insert(0, rchainner_path)
            from model_loading import model_loading as ml
            model_loading = ml
            print("👑FRED: ✓ ReActor's rchainner.model_loading imported")
        except ImportError as e:
            print(f"👑FRED: ❌ All model_loading imports failed: {e}")
            model_loading = None

try:
    from scripts.reactor_logger import logger
    print("👑FRED: ✓ reactor_logger imported")
except ImportError:
    # Fallback logger
    class SimpleLogger:
        def status(self, msg): print(f"[ReActor] {msg}")
        def error(self, msg): print(f"[ReActor] ERROR - {msg}")
    logger = SimpleLogger()
    print("👑FRED: ⚠️ Using fallback logger")

# Import CodeFormer architecture (CRITICAL!)
CODEFORMER_AVAILABLE = False
if ARCH_REGISTRY is not None:
    try:
        import scripts.rarchs.codeformer_arch
        CODEFORMER_AVAILABLE = True
        print("👑FRED: ✓ CodeFormer architecture registered")
    except ImportError as e:
        print(f"👑FRED: ⚠️ CodeFormer not available: {e}")


def setup_facerestore_models_dir():
    models_dir = folder_paths.models_dir
    dir_facerestore_models = os.path.join(models_dir, "facerestore_models")
    os.makedirs(dir_facerestore_models, exist_ok=True)

    if "facerestore_models" not in folder_paths.folder_names_and_paths:
        folder_paths.folder_names_and_paths["facerestore_models"] = (
            [dir_facerestore_models],
            folder_paths.supported_pt_extensions
        )
    return dir_facerestore_models


def get_restorers():
    setup_facerestore_models_dir()
    models_dir = folder_paths.models_dir
    models_path = os.path.join(models_dir, "facerestore_models", "*")
    models = glob.glob(models_path)
    models = [x for x in models if x.endswith(".pth") or x.endswith(".onnx")]
    return models


def get_model_names():
    models = get_restorers()
    names = []
    for x in models:
        names.append(os.path.basename(x))
    names.sort(key=str.lower)
    names.insert(0, "none")
    return names


# Global cache (like ReActor)
FACE_SIZE = int(512)
FACE_HELPER = None


class FRED_RestoreFaceWithMask:
    """Face restoration based on ReActor with custom mask support"""

    def __init__(self):
        self.face_helper = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "face_detection": ([
                    "retinaface_resnet50",
                    "retinaface_mobile0.25", 
                    "YOLOv5l",
                    "YOLOv5n"
                ],),
                "model": (get_model_names(),),
                "visibility": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05
                }),
                "codeformer_weight": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05
                }),
                "use_custom_mask": ("BOOLEAN", {
                    "default": False,
                    "label_on": "enabled",
                    "label_off": "disabled"
                }),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "👑FRED/image/postprocessing"

    def execute(self, image, model, visibility, codeformer_weight, face_detection, use_custom_mask=False, mask=None):
        """Execute face restoration - EXACT ReActor implementation with mask"""
        result = image

        if model != "none" and not model_management.processing_interrupted():
            global FACE_SIZE, FACE_HELPER

            self.face_helper = FACE_HELPER

            # Determine face size (EXACT ReActor)
            face_size = 512
            if "1024" in model.lower():
                face_size = 1024
            elif "2048" in model.lower():
                face_size = 2048

            logger.status(f"Restoring with {model} | Face Size: {face_size} | Custom mask: {use_custom_mask}")

            # Get model path
            model_path = folder_paths.get_full_path("facerestore_models", model)
            if model_path is None:
                logger.error(f"Model not found: {model}")
                return (result,)

            device = model_management.get_torch_device()

            # Load model (EXACT ReActor method)
            try:
                if "codeformer" in model.lower():
                    if not CODEFORMER_AVAILABLE or ARCH_REGISTRY is None:
                        logger.error("CodeFormer not available - use GFPGAN instead")
                        return (result,)

                    codeformer_net = ARCH_REGISTRY.get("CodeFormer")(
                        dim_embd=512,
                        codebook_size=1024,
                        n_head=8,
                        n_layers=9,
                        connect_list=["32", "64", "128", "256"],
                    ).to(device)

                    checkpoint = torch.load(model_path)["params_ema"]
                    codeformer_net.load_state_dict(checkpoint)
                    face_restore_model = codeformer_net.eval()

                elif ".onnx" in model:
                    # ONNX models
                    try:
                        from reactor_utils import providers
                    except ImportError:
                        providers = ['CPUExecutionProvider']
                        if torch.cuda.is_available():
                            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

                    ort_session = set_ort_session(model_path, providers=providers)
                    if ort_session is None:
                        logger.error("Failed to create ONNX session")
                        return (result,)
                    ort_session_inputs = {}
                    face_restore_model = ort_session

                else:
                    # GFPGAN (PTH) - CRITICAL FIX!
                    if model_loading is None:
                        logger.error("model_loading not available - cannot load GFPGAN")
                        logger.error("Please ensure comfy_extras.chainner_models is available")
                        return (result,)

                    logger.status("Loading GFPGAN model...")
                    sd = comfy.utils.load_torch_file(model_path, safe_load=True)

                    # Use model_loading.load_state_dict (ReActor way)
                    face_restore_model = model_loading.load_state_dict(sd).eval()
                    face_restore_model.to(device)
                    logger.status("GFPGAN loaded successfully")

            except Exception as e:
                logger.error(f"Error loading model: {e}")
                import traceback
                traceback.print_exc()
                return (result,)

            # Initialize FaceRestoreHelper (EXACT ReActor)
            if face_size != FACE_SIZE or self.face_helper is None:
                if FaceRestoreHelper is None:
                    logger.error("FaceRestoreHelper not available")
                    return (result,)

                try:
                    self.face_helper = FaceRestoreHelper(
                        1,
                        face_size=face_size,
                        crop_ratio=(1, 1),
                        det_model=face_detection,
                        save_ext="png",
                        use_parse=True,
                        device=device
                    )
                    FACE_SIZE = face_size
                    FACE_HELPER = self.face_helper
                except Exception as e:
                    logger.error(f"Error initializing FaceRestoreHelper: {e}")
                    return (result,)

            # Convert image (EXACT ReActor)
            image_np = 255.0 * result.cpu().numpy()
            total_images = image_np.shape[0]
            out_images = []

            # Process each image (EXACT ReActor)
            for i in range(total_images):
                if total_images > 1:
                    logger.status(f"Restoring {i+1}")

                cur_image_np = image_np[i, :, :, ::-1]
                original_resolution = cur_image_np.shape[0:2]

                if face_restore_model is None or self.face_helper is None:
                    return (result,)

                # Face detection (EXACT ReActor)
                self.face_helper.clean_all()
                self.face_helper.read_image(cur_image_np)
                self.face_helper.get_face_landmarks_5(
                    only_center_face=False,
                    resize=640,
                    eye_dist_threshold=5
                )
                self.face_helper.align_warp_face()

                restored_face = None

                # Restore faces (EXACT ReActor)
                for idx, cropped_face in enumerate(self.face_helper.cropped_faces):
                    cropped_face_t = img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
                    normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                    cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

                    try:
                        with torch.no_grad():
                            if ".onnx" in model:
                                # ONNX inference
                                for ort_session_input in ort_session.get_inputs():
                                    if ort_session_input.name == "input":
                                        cropped_face_prep = prepare_cropped_face(cropped_face)
                                        ort_session_inputs[ort_session_input.name] = cropped_face_prep
                                    if ort_session_input.name == "weight":
                                        weight = np.array([1], dtype=np.double)
                                        ort_session_inputs[ort_session_input.name] = weight

                                output = ort_session.run(None, ort_session_inputs)[0][0]
                                restored_face = normalize_cropped_face(output)
                            else:
                                # PTH models
                                if "codeformer" in model.lower():
                                    output = face_restore_model(cropped_face_t, w=codeformer_weight)[0]
                                else:
                                    output = face_restore_model(cropped_face_t)[0]

                                restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))

                            del output
                            torch.cuda.empty_cache()

                    except Exception as error:
                        print(f"inference error: {error}", file=sys.stderr)
                        restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))

                    # Apply visibility (EXACT ReActor)
                    if visibility < 1.0:
                        restored_face = cropped_face * (1 - visibility) + restored_face * visibility

                    restored_face = restored_face.astype("uint8")
                    self.face_helper.add_restored_face(restored_face)

                # Get inverse affine (EXACT ReActor)
                self.face_helper.get_inverse_affine(None)

                # Paste faces - with custom mask support (NEW!)
                if use_custom_mask and mask is not None:
                    logger.status("Applying custom mask")
                    # Get restored image first
                    restored_img = self.face_helper.paste_faces_to_input_image()
                    restored_img = restored_img[:, :, ::-1]  # BGR to RGB

                    # Process mask
                    if isinstance(mask, torch.Tensor):
                        mask_np = mask[i].cpu().numpy() if mask.shape[0] > i else mask[0].cpu().numpy()
                    else:
                        mask_np = mask

                    # Resize mask if needed
                    if mask_np.shape[:2] != original_resolution:
                        mask_np = cv2.resize(
                            mask_np,
                            (original_resolution[1], original_resolution[0]),
                            interpolation=cv2.INTER_LINEAR
                        )

                    if len(mask_np.shape) == 2:
                        mask_np = mask_np[:, :, np.newaxis]

                    # Blend with mask
                    original_img = cur_image_np[:, :, ::-1]
                    restored_img = (original_img * (1 - mask_np) + restored_img * mask_np).astype(np.uint8)
                else:
                    # Default ReActor paste (no mask)
                    restored_img = self.face_helper.paste_faces_to_input_image()
                    restored_img = restored_img[:, :, ::-1]

                # Resize if needed (EXACT ReActor)
                if original_resolution != restored_img.shape[0:2]:
                    restored_img = cv2.resize(
                        restored_img,
                        (0, 0),
                        fx=original_resolution[1]/restored_img.shape[1],
                        fy=original_resolution[0]/restored_img.shape[0],
                        interpolation=cv2.INTER_AREA
                    )

                self.face_helper.clean_all()
                out_images.append(restored_img)

                if model_management.processing_interrupted():
                    logger.status("Interrupted by User")
                    return (image,)

            # Convert back to tensor (EXACT ReActor)
            restored_img_np = np.array(out_images).astype(np.float32) / 255.0
            restored_img_tensor = torch.from_numpy(restored_img_np)
            result = restored_img_tensor

        return (result,)


NODE_CLASS_MAPPINGS = {
    "FRED_RestoreFaceWithMask": FRED_RestoreFaceWithMask
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FRED_RestoreFaceWithMask": "👑 FRED Restore Face (Custom Mask)"
}
