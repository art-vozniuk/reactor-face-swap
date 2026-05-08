import os
from typing import List

import cv2
import numpy as np
from PIL import Image

import insightface
from insightface.app.common import Face

# try:
#     import torch.cuda as cuda
# except:
#     cuda = None
import torch

import folder_paths

from scripts.reactor_logger import logger
from reactor_utils import move_path
from scripts.r_faceboost import swapper, restorer

import warnings

np.warnings = warnings
np.warnings.filterwarnings("ignore")

# PROVIDERS
try:
    if torch.cuda.is_available():
        providers = ["CUDAExecutionProvider"]
    elif torch.backends.mps.is_available():
        providers = ["CoreMLExecutionProvider"]
    elif hasattr(torch, "dml") or hasattr(torch, "privateuseone"):
        providers = ["ROCMExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
except Exception as e:
    logger.debug(f"ExecutionProviderError: {e}.\nEP is set to CPU.")
    providers = ["CPUExecutionProvider"]
# if cuda is not None:
#     if cuda.is_available():
#         providers = ["CUDAExecutionProvider"]
#     else:
#         providers = ["CPUExecutionProvider"]
# else:
#     providers = ["CPUExecutionProvider"]

models_path_old = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
insightface_path_old = os.path.join(models_path_old, "insightface")
insightface_models_path_old = os.path.join(insightface_path_old, "models")

models_path = folder_paths.models_dir
insightface_path = os.path.join(models_path, "insightface")
insightface_models_path = os.path.join(insightface_path, "models")
reswapper_path = os.path.join(models_path, "reswapper")

if os.path.exists(models_path_old):
    move_path(insightface_models_path_old, insightface_models_path)
    move_path(insightface_path_old, insightface_path)
    move_path(models_path_old, models_path)
# if os.path.exists(insightface_path) and os.path.exists(insightface_path_old):
#     shutil.rmtree(insightface_path_old)
#     shutil.rmtree(models_path_old)


FS_MODEL = None
CURRENT_FS_MODEL_PATH = None

ANALYSIS_MODELS = {
    "640": None,
    "320": None,
}


def unload_model(model):
    if model is not None:
        # check if model has unload method
        # if "unload" in model:
        #     model.unload()
        # if "model_unload" in model:
        #     model.model_unload()
        del model
    return None


def unload_all_models():
    global FS_MODEL, CURRENT_FS_MODEL_PATH
    FS_MODEL = unload_model(FS_MODEL)
    ANALYSIS_MODELS["320"] = unload_model(ANALYSIS_MODELS["320"])
    ANALYSIS_MODELS["640"] = unload_model(ANALYSIS_MODELS["640"])


def getAnalysisModel(det_size=(640, 640)):
    global ANALYSIS_MODELS
    ANALYSIS_MODEL = ANALYSIS_MODELS[str(det_size[0])]
    if ANALYSIS_MODEL is None:
        ANALYSIS_MODEL = insightface.app.FaceAnalysis(
            name="buffalo_l", providers=providers, root=insightface_path
        )
    ANALYSIS_MODEL.prepare(ctx_id=0, det_size=det_size)
    ANALYSIS_MODELS[str(det_size[0])] = ANALYSIS_MODEL
    return ANALYSIS_MODEL


def _resolve_inswapper_model_path(model: str) -> str:
    # Accept absolute path
    if os.path.isabs(model) and os.path.exists(model):
        return model

    # Check our configured models folder first (Comfy-style)
    candidate = os.path.join(insightface_path, model)
    if os.path.exists(candidate):
        return candidate

    # Try user cache used by InsightFace
    user_cache_dir = os.path.expanduser("~/.insightface/models")
    user_cache_path = os.path.join(user_cache_dir, model)
    if os.path.exists(user_cache_path):
        return user_cache_path

    # Ask InsightFace to ensure availability (downloads if needed)
    try:
        from insightface.utils import ensure_available

        # strip extension for ensure_available name
        name = os.path.splitext(model)[0]
        ensured = ensure_available(
            "models", name, root=os.path.expanduser("~/.insightface")
        )
        # ensured may be a directory; if so, append the filename
        if os.path.isdir(ensured):
            ensured_path = os.path.join(ensured, model)
        else:
            ensured_path = ensured
        if os.path.exists(ensured_path):
            return ensured_path
    except Exception:
        pass

    # Fallback to original string; downstream will error if missing
    return model


def getFaceSwapModel(model_path: str):
    global FS_MODEL, CURRENT_FS_MODEL_PATH
    resolved_path = _resolve_inswapper_model_path(model_path)
    if (
        FS_MODEL is None
        or CURRENT_FS_MODEL_PATH is None
        or CURRENT_FS_MODEL_PATH != resolved_path
    ):
        CURRENT_FS_MODEL_PATH = resolved_path
        FS_MODEL = unload_model(FS_MODEL)
        FS_MODEL = insightface.model_zoo.get_model(resolved_path, providers=providers)

    return FS_MODEL


def half_det_size(det_size):
    logger.status("Trying to halve 'det_size' parameter")
    return (det_size[0] // 2, det_size[1] // 2)


def analyze_faces(img_data: np.ndarray, det_size=(640, 640)):
    face_analyser = getAnalysisModel(det_size)

    faces = []
    try:
        faces = face_analyser.get(img_data)
    except:
        logger.error("No faces found")

    # Try halving det_size if no faces are found
    if len(faces) == 0 and det_size[0] > 320 and det_size[1] > 320:
        det_size_half = half_det_size(det_size)
        return analyze_faces(img_data, det_size_half)

    return faces


def detect_faces(img: Image.Image) -> List[Face]:
    """Detect faces in a single image.

    Returns insightface ``Face`` objects (each carries bbox, embedding,
    keypoints, etc.) in detector order. The caller picks which one to
    operate on — bbox is the user-visible identifier.
    """
    import time

    t0 = time.perf_counter()
    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    faces = list(analyze_faces(img_bgr) or [])
    logger.info(
        f"detect_faces: found {len(faces)} face(s), took {(time.perf_counter() - t0) * 1000:.1f}ms"
    )
    return faces


def swap_specific_face(
    target_img: Image.Image,
    source_face: Face,
    target_face: Face,
    model: str = "inswapper_128.onnx",
    face_restore_model=None,
    face_restore_visibility: int = 1,
    codeformer_weight: float = 0.5,
    interpolation: str = "Bicubic",
) -> Image.Image:
    """Swap exactly ``source_face`` onto exactly ``target_face`` in ``target_img``.

    Caller resolves which faces to operate on (e.g. via :func:`detect_faces`
    + IoU-matching against a UI-selected bbox) and passes Face objects in
    directly. There is no internal re-detection, sorting, gender filtering
    or index-based selection — those concerns belong to the caller.

    ``source_face`` carries the embedding used to drive the swap; the source
    image itself isn't needed at this stage.
    """
    import time

    t0 = time.perf_counter()
    target_img_bgr = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)

    if "inswapper" in model:
        model_path = _resolve_inswapper_model_path(model)
    elif "reswapper" in model:
        candidate = os.path.join(reswapper_path, model)
        model_path = candidate if os.path.exists(candidate) else model
    else:
        model_path = model
    face_swapper = getFaceSwapModel(model_path)

    if face_restore_model:
        bgr_fake, M = face_swapper.get(
            target_img_bgr, target_face, source_face, paste_back=False
        )
        bgr_fake, scale = restorer.get_restored_face(
            bgr_fake,
            face_restore_model,
            face_restore_visibility,
            codeformer_weight,
            interpolation,
        )
        M *= scale
        result = swapper.in_swap(target_img_bgr, bgr_fake, M)
    else:
        result = face_swapper.get(target_img_bgr, target_face, source_face)

    result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    logger.info(
        f"swap_specific_face: TOTAL took {(time.perf_counter() - t0) * 1000:.1f}ms"
    )
    return result_image
