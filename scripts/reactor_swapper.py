import os
from typing import List, Union

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


def sort_by_order(face, order: str):
    if order == "left-right":
        return sorted(face, key=lambda x: x.bbox[0])
    if order == "right-left":
        return sorted(face, key=lambda x: x.bbox[0], reverse=True)
    if order == "top-bottom":
        return sorted(face, key=lambda x: x.bbox[1])
    if order == "bottom-top":
        return sorted(face, key=lambda x: x.bbox[1], reverse=True)
    if order == "small-large":
        return sorted(
            face, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1])
        )
    # by default "large-small":
    return sorted(
        face,
        key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]),
        reverse=True,
    )


def get_face_gender(
    face,
    face_index,
    gender_condition,
    operated: str,
    order: str,
):
    filtered_faces = [
        f
        for f in face
        if (gender_condition == 0)
        or (gender_condition == 1 and f.sex == "F")
        or (gender_condition == 2 and f.sex == "M")
    ]

    gender = (
        "Female" if gender_condition == 1 else "Male" if gender_condition == 0 else ""
    )

    if len(filtered_faces) == 0:
        if gender_condition != 0:
            logger.status(f"No faces found for -{gender}-")
        return None, 0  # treat as "wrong gender" to skip

    faces_sorted = sort_by_order(filtered_faces, order)

    if face_index >= len(faces_sorted):
        logger.info(
            "Requested face index (%s) is out of bounds (max available index is %s)",
            face_index,
            len(faces_sorted),
        )
        return None, 0, None

    face_selected = faces_sorted[face_index]

    logger.info(
        "%s Face %s: Detected Gender -%s-", operated, face_index, face_selected.sex
    )

    expected_gender = "F" if gender_condition == 1 else "M"
    if gender_condition != 0 and face_selected.sex != expected_gender:
        logger.info(f"{operated} Face {face_index}: WRONG gender ({face_selected.sex})")
        return face_selected, 1, face_index  # <-- есть, но не тот пол

    return face_selected, 0, face_index


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


def get_face_single(
    img_data: np.ndarray,
    face,
    face_index=0,
    det_size=(640, 640),
    gender_source=0,
    gender_target=0,
    order="large-small",
):
    buffalo_path = os.path.join(insightface_models_path, "buffalo_l.zip")
    # if os.path.exists(buffalo_path):
    #   os.remove(buffalo_path)

    if gender_source != 0:
        if len(face) == 0 and det_size[0] > 320 and det_size[1] > 320:
            det_size_half = half_det_size(det_size)
            return get_face_single(
                img_data,
                analyze_faces(img_data, det_size_half),
                face_index,
                det_size_half,
                gender_source,
                gender_target,
                order,
            )
        return get_face_gender(face, face_index, gender_source, "Source", order)

    if gender_target != 0:
        if len(face) == 0 and det_size[0] > 320 and det_size[1] > 320:
            det_size_half = half_det_size(det_size)
            return get_face_single(
                img_data,
                analyze_faces(img_data, det_size_half),
                face_index,
                det_size_half,
                gender_source,
                gender_target,
                order,
            )
        return get_face_gender(face, face_index, gender_target, "Target", order)

    if len(face) == 0 and det_size[0] > 320 and det_size[1] > 320:
        det_size_half = half_det_size(det_size)
        return get_face_single(
            img_data,
            analyze_faces(img_data, det_size_half),
            face_index,
            det_size_half,
            gender_source,
            gender_target,
            order,
        )

    try:
        faces_sorted = sort_by_order(face, order)
        return faces_sorted[face_index], 0, face_index
        # return sorted(face, key=lambda x: x.bbox[0])[face_index], 0
    except IndexError:
        return None, 0, None


def swap_face(
    source_img: Image.Image,
    target_img: Image.Image,
    model: str,
    source_faces_index: List[int] = [0],
    faces_index: List[int] = [0],
    gender_source: int = 0,
    gender_target: int = 0,
    faces_order: List = ["large-small", "large-small"],
    face_boost_enabled: bool = False,
    face_restore_model=None,
    face_restore_visibility: int = 1,
    codeformer_weight: float = 0.5,
    interpolation: str = "Bicubic",
):
    result_image = target_img
    bbox = []
    swapped_indexes = []

    target_img = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)
    source_img = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)

    logger.status("Analyzing Source Image...")
    source_faces = analyze_faces(source_img)

    if not source_faces:
        logger.error("No source face(s) found")
        return result_image, bbox, swapped_indexes

    logger.status("Analyzing Target Image...")
    target_faces = analyze_faces(target_img)

    if len(target_faces) == 0:
        logger.status("No target faces found, skipping...")
        return result_image, bbox, swapped_indexes

    source_face, src_wrong_gender, source_face_index = get_face_single(
        source_img,
        source_faces,
        face_index=source_faces_index[0],
        gender_source=gender_source,
        order=faces_order[1],
    )

    if source_face is None:
        logger.status("No source face in the provided Index")
        return result_image, bbox, swapped_indexes

    if src_wrong_gender != 0:
        logger.status("Wrong source gender detected")
        return result_image, bbox, swapped_indexes

    if "inswapper" in model:
        model_path = _resolve_inswapper_model_path(model)
    elif "reswapper" in model:
        candidate = os.path.join(reswapper_path, model)
        model_path = candidate if os.path.exists(candidate) else model
    else:
        model_path = model

    face_swapper = getFaceSwapModel(model_path)
    result = target_img

    for face_num in faces_index:
        if face_num >= len(target_faces):
            logger.status("Face index out of bounds, skipping...")
            break

        target_face, wrong_gender, target_face_index = get_face_single(
            target_img,
            target_faces,
            face_index=face_num,
            gender_target=gender_target,
            order=faces_order[0],
        )

        if target_face is None or wrong_gender != 0:
            if wrong_gender == 1:
                logger.status("Wrong target gender detected")
            else:
                logger.info(f"No target face found for {face_num}")
            continue

        logger.status("Swapping...")
        if face_boost_enabled:
            logger.status("Face Boost is enabled")
            bgr_fake, M = face_swapper.get(
                result, target_face, source_face, paste_back=False
            )
            bgr_fake, scale = restorer.get_restored_face(
                bgr_fake,
                face_restore_model,
                face_restore_visibility,
                codeformer_weight,
                interpolation,
            )
            M *= scale
            result = swapper.in_swap(target_img, bgr_fake, M)
        else:
            result = face_swapper.get(result, target_face, source_face)

        bbox = [tuple(map(float, target_face.bbox))]
        swapped_indexes = [target_face_index]

    result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    return result_image, bbox, swapped_indexes
