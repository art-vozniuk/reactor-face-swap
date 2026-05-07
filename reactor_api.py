import os
from typing import List, Tuple
from PIL import Image

from scripts.reactor_swapper import (
    FaceRecognitionResult,
    recognize_faces,
    swap_face_from_recognition,
)
import folder_paths


_FACERESTORE_URLS = {
    "GFPGANv1.3.pth": "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/GFPGANv1.3.pth",
    "GFPGANv1.4.pth": "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/GFPGANv1.4.pth",
    "codeformer-v0.1.0.pth": "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/codeformer-v0.1.0.pth",
    "GPEN-BFR-512.onnx": "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/GPEN-BFR-512.onnx",
}


def _ensure_facerestore_model(name_or_path: str) -> str:
    # If absolute path exists, use it
    if os.path.isabs(name_or_path) and os.path.exists(name_or_path):
        return name_or_path

    # Resolve via folder_paths
    resolved = folder_paths.get_full_path("facerestore_models", name_or_path)
    if os.path.exists(resolved):
        return resolved

    # Try download if known
    filename = os.path.basename(name_or_path)
    url = _FACERESTORE_URLS.get(filename)
    if url is not None:
        os.makedirs(os.path.dirname(resolved), exist_ok=True)
        import urllib.request

        urllib.request.urlretrieve(url, resolved)
        return resolved

    # Fallback to given string; caller must ensure existence
    return name_or_path


def recognize_faces_api(
    source: Image.Image,
    target: Image.Image,
) -> FaceRecognitionResult:
    """Pipeline 1 entrypoint: detect faces in source and target images.

    Returns a :class:`FaceRecognitionResult` whose ``source_bboxes`` and
    ``target_bboxes`` properties expose the rectangles for every detected
    face. The result is the input expected by :func:`swap_face_api_from_recognition`.
    """
    return recognize_faces(source, target)


def swap_face_api_from_recognition(
    recognition: FaceRecognitionResult,
    model: str = "inswapper_128.onnx",
    source_face_index: int = 0,
    target_face_index: int = 0,
    face_boost_model: str | None = None,
    visibility: int = 1,
    codeformer_weight: float = 0.5,
    interpolation: str = "Bicubic",
) -> Tuple[Image.Image, List[tuple]]:
    """Pipeline 2 entrypoint: swap faces using a recognition result."""
    face_restore_model = None
    if face_boost_model:
        face_restore_model = _ensure_facerestore_model(face_boost_model)

    result, bbox, _ = swap_face_from_recognition(
        recognition,
        model=model,
        source_faces_index=[source_face_index],
        faces_index=[target_face_index],
        gender_source=0,
        gender_target=0,
        faces_order=["large-small", "large-small"],
        face_boost_enabled=bool(face_restore_model),
        face_restore_model=face_restore_model or "none",
        face_restore_visibility=visibility,
        codeformer_weight=codeformer_weight,
        interpolation=interpolation,
    )
    return result, bbox


def swap_face_api(
    source: Image.Image,
    target: Image.Image,
    model: str = "inswapper_128.onnx",
    source_face_index: int = 0,
    target_face_index: int = 0,
    face_boost_model: str | None = None,
    visibility: int = 1,
    codeformer_weight: float = 0.5,
    interpolation: str = "Bicubic",
) -> Tuple[Image.Image, List[tuple]]:
    """Combined entrypoint: run recognition then swap."""
    recognition = recognize_faces_api(source, target)
    return swap_face_api_from_recognition(
        recognition,
        model=model,
        source_face_index=source_face_index,
        target_face_index=target_face_index,
        face_boost_model=face_boost_model,
        visibility=visibility,
        codeformer_weight=codeformer_weight,
        interpolation=interpolation,
    )
