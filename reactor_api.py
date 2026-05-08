import os
from typing import List
from PIL import Image
from insightface.app.common import Face

from scripts.reactor_swapper import (
    detect_faces as _detect_faces,
    swap_specific_face as _swap_specific_face,
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


def detect_faces(img: Image.Image) -> List[Face]:
    """Detect faces in a single image.

    Returns insightface ``Face`` objects in detector order; each one carries
    its bbox plus the embedding/landmarks needed by :func:`swap_specific_face`.
    """
    return _detect_faces(img)


def swap_specific_face(
    target_img: Image.Image,
    source_face: Face,
    target_face: Face,
    model: str = "inswapper_128.onnx",
    face_boost_model: str | None = None,
    visibility: int = 1,
    codeformer_weight: float = 0.5,
    interpolation: str = "Bicubic",
) -> Image.Image:
    """Replace ``target_face`` in ``target_img`` with ``source_face``.

    Both ``Face`` objects are expected to come from :func:`detect_faces` runs
    on the appropriate images. Caller picks which face to operate on — this
    function does no detection, no sorting, no gender filtering.

    ``face_boost_model``, when provided, is resolved to a local restore-model
    path (downloaded on first use); pass ``None`` to skip restoration.
    """
    face_restore_model = None
    if face_boost_model:
        face_restore_model = _ensure_facerestore_model(face_boost_model)

    return _swap_specific_face(
        target_img,
        source_face,
        target_face,
        model=model,
        face_restore_model=face_restore_model,
        face_restore_visibility=visibility,
        codeformer_weight=codeformer_weight,
        interpolation=interpolation,
    )
