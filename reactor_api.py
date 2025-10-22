import os
from typing import List, Tuple
from PIL import Image

from scripts.reactor_swapper import swap_face
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


def swap_face_api(
    source: Image.Image,
    target: Image.Image,
    model: str = "inswapper_128.onnx",
    source_face_index: int = 0,
    target_face_index: int = 0,
    face_boost_model: str | None = None,
    visibility: float = 1.0,
    codeformer_weight: float = 0.5,
    interpolation: str = "Bicubic",
) -> Tuple[Image.Image, List[tuple]]:
    face_restore_model = None
    if face_boost_model:
        face_restore_model = _ensure_facerestore_model(face_boost_model)

    result, bbox, _ = swap_face(
        source_img=source,
        target_img=target,
        model=model,
        source_faces_index=[source_face_index],
        faces_index=[target_face_index],
        gender_source=0,
        gender_target=0,
        face_model=None,
        faces_order=["large-small", "large-small"],
        face_boost_enabled=bool(face_restore_model),
        face_restore_model=face_restore_model or "none",
        face_restore_visibility=visibility,
        codeformer_weight=codeformer_weight,
        interpolation=interpolation,
    )
    return result, bbox
