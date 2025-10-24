import os
from pathlib import Path
from typing import Dict, Tuple, List, Set


models_dir = str((Path(__file__).resolve().parent / "models").absolute())

supported_pt_extensions: Set[str] = {".pth", ".onnx"}

# category -> (paths, extensions)
folder_names_and_paths: Dict[str, Tuple[List[str], Set[str]]] = {
    "facerestore_models": (
        [os.path.join(models_dir, "facerestore_models")],
        supported_pt_extensions,
    ),
    "insightface": ([os.path.join(models_dir, "insightface")], {".onnx", ".pth"}),
    "reswapper": ([os.path.join(models_dir, "reswapper")], {".onnx", ".pth"}),
}


def add_model_folder_path(folder_name: str, full_folder_path: str) -> None:
    entry = folder_names_and_paths.get(folder_name)
    if entry:
        paths, exts = entry
        if full_folder_path not in paths:
            paths.append(full_folder_path)
        folder_names_and_paths[folder_name] = (paths, exts)
    else:
        folder_names_and_paths[folder_name] = ([full_folder_path], set())


def get_full_path(category: str, filename: str) -> str:
    if os.path.isabs(filename) and os.path.exists(filename):
        return filename
    if category in folder_names_and_paths:
        paths, _ = folder_names_and_paths[category]
        for p in paths:
            cand = os.path.join(p, filename)
            if os.path.exists(cand):
                return cand
    # default location
    return os.path.join(models_dir, category, filename)
