import torch


class ProgressBar:
    def __init__(self, total: int):
        self.total = int(total)
        self.current = 0

    def update(self, n: int = 1) -> None:
        self.current = min(self.current + int(n), self.total)


def load_torch_file(path: str, safe_load: bool = True):
    # Simple wrapper used by ReActor; mirrors ComfyUI behavior sufficiently for our use
    return torch.load(path, map_location="cpu")


