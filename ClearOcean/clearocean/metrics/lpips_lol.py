import numpy as np
import torch

from basicsr.utils.registry import METRIC_REGISTRY


def _to_lpips_tensor(img: np.ndarray) -> torch.Tensor:
    assert len(img.shape) == 3
    assert img.dtype == np.uint8
    chw = np.transpose(img, [2, 0, 1])
    bchw = np.expand_dims(chw, axis=0)
    return torch.tensor(bchw, dtype=torch.float32) / 127.5 - 1.0


@METRIC_REGISTRY.register()
def calculate_lpips_lol(img, img2, device, model):
    tensor_a = _to_lpips_tensor(img).to(device)
    tensor_b = _to_lpips_tensor(img2).to(device)
    return model.forward(tensor_a, tensor_b).item()
