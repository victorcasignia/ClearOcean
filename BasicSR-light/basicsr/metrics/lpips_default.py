import numpy as np
import torch

from basicsr.utils.registry import METRIC_REGISTRY

try:
    import lpips
except ImportError:
    lpips = None


def _to_lpips_tensor(img):
    """Convert HWC uint8 RGB image to LPIPS input tensor in [-1, 1]."""
    assert len(img.shape) == 3
    assert img.dtype == np.uint8
    img = np.transpose(img, [2, 0, 1])
    img = np.expand_dims(img, axis=0)
    return torch.tensor(img, dtype=torch.float32) / 127.5 - 1


@METRIC_REGISTRY.register()
def calculate_lpips_default(img, img2, device='cuda', model=None):
    """Calculate LPIPS distance with AlexNet backbone."""
    if model is None:
        if lpips is None:
            raise ImportError('lpips is required for calculate_lpips_default. Please install lpips.')
        model = lpips.LPIPS(net='alex').to(device)
    model.eval()

    tA = _to_lpips_tensor(img).to(device)
    tB = _to_lpips_tensor(img2).to(device)
    with torch.no_grad():
        dist = model(tA, tB).item()
    return dist