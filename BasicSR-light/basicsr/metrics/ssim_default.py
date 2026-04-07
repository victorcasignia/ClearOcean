import numpy as np
from skimage.metrics import structural_similarity

from basicsr.metrics.metric_util import reorder_image, to_y_channel
from basicsr.utils.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register()
def calculate_ssim_default(img, img2, input_order='HWC', test_y_channel=False, **kwargs):
    """Calculate SSIM with skimage.metrics implementation."""
    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are ' '"HWC" and "CHW"')

    img = reorder_image(img, input_order=input_order).astype(np.float64)
    img2 = reorder_image(img2, input_order=input_order).astype(np.float64)

    if test_y_channel:
        img = to_y_channel(img)
        img2 = to_y_channel(img2)

    if img.ndim == 3 and img.shape[2] == 1:
        img = img[..., 0]
        img2 = img2[..., 0]

    try:
        return structural_similarity(img, img2, data_range=255, channel_axis=-1 if img.ndim == 3 else None)
    except TypeError:
        # Backward compatibility with older skimage that uses multichannel.
        return structural_similarity(img, img2, data_range=255, multichannel=(img.ndim == 3))
