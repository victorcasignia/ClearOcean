import cv2
from skimage.metrics import structural_similarity

from basicsr.utils.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register()
def calculate_ssim_lol(img, img2, gray_scale=True):
    if gray_scale:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        return structural_similarity(img, img2, data_range=255, channel_axis=None)
    return structural_similarity(img, img2, data_range=255, channel_axis=-1)
