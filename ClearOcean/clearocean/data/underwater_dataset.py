import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.utils.data as data
from basicsr.utils import get_root_logger
from basicsr.utils.registry import DATASET_REGISTRY
from torchvision.transforms.functional import normalize

from scripts.utils import generate_position_encoding, hiseq_color_cv2_img


_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _collect_images(folder: Path):
    if not folder.exists():
        return []
    files = [p for p in folder.rglob("*") if p.suffix.lower() in _IMAGE_EXTENSIONS]
    return sorted(files)


@DATASET_REGISTRY.register()
class UW_Dataset_Preloaded2(data.Dataset):
    """Minimal preloaded underwater dataset used by clearocean config.

    It supports `uieb`, `lsui`, and `cvmig` via `dataset_names` and falls back
    to placeholder tensors when data is unavailable.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.dataset_names = opt.get("dataset_names", ["uieb", "lsui"])
        self.phase = opt.get("phase", opt.get("name", "train"))
        self.train_percent = int(opt.get("train_percent", 90))

        self.input_root = Path(opt.get("input_root", ""))
        self.gt_root = Path(opt.get("gt_root", ""))

        self.mean = opt.get("mean", [0.5, 0.5, 0.5])
        self.std = opt.get("std", [0.5, 0.5, 0.5])
        self.should_normalize = bool(opt.get("should_normalize", True))

        default_placeholder = 16 if self.phase == "train" else 4
        self.placeholder_length = int(opt.get("placeholder_length", default_placeholder))

        self.pairs = self._build_pairs()
        self.use_placeholder = len(self.pairs) == 0

        logger = get_root_logger()
        if self.use_placeholder:
            logger.warning(
                "No images were found for %s. Using %d placeholder samples.",
                self.dataset_names,
                self.placeholder_length,
            )
            self.gts = []
            self.inputs = []
        else:
            self.inputs = [self._read_rgb(input_path) for input_path, _ in self.pairs]
            self.gts = [self._read_rgb(gt_path) for _, gt_path in self.pairs]
            logger.info("Loaded %d image pairs for %s.", len(self.pairs), self.phase)

    def _build_pairs(self):
        all_pairs = []
        for dataset_name in self.dataset_names:
            all_pairs.extend(self._build_pairs_for_dataset(dataset_name))

        if not all_pairs:
            return []

        unique_pairs = []
        seen = set()
        for input_path, gt_path in all_pairs:
            key = (str(input_path), str(gt_path))
            if key in seen:
                continue
            seen.add(key)
            unique_pairs.append((input_path, gt_path))

        unique_pairs = sorted(unique_pairs)
        split_idx = int(len(unique_pairs) * self.train_percent / 100)

        if self.phase == "train":
            selected = unique_pairs[:split_idx] if split_idx > 0 else unique_pairs
        else:
            selected = unique_pairs[split_idx:] if split_idx < len(unique_pairs) else unique_pairs
        return selected

    def _build_pairs_for_dataset(self, dataset_name):
        input_dirs = [self.input_root / dataset_name, self.input_root]
        gt_dirs = [self.gt_root / dataset_name, self.gt_root]

        input_files = []
        gt_files = []
        for directory in input_dirs:
            input_files.extend(_collect_images(directory))
        for directory in gt_dirs:
            gt_files.extend(_collect_images(directory))

        if not input_files or not gt_files:
            return []

        gt_by_name = {p.stem: p for p in gt_files}
        pairs = [(inp, gt_by_name[inp.stem]) for inp in input_files if inp.stem in gt_by_name]

        if pairs:
            return sorted(pairs)

        min_len = min(len(input_files), len(gt_files))
        return list(zip(sorted(input_files)[:min_len], sorted(gt_files)[:min_len]))

    def _read_rgb(self, path: Path):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _pad_image(self, image, target_h, target_w):
        if not self.opt.get("should_pad", False):
            return image

        if target_h is None or target_w is None:
            return image

        h, w = image.shape[:2]
        if h >= target_h and w >= target_w:
            return image

        pad_h = max(0, target_h - h)
        pad_w = max(0, target_w - w)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        return cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT)

    def _crop_pair(self, input_img, gt_img):
        crop_size = self.opt.get("crop_size")
        if crop_size is None:
            return input_img, gt_img

        if isinstance(crop_size, (list, tuple)):
            crop_h, crop_w = int(crop_size[0]), int(crop_size[1])
        else:
            crop_h, crop_w = int(crop_size), int(crop_size)

        input_img = self._pad_image(input_img, crop_h, crop_w)
        gt_img = self._pad_image(gt_img, crop_h, crop_w)

        h, w = input_img.shape[:2]
        if h == crop_h and w == crop_w:
            return input_img, gt_img

        top = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)

        input_img = input_img[top:top + crop_h, left:left + crop_w, :]
        gt_img = gt_img[top:top + crop_h, left:left + crop_w, :]
        return input_img, gt_img

    def _concat_hiseq(self, input_img, orig_img):
        hiseq = cv2.cvtColor(hiseq_color_cv2_img(orig_img), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if self.opt.get("hiseq_random_cat", False) and np.random.rand() < self.opt.get("hiseq_random_cat_p", 0.5):
            return np.concatenate([hiseq, input_img], axis=2)
        return np.concatenate([input_img, hiseq], axis=2)

    def _concat_position_encoding(self, input_img):
        h, w, _ = input_img.shape
        l_freq = int(self.opt.get("position_encoding_L", 1))
        position_encoding = generate_position_encoding(h, w, l_freq).numpy()
        return np.concatenate([input_img, position_encoding], axis=2)

    def _to_tensor(self, image, use_precomputed=False):
        tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()

        if self.should_normalize and use_precomputed:
            base_mean = self.mean
            base_std = self.std
        else:
            base_mean = [0.5, 0.5, 0.5]
            base_std = [0.5, 0.5, 0.5]

        repeats = tensor.shape[0] // 3
        remainder = tensor.shape[0] % 3
        mean = base_mean * repeats + [0.5] * remainder
        std = base_std * repeats + [0.5] * remainder
        normalize(tensor, mean, std, inplace=True)
        return tensor

    def _placeholder_pair(self, index):
        crop_size = self.opt.get("crop_size", 256)
        if isinstance(crop_size, (list, tuple)):
            h, w = int(crop_size[0]), int(crop_size[1])
        else:
            h = w = int(crop_size)

        rng = np.random.default_rng(seed=index)
        gt_img = rng.random((h, w, 3), dtype=np.float32)
        noisy = rng.normal(loc=0.0, scale=0.05, size=(h, w, 3)).astype(np.float32)
        input_img = np.clip(gt_img + noisy, 0.0, 1.0)
        return input_img, gt_img

    def __getitem__(self, index):
        if self.use_placeholder:
            input_img, gt_img = self._placeholder_pair(index)
            lq_path = f"placeholder_{index}.png"
        else:
            input_rgb = self.inputs[index].astype(np.float32) / 255.0
            gt_rgb = self.gts[index].astype(np.float32) / 255.0
            input_img = input_rgb
            gt_img = gt_rgb
            lq_path = str(self.pairs[index][0])

        if self.opt.get("bright_aug", False):
            low, high = self.opt.get("bright_aug_range", [0.5, 1.5])
            input_img = np.clip(input_img * np.random.uniform(low, high), 0.0, 1.0)

        if self.opt.get("concat_with_hiseq", False):
            if self.use_placeholder:
                orig = (input_img[:, :, :3] * 255.0).astype(np.uint8)
                orig = cv2.cvtColor(orig, cv2.COLOR_RGB2BGR)
            else:
                orig = self.inputs[index]
            input_img = self._concat_hiseq(input_img, orig)

        if self.opt.get("concat_with_position_encoding", False):
            input_img = self._concat_position_encoding(input_img)

        if self.opt.get("use_flip", False) and random.random() < 0.5:
            gt_img = np.ascontiguousarray(np.flip(gt_img, axis=1))
            input_img = np.ascontiguousarray(np.flip(input_img, axis=1))

        if self.opt.get("input_mode", "crop") == "crop":
            input_img, gt_img = self._crop_pair(input_img, gt_img)

        gt_tensor = self._to_tensor(gt_img, use_precomputed=False)
        lr_tensor = self._to_tensor(input_img, use_precomputed=True)

        return {
            "LR": lr_tensor,
            "HR": gt_tensor,
            "lq_path": f"{index}_{lq_path}",
        }

    def __len__(self):
        if self.use_placeholder:
            return self.placeholder_length
        return len(self.pairs)


@DATASET_REGISTRY.register()
class UW_Dataset(UW_Dataset_Preloaded2):
    pass


@DATASET_REGISTRY.register()
class UW_Dataset_Preloaded(UW_Dataset_Preloaded2):
    pass


@DATASET_REGISTRY.register()
class Test_UW_Dataset_Preloaded(UW_Dataset_Preloaded2):
    pass


@DATASET_REGISTRY.register()
class CVMIGDataset(UW_Dataset_Preloaded2):
    pass
