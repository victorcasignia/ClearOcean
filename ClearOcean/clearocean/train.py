# flake8: noqa
import os.path as osp
import sys

_CURRENT_DIR = osp.dirname(osp.abspath(__file__))
_CLEAROCEAN_ROOT = osp.abspath(osp.join(_CURRENT_DIR, osp.pardir))

for _path in [_CLEAROCEAN_ROOT]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from clearocean.runtime_train import train_pipeline

import clearocean.archs
import clearocean.data
import clearocean.metrics
import clearocean.models

if __name__ == '__main__':
    root_path = _CLEAROCEAN_ROOT
    train_pipeline(root_path)
