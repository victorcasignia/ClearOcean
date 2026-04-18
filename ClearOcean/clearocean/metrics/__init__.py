import importlib
from os import path as osp

from basicsr.utils import scandir

metrics_folder = osp.dirname(osp.abspath(__file__))
metric_filenames = [
    osp.splitext(osp.basename(v))[0]
    for v in scandir(metrics_folder)
    if v.endswith(".py") and not v.startswith("__")
]
_metric_modules = [importlib.import_module(f"clearocean.metrics.{file_name}") for file_name in metric_filenames]
