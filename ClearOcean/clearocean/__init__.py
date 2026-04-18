# flake8: noqa
from .compat import patch_torchvision_compat

patch_torchvision_compat()

from .archs import *
from .data import *
from .metrics import *
from .models import *
