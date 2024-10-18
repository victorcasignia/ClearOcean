# flake8: noqa
# import os.path as osp
# from basicsr.train import train_pipeline
from basicsr.utils.options import copy_opt_file, dict2str, parse_options

# import pydiff.archs
# import pydiff.data
# import pydiff.models
from torchviz import make_dot
import torch
from pydiff.models.pydiff_model import PyDiffModel

if __name__ == '__main__':
    opts, args = parse_options('../options/uw_train_v7_lsui_uieb_with_ca_affine.yaml')
    print(opts['num_gpu'])
    pydiff = PyDiffModel(opts)
    model = pydiff.unet.cuda()
    x = torch.randn(8, 13, 256, 256).cuda()
    time = torch.randn(8, 1).cuda()
    y = model(x, time)
    make_dot(y, params=dict(list(model.named_parameters()))).render("rnn_torchviz", format="png")
