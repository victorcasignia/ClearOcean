import os
import time
from collections import OrderedDict
from copy import deepcopy

import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel

from basicsr.models import lr_scheduler as lr_scheduler
from basicsr.utils import get_root_logger
from basicsr.utils.dist_util import master_only


class BaseModelCompat:
    """Portable BaseModel for ClearOcean that works with pip-installed basicsr."""

    def __init__(self, opt):
        self.opt = opt
        device_name = opt.get("device")
        if device_name is None:
            if torch.cuda.is_available() and int(opt.get("num_gpu", 0)) > 0:
                device_name = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device_name = "mps"
            else:
                device_name = "cpu"
        self.device = torch.device(device_name)
        self.is_train = opt["is_train"]
        self.schedulers = []
        self.optimizers = []

    def feed_data(self, data):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        pass

    def save(self, epoch, current_iter):
        pass

    def validation(self, dataloader, current_iter, tb_logger, save_img=False):
        if self.opt["dist"]:
            self.dist_validation(dataloader, current_iter, tb_logger, save_img)
        else:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def get_current_log(self):
        return self.log_dict

    def model_to_device(self, net):
        net = net.to(self.device)
        if self.opt["dist"]:
            if self.device.type != "cuda":
                raise ValueError("Distributed mode currently requires CUDA device.")
            find_unused_parameters = self.opt.get("find_unused_parameters", False)
            net = DistributedDataParallel(
                net,
                device_ids=[torch.cuda.current_device()],
                find_unused_parameters=find_unused_parameters,
                broadcast_buffers=False,
            )
        elif self.opt.get("num_gpu", 0) > 1 and self.device.type == "cuda":
            net = DataParallel(net)
        return net

    def get_optimizer(self, optim_type, params, lr, **kwargs):
        if optim_type == "Adam":
            return torch.optim.Adam(params, lr, **kwargs)
        if optim_type == "AdamW":
            return torch.optim.AdamW(params, lr, **kwargs)
        raise NotImplementedError(f"optimizer {optim_type} is not supported")

    def setup_schedulers(self):
        train_opt = self.opt["train"]
        scheduler_type = train_opt["scheduler"].pop("type")
        if scheduler_type in ["MultiStepLR", "MultiStepRestartLR"]:
            for optimizer in self.optimizers:
                self.schedulers.append(lr_scheduler.MultiStepRestartLR(optimizer, **train_opt["scheduler"]))
        elif scheduler_type == "CosineAnnealingRestartLR":
            for optimizer in self.optimizers:
                self.schedulers.append(lr_scheduler.CosineAnnealingRestartLR(optimizer, **train_opt["scheduler"]))
        elif scheduler_type == "PolyLR":
            for index, optimizer in enumerate(self.optimizers):
                self.schedulers.append(lr_scheduler.PolyLR(optimizer, train_opt["scheduler"], g_flag=(index == 0)))
        else:
            raise NotImplementedError(f"Scheduler {scheduler_type} is not implemented")

    def get_bare_model(self, net):
        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net = net.module
        return net

    @master_only
    def print_network(self, net):
        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net_cls_str = f"{net.__class__.__name__} - {net.module.__class__.__name__}"
        else:
            net_cls_str = f"{net.__class__.__name__}"

        net = self.get_bare_model(net)
        net_str = str(net)
        net_params = sum(map(lambda x: x.numel(), net.parameters()))

        logger = get_root_logger()
        logger.info(f"Network: {net_cls_str}, with parameters: {net_params:,d}")
        logger.info(net_str)

    def _set_lr(self, lr_groups_l):
        for optimizer, lr_groups in zip(self.optimizers, lr_groups_l):
            for param_group, lr in zip(optimizer.param_groups, lr_groups):
                param_group["lr"] = lr

    def _get_init_lr(self):
        init_lr_groups_l = []
        for optimizer in self.optimizers:
            init_lr_groups_l.append([v["initial_lr"] for v in optimizer.param_groups])
        return init_lr_groups_l

    def update_learning_rate(self, current_iter, warmup_iter=-1):
        if current_iter > 1:
            for scheduler in self.schedulers:
                scheduler.step()
        if current_iter < warmup_iter:
            init_lr_g_l = self._get_init_lr()
            warm_up_lr_l = []
            for init_lr_g in init_lr_g_l:
                warm_up_lr_l.append([v / warmup_iter * current_iter for v in init_lr_g])
            self._set_lr(warm_up_lr_l)

    def get_current_learning_rate(self):
        return [param_group["lr"] for param_group in self.optimizers[0].param_groups]

    @master_only
    def save_network(self, net, net_label, current_iter, param_key="params"):
        if current_iter == -1:
            current_iter = "latest"
        save_filename = f"{net_label}_{current_iter}.pth"
        save_path = os.path.join(self.opt["path"]["models"], save_filename)

        net = net if isinstance(net, list) else [net]
        param_key = param_key if isinstance(param_key, list) else [param_key]
        assert len(net) == len(param_key), "The lengths of net and param_key should be the same."

        save_dict = {}
        for net_, param_key_ in zip(net, param_key):
            net_ = self.get_bare_model(net_)
            state_dict = net_.state_dict()
            for key, param in state_dict.items():
                if key.startswith("module."):
                    key = key[7:]
                state_dict[key] = param.cpu()
            save_dict[param_key_] = state_dict

        retry = 3
        while retry > 0:
            try:
                torch.save(save_dict, save_path)
            except Exception as err:
                logger = get_root_logger()
                logger.warning("Save model error: %s, remaining retry times: %d", err, retry - 1)
                time.sleep(1)
            else:
                break
            finally:
                retry -= 1

    def _print_different_keys_loading(self, crt_net, load_net, strict=True):
        crt_net = self.get_bare_model(crt_net).state_dict()
        crt_net_keys = set(crt_net.keys())
        load_net_keys = set(load_net.keys())

        logger = get_root_logger()
        if crt_net_keys != load_net_keys:
            logger.warning("Current net - loaded net:")
            for key in sorted(list(crt_net_keys - load_net_keys)):
                logger.warning("  %s", key)
            logger.warning("Loaded net - current net:")
            for key in sorted(list(load_net_keys - crt_net_keys)):
                logger.warning("  %s", key)

        if not strict:
            common_keys = crt_net_keys & load_net_keys
            for key in common_keys:
                if crt_net[key].size() != load_net[key].size():
                    logger.warning(
                        "Size different, ignore [%s]: crt_net: %s; load_net: %s",
                        key,
                        crt_net[key].shape,
                        load_net[key].shape,
                    )
                    load_net[key + ".ignore"] = load_net.pop(key)

    def load_network(self, net, load_path, strict=True, param_key="params"):
        logger = get_root_logger()
        net = self.get_bare_model(net)
        load_net = torch.load(load_path, map_location=lambda storage, loc: storage)
        if param_key is not None:
            if param_key not in load_net and "params" in load_net:
                param_key = "params"
                logger.info("Loading: params_ema does not exist, use params.")
            load_net = load_net[param_key]

        logger.info("Loading %s model from %s with param key [%s].", net.__class__.__name__, load_path, param_key)

        for key, value in deepcopy(load_net).items():
            if key.startswith("module."):
                load_net[key[7:]] = value
                load_net.pop(key)

        self._print_different_keys_loading(net, load_net, strict)
        net.load_state_dict(load_net, strict=strict)

    @master_only
    def save_training_state(self, epoch, current_iter):
        if current_iter == -1:
            return

        state = {"epoch": epoch, "iter": current_iter, "optimizers": [], "schedulers": []}
        for optimizer in self.optimizers:
            state["optimizers"].append(optimizer.state_dict())
        for scheduler in self.schedulers:
            state["schedulers"].append(scheduler.state_dict())

        save_filename = f"{current_iter}.state"
        save_path = os.path.join(self.opt["path"]["training_states"], save_filename)

        retry = 3
        while retry > 0:
            try:
                torch.save(state, save_path)
            except Exception as err:
                logger = get_root_logger()
                logger.warning("Save training state error: %s, remaining retry times: %d", err, retry - 1)
                time.sleep(1)
            else:
                break
            finally:
                retry -= 1

    def resume_training(self, resume_state):
        resume_optimizers = resume_state["optimizers"]
        resume_schedulers = resume_state["schedulers"]
        assert len(resume_optimizers) == len(self.optimizers), "Wrong lengths of optimizers"
        assert len(resume_schedulers) == len(self.schedulers), "Wrong lengths of schedulers"
        for index, optimizer_state in enumerate(resume_optimizers):
            self.optimizers[index].load_state_dict(optimizer_state)
        for index, scheduler_state in enumerate(resume_schedulers):
            self.schedulers[index].load_state_dict(scheduler_state)

    def reduce_loss_dict(self, loss_dict):
        with torch.no_grad():
            if self.opt["dist"]:
                keys = []
                losses = []
                for name, value in loss_dict.items():
                    keys.append(name)
                    losses.append(value)
                losses = torch.stack(losses, 0)
                torch.distributed.reduce(losses, dst=0)
                if self.opt["rank"] == 0:
                    losses /= self.opt["world_size"]
                loss_dict = {key: loss for key, loss in zip(keys, losses)}

            log_dict = OrderedDict()
            for name, value in loss_dict.items():
                log_dict[name] = value.mean().item()
            return log_dict
