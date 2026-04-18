import copy
import os.path as osp
from collections import OrderedDict

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.parallel import DataParallel, DistributedDataParallel

from basicsr.archs import build_network
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY

from clearocean.models.base_model_compat import BaseModelCompat
from scripts.utils import pad_tensor_back


@MODEL_REGISTRY.register()
class ClearOceanModel(BaseModelCompat):
    """Lean ClearOcean model path used by the clearocean config."""

    def __init__(self, opt):
        super().__init__(opt)

        self.unet = self.model_to_device(build_network(opt["network_unet"]))
        opt["network_ddpm"]["denoise_fn"] = self.unet

        self.global_corrector = self.model_to_device(build_network(opt["network_global_corrector"]))
        opt["network_ddpm"]["color_fn"] = self.global_corrector

        self.ddpm = self.model_to_device(build_network(opt["network_ddpm"]))
        self.bare_model = self.get_bare_model(self.ddpm)

        self.bare_model.set_new_noise_schedule(schedule_opt=opt["ddpm_schedule"], device=self.device)
        self.bare_model.set_loss(device=self.device)
        self.print_network(self.ddpm)

        load_path = opt["path"].get("pretrain_network_g")
        if load_path is not None:
            param_key = opt["path"].get("param_key_g", "params")
            strict = opt["path"].get("strict_load_g", True)
            self.load_network(self.ddpm, load_path, strict, param_key)

        self.lpips_model = None
        self.lpips_device = self.device
        self._setup_lpips_if_needed()

        if self.is_train:
            self.init_training_settings()

    def _setup_lpips_if_needed(self):
        metrics = self.opt.get("val", {}).get("metrics", {})
        needs_lpips = any("lpips" in metric_opt.get("type", "") for metric_opt in metrics.values())
        if not needs_lpips:
            return

        import lpips

        if self.device.type == "mps":
            self.lpips_device = torch.device("cpu")
        self.lpips_model = lpips.LPIPS(net="alex").to(self.lpips_device)
        self.lpips_model.eval()

    def init_training_settings(self):
        self.ddpm.train()
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt["train"]
        logger = get_root_logger()

        params = []
        for name, param in self.ddpm.named_parameters():
            if self.opt["train"].get("frozen_denoise", False) and "denoise" in name:
                logger.info("Frozen parameter: %s", name)
                continue
            params.append(param)

        optim_type = train_opt["optim_g"].pop("type")
        lr = train_opt["optim_g"]["lr"]
        self.optimizer_g = self.get_optimizer(optim_type, [{"params": params, "lr": lr}], lr, betas=(0.0, 0.99))
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.LR = data["LR"].to(self.device)
        self.HR = data["HR"].to(self.device)
        if "pad_left" in data:
            self.pad_left = data["pad_left"].to(self.device)
            self.pad_right = data["pad_right"].to(self.device)
            self.pad_top = data["pad_top"].to(self.device)
            self.pad_bottom = data["pad_bottom"].to(self.device)

    def optimize_parameters(self, current_iter):
        train_type = self.opt["train"].get("train_type", "")
        if "ddpm_cs" not in train_type:
            raise ValueError("train_type must contain ddpm_cs for this model.")

        self.optimizer_g.zero_grad()
        pred_noise, noise, x_recon_cs, x_start, _, color_scale = self.ddpm(
            self.HR,
            self.LR,
            train_type=train_type,
            different_t_in_one_batch=self.opt["train"].get("different_t_in_one_batch"),
            t_sample_type=self.opt["train"].get("t_sample_type"),
            pred_type=self.opt["train"].get("pred_type"),
            clip_noise=self.opt["train"].get("clip_noise"),
            color_shift=self.opt["train"].get("color_shift"),
            color_shift_with_schedule=self.opt["train"].get("color_shift_with_schedule"),
            t_range=self.opt["train"].get("t_range"),
            cs_on_shift=self.opt["train"].get("cs_on_shift"),
            cs_shift_range=self.opt["train"].get("cs_shift_range"),
            t_border=self.opt["train"].get("t_border"),
            down_uniform=self.opt["train"].get("down_uniform", False),
            down_hw_split=self.opt["train"].get("down_hw_split", False),
            pad_after_crop=self.opt["train"].get("pad_after_crop", False),
            input_mode=self.opt["train"].get("input_mode"),
            crop_size=self.opt["train"].get("crop_size"),
            divide=self.opt["train"].get("divide"),
            frozen_denoise=self.opt["train"].get("frozen_denoise"),
            cs_independent=self.opt["train"].get("cs_independent"),
            shift_x_recon_detach=self.opt["train"].get("shift_x_recon_detach"),
        )

        loss_dict = OrderedDict()
        total_loss = 0

        l_g_x0 = F.l1_loss(x_recon_cs, x_start) * self.opt["train"].get("l_g_x0_w", 1.0)
        gamma_limit = self.opt["train"].get("gamma_limit_train")
        if gamma_limit is not None and color_scale <= gamma_limit:
            l_g_x0 = l_g_x0 * 1e-12
        loss_dict["l_g_x0"] = l_g_x0
        total_loss += l_g_x0

        if not self.opt["train"].get("frozen_denoise", False):
            l_g_noise = F.l1_loss(pred_noise, noise)
            loss_dict["l_g_noise"] = l_g_noise
            total_loss += l_g_noise

        total_loss.backward()
        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        with torch.no_grad():
            self.bare_model.denoise_fn.eval()
            self.output = self.bare_model.ddim_pyramid_sample(
                self.LR,
                pyramid_list=self.opt["val"].get("pyramid_list"),
                continous=self.opt["val"].get("ret_process", False),
                ddim_timesteps=self.opt["val"].get("ddim_timesteps", 50),
                return_pred_noise=self.opt["val"].get("return_pred_noise", False),
                return_x_recon=self.opt["val"].get("ret_x_recon", False),
                ddim_discr_method=self.opt["val"].get("ddim_discr_method", "uniform"),
                ddim_eta=self.opt["val"].get("ddim_eta", 0.0),
                pred_type=self.opt["val"].get("pred_type", "noise"),
                clip_noise=self.opt["val"].get("clip_noise", False),
                save_noise=self.opt["val"].get("save_noise", False),
                color_gamma=self.opt["val"].get("color_gamma"),
                color_times=self.opt["val"].get("color_times", 1),
                return_all=self.opt["val"].get("ret_all", False),
                fine_diffV2=self.opt["val"].get("fine_diffV2", False),
                fine_diffV2_st=self.opt["val"].get("fine_diffV2_st", 200),
                fine_diffV2_num_timesteps=self.opt["val"].get("fine_diffV2_num_timesteps", 20),
                do_some_global_deg=self.opt["val"].get("do_some_global_deg", False),
                use_up_v2=self.opt["val"].get("use_up_v2", False),
            )
            self.bare_model.denoise_fn.train()

            if hasattr(self, "pad_left") and not self.opt["val"].get("ret_process", False):
                self.output = pad_tensor_back(self.output, self.pad_left, self.pad_right, self.pad_top, self.pad_bottom)
                self.LR = pad_tensor_back(self.LR, self.pad_left, self.pad_right, self.pad_top, self.pad_bottom)
                self.HR = pad_tensor_back(self.HR, self.pad_left, self.pad_right, self.pad_top, self.pad_bottom)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt["rank"] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt["name"]
        with_metrics = self.opt.get("val", {}).get("metrics") is not None
        if self.opt["val"].get("ret_process", False):
            with_metrics = False

        if with_metrics:
            self.metric_results = {metric: 0.0 for metric in self.opt["val"]["metrics"]}

        metric_data_np = {}
        metric_data_torch = {}

        for idx, val_data in enumerate(dataloader):
            if (
                not self.opt["val"].get("cal_all", False)
                and not self.opt["val"].get("cal_score", False)
                and int(self.opt["ddpm_schedule"]["n_timestep"]) >= 4
                and idx >= 3
            ):
                break

            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals["sr"]], min_max=(-1, 1))
            gt_img = tensor2img([visuals["gt"]], min_max=(-1, 1))
            lq_img = tensor2img([visuals["lq"]], min_max=(-1, 1))

            if self.opt["val"].get("use_kind_align", False):
                gt_mean = np.mean(gt_img)
                sr_mean = np.mean(sr_img)
                sr_img = np.clip(sr_img * gt_mean / max(sr_mean, 1e-8), 0, 255).astype("uint8")

            metric_data_np["img"] = sr_img
            metric_data_np["img2"] = gt_img
            metric_data_torch["img"] = self.output
            metric_data_torch["img2"] = self.HR

            if save_img:
                img_name = osp.splitext(osp.basename(val_data["lq_path"][0]))[0]
                if self.opt["is_train"]:
                    save_img_path = osp.join(self.opt["path"]["visualization"], img_name, f"{img_name}_{current_iter}.png")
                else:
                    save_img_path = osp.join(self.opt["path"]["visualization"], dataset_name, f"{img_name}_{self.opt['name']}.png")
                if idx < self.opt["val"].get("show_num", 3) or self.opt["val"].get("show_all", False):
                    if not self.opt["val"].get("ret_process", False):
                        imwrite(np.concatenate([lq_img, sr_img, gt_img], axis=1), save_img_path)
                    else:
                        imwrite(sr_img, save_img_path)

            if with_metrics:
                for metric_name, metric_opt in self.opt["val"]["metrics"].items():
                    metric_opt_local = copy.deepcopy(metric_opt)
                    metric_type = metric_opt_local.get("type", "")

                    if "lpips" in metric_type:
                        metric_opt_local["device"] = self.lpips_device
                        metric_opt_local["model"] = self.lpips_model

                    if "pytorch" in metric_type:
                        self.metric_results[metric_name] += float(calculate_metric(metric_data_torch, metric_opt_local))
                    else:
                        self.metric_results[metric_name] += float(calculate_metric(metric_data_np, metric_opt_local))

            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        if with_metrics and idx >= 0:
            for metric in self.metric_results:
                self.metric_results[metric] /= (idx + 1)
            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        logger = get_root_logger()
        log_str = f"Validation {dataset_name}\n"
        for metric, value in self.metric_results.items():
            log_str += f"\t # {metric}: {value:.4f}\n"
        logger.info(log_str)

        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f"metrics/{metric}", value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        if self.LR.shape != self.output.shape:
            self.LR = F.interpolate(self.LR, self.output.shape[2:])
            self.HR = F.interpolate(self.HR, self.output.shape[2:])
        out_dict["gt"] = self.HR.detach().cpu()
        out_dict["sr"] = self.output.detach().cpu()
        out_dict["lq"] = self.LR[:, :3, :, :].detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network([self.ddpm], "net_g", current_iter, param_key=["params"])


@MODEL_REGISTRY.register()
class PyDiffModel(ClearOceanModel):
    """Backward-compatible alias for older option files."""

    pass
