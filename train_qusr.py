"""
QUSR Training Script
"""

import os
import gc
import lpips
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
import sys
import logging
from datetime import datetime
from pathlib import Path
from accelerate.utils import ProjectConfiguration
from accelerate import DistributedDataParallelKwargs

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

_QUSR_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _QUSR_ROOT)
for iqa_path in [os.path.join(_QUSR_ROOT, '..', 'IQA-PyTorch-main'), os.path.join(_QUSR_ROOT, '..', 'IQA-PyTorch-main', 'IQA-PyTorch-main')]:
    if os.path.exists(iqa_path):
        sys.path.append(iqa_path)
        break
import pyiqa

import diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler
import transformers

from qusr import CSDLoss, QUSR
from src.my_utils.training_utils import parse_args
from src.my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix


def main(args):
    log_dir = Path(args.output_dir, "logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_log_{timestamp}.txt"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)
    logger.info("=== QUSR Training ===")
    logger.info(f"Arguments: {vars(args)}")

    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs],
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)

    net_qusr = QUSR(args)

    if args.enable_xformers_memory_efficient_attention and is_xformers_available():
        net_qusr.unet.enable_xformers_memory_efficient_attention()

    if args.gradient_checkpointing:
        net_qusr.unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    net_csd = CSDLoss(args=args, accelerator=accelerator)
    net_csd.requires_grad_(False)

    net_lpips = lpips.LPIPS(net='vgg').cuda()
    net_lpips.requires_grad_(False)

    psnr_metric = pyiqa.create_metric('psnr', test_y_channel=True, color_space='ycbcr', device='cuda')
    ssim_metric = pyiqa.create_metric('ssim', test_y_channel=True, color_space='ycbcr', device='cuda')
    lpips_metric = pyiqa.create_metric('lpips', device='cuda')
    dists_metric = pyiqa.create_metric('dists', device='cuda')
    clipiqa_metric = pyiqa.create_metric('clipiqa', device='cuda')
    niqe_metric = pyiqa.create_metric('niqe', device='cuda')
    musiq_metric = pyiqa.create_metric('musiq', device='cuda')
    fid_metric = pyiqa.create_metric('fid', device='cuda')

    net_qusr.unet.set_adapter(['default_encoder_sem', 'default_decoder_sem', 'default_others_sem'])
    net_qusr.set_train_sem()

    layers_to_opt = []
    for n, _p in net_qusr.unet.named_parameters():
        if "lora" in n:
            layers_to_opt.append(_p)
    if hasattr(net_qusr, 'enable_uncertainty') and net_qusr.enable_uncertainty:
        for _p in net_qusr.uem.parameters():
            layers_to_opt.append(_p)

    optimizer = torch.optim.AdamW(layers_to_opt, lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay, eps=args.adam_epsilon)
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, power=args.lr_power)

    if args.resume_ckpt:
        ckpt_filename = os.path.basename(args.resume_ckpt)
        if ckpt_filename.startswith("model_") and ckpt_filename.endswith(".pkl"):
            try:
                resume_step = int(ckpt_filename.replace("model_", "").replace(".pkl", ""))
                for _ in range(resume_step):
                    lr_scheduler.step()
            except ValueError:
                pass

    if getattr(args, 'use_online_degradation', False):
        from src.datasets.dataset import PairedSROnlineTxtDataset
        dataset_train = PairedSROnlineTxtDataset(split="train", args=args)
        dataset_val = PairedSROnlineTxtDataset(split="test", args=args)
    else:
        from src.datasets.dataset_pregen import PairedSRPreGenDataset
        if args.consistent_crop_images and os.path.exists(args.consistent_crop_images):
            with open(args.consistent_crop_images, 'r') as f:
                args.consistent_crop_images_list = [line.strip() for line in f.readlines()]
        dataset_train = PairedSRPreGenDataset(split="train", args=args)
        dataset_val = PairedSRPreGenDataset(split="test", args=args)

    dl_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers)
    dl_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0)

    net_qusr, optimizer, dl_train, lr_scheduler = accelerator.prepare(net_qusr, optimizer, dl_train, lr_scheduler)
    net_lpips = accelerator.prepare(net_lpips)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, config=dict(vars(args)))

    global_step = 0
    if args.resume_ckpt:
        ckpt_filename = os.path.basename(args.resume_ckpt)
        if ckpt_filename.startswith("model_") and ckpt_filename.endswith(".pkl"):
            try:
                global_step = int(ckpt_filename.replace("model_", "").replace(".pkl", ""))
            except ValueError:
                pass

    progress_bar = tqdm(range(0, args.max_train_steps), initial=global_step, desc="Steps", disable=not accelerator.is_local_main_process)
    should_stop = False
    lambda_l2 = args.lambda_l2
    lambda_lpips = args.lambda_lpips
    lambda_csd = args.lambda_csd

    for epoch in range(0, args.num_training_epochs):
        if should_stop:
            break
        for step, batch in enumerate(dl_train):
            with accelerator.accumulate(net_qusr):
                x_src = batch["conditioning_pixel_values"]
                x_tgt = batch["output_pixel_values"]
                batch_size = x_src.shape[0]
                batch["prompt"] = ["A high quality image with good details and clarity." for _ in range(batch_size)]

                setattr(args, "use_uncertainty_now", global_step >= 0)
                x_tgt_pred, latents_pred, prompt_embeds, neg_prompt_embeds, uncertainty_map = net_qusr(x_src, x_tgt, batch=batch, args=args)

                loss_l2 = F.mse_loss(x_tgt_pred.float(), x_tgt.float(), reduction="mean") * lambda_l2
                loss_lpips = net_lpips(x_tgt_pred.float(), x_tgt.float()).mean() * lambda_lpips
                loss = loss_l2 + loss_lpips

                use_uncertainty = (
                    hasattr(net_qusr.module, 'enable_uncertainty') and
                    net_qusr.module.enable_uncertainty and
                    uncertainty_map is not None and
                    global_step >= 0
                )

                loss_uncertainty = torch.tensor(0.0, device=x_tgt_pred.device)
                if use_uncertainty:
                    loss_uncertainty = net_qusr.module.compute_uncertainty_loss(x_tgt_pred, x_tgt, uncertainty_map)
                    loss = loss + loss_uncertainty * args.lambda_uncertainty

                loss_csd = net_csd.cal_csd(latents_pred, prompt_embeds, neg_prompt_embeds, args) * lambda_csd
                loss = loss + loss_csd

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    logs = {}
                    if accelerator.is_main_process:
                        logs["loss_csd"] = loss_csd.detach().item()
                        logs["loss_l2"] = loss_l2.detach().item()
                        logs["loss_lpips"] = loss_lpips.detach().item()
                        if use_uncertainty:
                            logs["loss_uncertainty"] = loss_uncertainty.detach().item()
                        progress_bar.set_postfix(**logs)
                        logger.info(f"Step {global_step}: " + " | ".join([f"{k}: {v:.6f}" for k, v in logs.items()]))

                    if global_step % args.checkpointing_steps == 1:
                        outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                        accelerator.unwrap_model(net_qusr).save_model(outf)

                    if global_step % args.eval_freq == 1:
                        os.makedirs(os.path.join(args.output_dir, "eval", f"fid_{global_step}"), exist_ok=True)
                        psnr_values, ssim_values, lpips_values, dists_values = [], [], [], []
                        clipiqa_values, niqe_values, musiq_values = [], [], []

                        for step, batch_val in enumerate(dl_val):
                            x_src = batch_val["conditioning_pixel_values"].cuda()
                            x_tgt = batch_val["output_pixel_values"].cuda()
                            x_basename = batch_val["base_name"][0]
                            B, C, H, W = x_src.shape
                            assert B == 1, "Use batch size 1 for eval."
                            with torch.no_grad():
                                net_qusr.eval()
                                batch_val["prompt"] = ["A high quality image with good details and clarity."]
                                x_tgt_pred, latents_pred, _, _, _ = accelerator.unwrap_model(net_qusr)(x_src, x_tgt, batch=batch_val, args=args)

                                pred_norm = x_tgt_pred * 0.5 + 0.5
                                gt_norm = x_tgt * 0.5 + 0.5
                                psnr_values.append(psnr_metric(pred_norm, gt_norm).item())
                                ssim_values.append(ssim_metric(pred_norm, gt_norm).item())
                                lpips_values.append(lpips_metric(pred_norm, gt_norm).item())
                                dists_values.append(dists_metric(pred_norm, gt_norm).item())
                                clipiqa_values.append(clipiqa_metric(pred_norm).item())
                                niqe_values.append(niqe_metric(pred_norm).item())
                                musiq_values.append(musiq_metric(pred_norm).item())

                                output_pil = transforms.ToPILImage()(x_tgt_pred[0].cpu() * 0.5 + 0.5)
                                input_image = transforms.ToPILImage()(x_src[0].cpu() * 0.5 + 0.5)
                                if args.align_method == 'adain':
                                    output_pil = adain_color_fix(target=output_pil, source=input_image)
                                elif args.align_method == 'wavelet':
                                    output_pil = wavelet_color_fix(target=output_pil, source=input_image)
                                outf = os.path.join(args.output_dir, "eval", f"fid_{global_step}", f"{x_basename}")
                                output_pil.save(outf)

                        avg_psnr = np.mean(psnr_values)
                        avg_ssim = np.mean(ssim_values)
                        avg_lpips = np.mean(lpips_values)
                        avg_dists = np.mean(dists_values)
                        avg_clipiqa = np.mean(clipiqa_values)
                        avg_niqe = np.mean(niqe_values)
                        avg_musiq = np.mean(musiq_values)
                        eval_output_dir = os.path.join(args.output_dir, "eval", f"fid_{global_step}")
                        gt_dir = args.dataset_test_folder if hasattr(args, 'dataset_test_folder') else "preset/testfolder"
                        try:
                            fid_value = fid_metric(gt_dir, eval_output_dir).item()
                        except Exception:
                            fid_value = 0.0

                        logs.update({"avg_psnr": avg_psnr, "avg_ssim": avg_ssim, "avg_lpips": avg_lpips, "avg_dists": avg_dists,
                            "avg_clipiqa": avg_clipiqa, "avg_niqe": avg_niqe, "avg_musiq": avg_musiq, "fid": fid_value})
                        if accelerator.is_main_process:
                            logger.info(f"Validation Step {global_step}: PSNR={avg_psnr:.4f}, SSIM={avg_ssim:.4f}, FID={fid_value:.4f}")
                        net_qusr.train()
                        gc.collect()
                        torch.cuda.empty_cache()

                    accelerator.log(logs, step=global_step)

                if global_step >= args.max_train_steps:
                    should_stop = True
                    break

    logger.info("Training completed.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
