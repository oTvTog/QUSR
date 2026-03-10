"""
Image Quality Assessment for QUSR outputs.
Evaluates PSNR, SSIM, LPIPS, DISTS, CLIPIQA, NIQE, MUSIQ, MANIQA, FID.
Requires: pyiqa, basicsr
"""

import os
import sys
import glob
import argparse
import logging
from datetime import datetime
import time

import cv2
import numpy as np
import torch

# Add IQA path (parent or sibling)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_QUSR_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
for _path in [os.path.join(_QUSR_ROOT, '..', 'IQA-PyTorch-main'), 
              os.path.join(_QUSR_ROOT, '..', 'IQA-PyTorch-main', 'IQA-PyTorch-main')]:
    if os.path.exists(_path):
        sys.path.insert(0, _path)
        break

import pyiqa
from basicsr.utils import img2tensor


def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        fmt='%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
        datefmt='%y-%m-%d %H:%M:%S'
    )
    logger.setLevel(level)
    logger.handlers = []
    if tofile:
        log_file = os.path.join(root, f"{phase}_{get_timestamp()}.log")
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)
    return logger


def find_gt_path(gt_dir, sr_basename):
    """Match SR filename to GT. Tries: _LR4->_HR, _LR4->_gt, etc."""
    base = sr_basename.replace('_LR4.png', '').replace('.png', '')
    for suffix in ['_HR.png', '_gt.png', '.png']:
        p = os.path.join(gt_dir, base + suffix)
        if os.path.exists(p):
            return p
    return None


def main():
    parser = argparse.ArgumentParser(description="QUSR Image Quality Assessment")
    parser.add_argument("--inp_imgs", nargs="+", required=True, help="SR output directory(ies)")
    parser.add_argument("--gt_imgs", nargs="+", required=True, help="GT directory(ies)")
    parser.add_argument("--log", type=str, required=True, help="Log output directory")
    parser.add_argument("--log_name", type=str, default="METRICS", help="Log file base name")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.log, exist_ok=True)

    log_name = os.path.basename(os.path.normpath(args.inp_imgs[0])) if args.inp_imgs else args.log_name
    logger = setup_logger('base', args.log, f'test_{log_name}', level=logging.INFO, screen=True, tofile=True)
    logger.info("===== Configuration =====")
    logger.info(f"  inp_imgs: {args.inp_imgs}\n  gt_imgs: {args.gt_imgs}\n  log: {args.log}\n  log_name: {args.log_name}")
    logger.info("==========================\n")

    logger.info("Initializing IQA metrics...")
    iqa_metrics = {
        'PSNR': pyiqa.create_metric('psnr', test_y_channel=True, color_space='ycbcr').to(device),
        'SSIM': pyiqa.create_metric('ssim', test_y_channel=True, color_space='ycbcr').to(device),
        'LPIPS': pyiqa.create_metric('lpips', device=device),
        'DISTS': pyiqa.create_metric('dists', device=device),
        'CLIPIQA': pyiqa.create_metric('clipiqa', device=device),
        'NIQE': pyiqa.create_metric('niqe', device=device),
        'MUSIQ': pyiqa.create_metric('musiq', device=device),
        'MANIQA': pyiqa.create_metric('maniqa-pipal', device=device),
    }
    fid_metric = pyiqa.create_metric('fid', device=device)
    logger.info("IQA metrics initialized.\n")

    if len(args.inp_imgs) != len(args.gt_imgs):
        logger.error("Number of inp_imgs and gt_imgs must match.")
        sys.exit(1)

    for dir_idx, init_dir in enumerate(args.inp_imgs):
        gt_dir = args.gt_imgs[dir_idx]
        img_sr_list = sorted(glob.glob(os.path.join(init_dir, '*.png')))
        img_gt_list = sorted(glob.glob(os.path.join(gt_dir, '*.png')))
        dir_name = os.path.basename(os.path.normpath(init_dir))
        logger.info(f"Directory [{dir_name}]: {len(img_gt_list)} GT images vs {len(img_sr_list)} SR images.\n")
        logger.info("===== Starting Evaluation =====\n")
        logger.info(f"Testing Directory: [{dir_name}]")

        metrics_accum = {k: 0.0 for k in iqa_metrics}
        valid_count = 0

        for sr_path in img_sr_list:
            img_name = os.path.basename(sr_path)
            gt_path = find_gt_path(gt_dir, img_name)
            if gt_path is None:
                gt_path = os.path.join(gt_dir, img_name.replace('_LR4.png', '_HR.png'))
            if not os.path.exists(gt_path):
                logger.warning(f"GT not found for {img_name}, skipping.")
                continue

            start_time = time.time()
            sr_img = cv2.imread(sr_path, cv2.IMREAD_COLOR)
            gt_img = cv2.imread(gt_path, cv2.IMREAD_COLOR)
            if sr_img is None or gt_img is None:
                logger.warning(f"Read failed: {img_name}")
                continue

            sr_tensor = img2tensor(sr_img, bgr2rgb=True, float32=True).unsqueeze(0).to(device).contiguous() / 255.0
            gt_tensor = img2tensor(gt_img, bgr2rgb=True, float32=True).unsqueeze(0).to(device).contiguous() / 255.0

            with torch.no_grad():
                metrics = {}
                for name, metric in iqa_metrics.items():
                    if name in ['CLIPIQA', 'NIQE', 'MUSIQ', 'MANIQA']:
                        metrics[name] = metric(sr_tensor).item()
                    else:
                        metrics[name] = metric(sr_tensor, gt_tensor).item()

            for k in metrics_accum:
                metrics_accum[k] += metrics[k]
            valid_count += 1
            runtime = time.time() - start_time
            metrics_str = "; ".join([f"{k}: {v:.6f}" for k, v in metrics.items()])
            logger.info(f"{dir_name}/{img_name} | {metrics_str} | Runtime: {runtime:.2f} sec")

        if valid_count == 0:
            logger.warning("No valid image pairs found.")
            continue

        avg_metrics = {k: round(v / valid_count, 4) for k, v in metrics_accum.items()}
        fid_start = time.time()
        fid_value = fid_metric(gt_dir, init_dir).item()
        fid_runtime = time.time() - fid_start

        avg_str = "; ".join([f"{k}: {v:.4f}" for k, v in avg_metrics.items()])
        logger.info(f"\n===== Average Metrics for [{dir_name}] =====\n{avg_str} | FID: {fid_value:.6f} | FID Runtime: {fid_runtime:.2f} sec\n")

    logger.info("===== Evaluation Completed =====")


if __name__ == "__main__":
    main()
