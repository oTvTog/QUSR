"""
QUSR Inference Script
"""

import os
import argparse
import random
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import glob

from qusr import QUSR_eval


def seed_everything(seed):
    """Fix random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
from src.my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix


def run_qusr(args):
    model = QUSR_eval(args)
    model.set_eval()

    if os.path.isdir(args.input_image):
        image_names = sorted(glob.glob(f'{args.input_image}/*.png')) + sorted(glob.glob(f'{args.input_image}/*.jpg'))
    else:
        image_names = [args.input_image]

    os.makedirs(args.output_dir, exist_ok=True)
    print(f'Processing {len(image_names)} images.')

    time_records = []
    for image_name in image_names:
        input_image = Image.open(image_name).convert('RGB')
        ori_width, ori_height = input_image.size
        rscale = args.upscale
        resize_flag = False

        if ori_width < args.process_size // rscale or ori_height < args.process_size // rscale:
            scale = (args.process_size // rscale) / min(ori_width, ori_height)
            input_image = input_image.resize((int(scale * ori_width), int(scale * ori_height)))
            resize_flag = True

        input_image = input_image.resize((input_image.size[0] * rscale, input_image.size[1] * rscale))
        new_width = input_image.width - input_image.width % 8
        new_height = input_image.height - input_image.height % 8
        input_image = input_image.resize((new_width, new_height), Image.LANCZOS)
        bname = os.path.basename(image_name)

        with torch.no_grad():
            c_t = F.to_tensor(input_image).unsqueeze(0).cuda() * 2 - 1
            inference_time, output_image = model(args.default, c_t, image_name=image_name)

        print(f"Inference time: {inference_time:.4f}s")
        time_records.append(inference_time)

        output_image = output_image * 0.5 + 0.5
        output_image = torch.clip(output_image, 0, 1)
        output_pil = transforms.ToPILImage()(output_image[0].cpu())

        if args.align_method == 'adain':
            output_pil = adain_color_fix(target=output_pil, source=input_image)
        elif args.align_method == 'wavelet':
            output_pil = wavelet_color_fix(target=output_pil, source=input_image)

        if resize_flag:
            output_pil = output_pil.resize((int(args.upscale * ori_width), int(args.upscale * ori_height)))
        output_pil.save(os.path.join(args.output_dir, bname))

    avg_time = np.mean(time_records[3:]) if len(time_records) > 3 else np.mean(time_records)
    print(f"Average inference time: {avg_time:.4f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', '-i', type=str, default='preset/test_datasets')
    parser.add_argument('--output_dir', '-o', type=str, default='experiments/test')
    parser.add_argument("--pretrained_model_path", type=str, default='preset/models/stable-diffusion-2-1-base')
    parser.add_argument('--pretrained_path', type=str, default='preset/models/qusr.pkl')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--process_size", type=int, default=512)
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--align_method", type=str, choices=['wavelet', 'adain', 'nofix'], default="adain")
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=224)
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024)
    parser.add_argument("--latent_tiled_size", type=int, default=96)
    parser.add_argument("--latent_tiled_overlap", type=int, default=32)
    parser.add_argument("--mixed_precision", type=str, default="fp16")
    parser.add_argument("--default", action="store_true")
    parser.add_argument("--quality_prompt_path", type=str, default="preset/test_lowlevel_prompt_q_RealSR")
    parser.add_argument("--enable_uncertainty", action="store_true", help="Enable UNG module (use with checkpoint trained with --enable_uncertainty)")

    args = parser.parse_args()
    seed_everything(args.seed)
    run_qusr(args)
