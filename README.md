# QUSR: Quality-Aware Prior with Uncertainty-Guided Noise Generation for Real-World Super-Resolution

## Experimental Results

Benchmark results on real-world super-resolution datasets (4× upscaling):

| Dataset | PSNR↑ | SSIM↑ | LPIPS↓ | DISTS↓ | CLIPIQA↑ | NIQE↓ | MUSIQ↑ | MANIQA↑ | FID↓ |
|---------|-------|-------|--------|--------|----------|-------|--------|---------|------|
| **RealSR** (100 images) | 25.54 | 0.729 | 0.297 | 0.220 | 0.682 | 5.66 | 69.17 | 0.656 | 125.27 |
| **DRealSR** (93 images) | 29.81 | 0.820 | 0.271 | 0.211 | 0.708 | 6.40 | 67.00 | 0.642 | 113.87 |

Full evaluation logs: [`logs/test_METRICS_real.log`](logs/test_METRICS_real.log), [`logs/test_METRICS_dreal.log`](logs/test_METRICS_dreal.log)

Visual results: [`experiments/test1w5_ur_real_f`](experiments/test1w5_ur_real_f) (RealSR), [`experiments/test1w5_ur_dreal_f`](experiments/test1w5_ur_dreal_f) (DRealSR)

---

## Installation

```bash
git clone https://github.com/oTvTog/QUSR
cd QUSR

conda create -n qusr python=3.10
conda activate qusr
pip install -r requirements.txt
```

## Dependencies

- PyTorch 2.0+
- diffusers 0.25.0
- transformers 4.28+
- peft 0.9.0
- xformers (optional, for memory efficiency)

## Quick Start

### 1. Download Pretrained Models

- **Stable Diffusion 2.1 Base**: [HuggingFace](https://huggingface.co/stabilityai/stable-diffusion-2-1-base)
- **Qwen2.5-VL-7B-Instruct** (for QAP generation): [HuggingFace](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- **RAM Model (Optional)**: [HuggingFace](https://huggingface.co/spaces/xinyu1205/recognize-anything/blob/main/ram_swin_large_14m.pth)
- **QUSR checkpoint**: Place in `preset/models/` (e.g., `qusr.pkl`)

### 2. Prepare Quality Prompts (QAP)

For inference, generate quality descriptions for your test images using an MLLM. Place `.txt` files in `preset/test_lowlevel_prompt_q_<dataset>/` with filenames matching your images (e.g., `image001.txt`).

**Note**: The `preset/` folder already provides quality prompts for the RealSR and DRealSR test sets (`preset/test_lowlevel_prompt_q_RealSR` and `preset/test_lowlevel_prompt_q_DrealSR`), so you can run inference directly without generating them yourself.

### 3. Run Inference

```bash
python test_qusr.py \
  --pretrained_model_path preset/models/stable-diffusion-2-1-base \
  --pretrained_path preset/models/qusr.pkl \
  --input_image preset/test_datasets \
  --output_dir experiments/test \
  --quality_prompt_path preset/test_lowlevel_prompt_q_RealSR \
  --enable_uncertainty \
  --default
```

## Training

### 1. Prepare Data

- GT image paths: `preset/gt_path.txt` (generate via `python scripts/get_path.py --folder /path/to/GT --output preset/gt_path.txt`)
- Pre-generated LR images: `preset/pre_generated_lr/`
- QAP prompts: `preset/lowlevel_prompt_q/` (generated via MLLM in `until_data/`)

**No RAM model required**—QUSR uses QAP (MLLM-generated quality prompts) instead.

### 2. Train

```bash
accelerate launch train_qusr.py \
  --pretrained_model_path preset/models/stable-diffusion-2-1-base \
  --pretrained_model_path_csd preset/models/stable-diffusion-2-1-base \
  --dataset_txt_paths preset/gt_path.txt \
  --highquality_dataset_txt_paths preset/gt_selected_path.txt \
  --dataset_test_folder preset/testfolder \
  --quality_prompt_path preset/lowlevel_prompt_q \
  --output_dir experiments/qusr \
  --train_batch_size 4 \
  --learning_rate 5e-5 \
  --lora_rank_unet_sem 4 \
  --timesteps1 1 \
  --lambda_l2 1.0 \
  --lambda_lpips 2.0 \
  --lambda_csd 1.0 \
  --enable_uncertainty \
  --lambda_uncertainty 0.1 \
  --enable_xformers_memory_efficient_attention \
  --checkpointing_steps 500
```

## Checkpoint Compatibility

**QUSR checkpoints use a standard format.** The saved keys (`state_dict_unet`, `state_dict_uncertainty`, etc.) are consistent across versions; only the code module names (QAP, UNG, UEM) were updated.

```bash
python test_qusr.py \
  --pretrained_path preset/models/qusr.pkl \
  --enable_uncertainty \
  --default
```

## Scripts

### Training

```bash
bash scripts/train/train_qusr.sh
```

### Testing

```bash
# Test on RealSR
bash scripts/test/test_qusr_real.sh [checkpoint_path] [output_dir]

# Test on DRealSR
bash scripts/test/test_qusr_dreal.sh [checkpoint_path] [output_dir]

# Evaluate metrics (PSNR, SSIM, LPIPS, FID, etc.)
python scripts/test/test_metric.py \
  --inp_imgs experiments/test_qusr_real \
  --gt_imgs preset/test_datasets/RealSR_test/test_HR \
  --log logs --log_name METRICS_real
```

### Data Preparation

```bash
# Generate preset/gt_path.txt from GT image folder
python scripts/get_path.py --folder /path/to/GT/images --output preset/gt_path.txt
```

## Project Structure

```
QUSR/
├── qusr.py           # QUSR model (QAP, UNG, UEM)
├── test_qusr.py      # Inference script
├── train_qusr.py     # Training script
├── scripts/
│   ├── train/        # train_qusr.sh
│   ├── test/         # test_qusr_real.sh, test_qusr_dreal.sh, test_metric.py
│   └── get_path.py   # Generate gt_path.txt
├── src/
│   ├── models/       # UNet, VAE
│   ├── datasets/     # Data loaders
│   └── my_utils/     # VAE tiling, color fix, etc.
├── preset/
│   ├── models/       # Pretrained weights
│   ├── test_datasets/
│   └── lowlevel_prompt_q*/
├── experiments/      # Visual results
│   ├── test1w5_ur_real_f/   # RealSR outputs (100 images)
│   └── test1w5_ur_dreal_f/  # DRealSR outputs (93 images)
├── logs/             # Evaluation metrics logs
└── until_data/       # QAP generation scripts (MLLM)
```

## License

Apache 2.0

## Citation

If you find QUSR useful, please cite our work.

## Acknowledgement

This project builds upon [PiSA-SR](https://github.com/csslc/PiSA-SR) and [OSEDiff](https://github.com/cswry/OSEDiff).
