#!/bin/bash
# QUSR Training (with UNG)
# Run from QUSR root: bash scripts/train/train_qusr.sh

cd "$(dirname "$0")/../.."
LOG_DIR="experiments/qusr/logs"
mkdir -p $LOG_DIR
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/training_${TIMESTAMP}.txt"

echo "=== QUSR Training ===" | tee $LOG_FILE
echo "Started: $(date)" | tee -a $LOG_FILE

CUDA_VISIBLE_DEVICES="0,1,2,3" accelerate launch train_qusr.py \
  --pretrained_model_path="preset/models/stable-diffusion-2-1-base" \
  --pretrained_model_path_csd="preset/models/stable-diffusion-2-1-base" \
  --dataset_txt_paths="preset/gt_path.txt" \
  --highquality_dataset_txt_paths="preset/gt_selected_path.txt" \
  --dataset_test_folder="preset/testfolder" \
  --quality_prompt_path="preset/lowlevel_prompt_q" \
  --output_dir="experiments/qusr" \
  --learning_rate=3e-5 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=2 \
  --enable_xformers_memory_efficient_attention \
  --gradient_checkpointing \
  --checkpointing_steps=500 \
  --seed=123 \
  --max_train_steps=30000 \
  --timesteps1=1 \
  --lambda_l2=0.5 \
  --lambda_lpips=2.0 \
  --lambda_csd=2.0 \
  --lora_rank_unet_sem=4 \
  --null_text_ratio=0.0 \
  --align_method="adain" \
  --tracker_project_name="QUSR" \
  --enable_uncertainty \
  --uncertainty_hidden_channels=64 \
  --min_noise=0.1 \
  --un=0.3 \
  --kappa=1.0 \
  --lambda_uncertainty=0.3 \
  2>&1 | tee -a $LOG_FILE

echo "Completed: $(date)" | tee -a $LOG_FILE
