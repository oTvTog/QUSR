#!/bin/bash
# QUSR Test on RealSR dataset
# Run from QUSR root: bash scripts/test/test_qusr_real.sh
# Set PRETRAINED_PATH to your checkpoint, e.g. preset/models/qusr.pkl

cd "$(dirname "$0")/../.."
PRETRAINED_PATH="${1:-preset/models/qusr.pkl}"
OUTPUT_DIR="${2:-experiments/test_qusr_real}"
ENABLE_UNG="${3:-true}"

echo "=== QUSR Test on RealSR ==="
echo "Checkpoint: $PRETRAINED_PATH"
echo "Output: $OUTPUT_DIR"

EXTRA_ARGS=()
if [ "$ENABLE_UNG" = "true" ]; then
  EXTRA_ARGS=(--enable_uncertainty)
fi

python test_qusr.py \
  --pretrained_model_path ../preset/models/stable-diffusion-2-1-base \
  --pretrained_path "$PRETRAINED_PATH" \
  --input_image ../preset/test_datasets/RealSR_test/test_SR_bicubic \
  --output_dir "$OUTPUT_DIR" \
  --quality_prompt_path ../preset/test_lowlevel_prompt_q_RealSR \
  --process_size 512 \
  --upscale 4 \
  --default \
  "${EXTRA_ARGS[@]}"

echo "Run metrics: python scripts/test/test_metric.py --inp_imgs $OUTPUT_DIR --gt_imgs preset/test_datasets/RealSR_test/test_HR --log logs --log_name METRICS_real"
