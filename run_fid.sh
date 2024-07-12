#!/bin/bash

export MODEL_NAME="bguisard/stable-diffusion-nano-2-1"
#export MODEL_NAME="stabilityai/stable-diffusion-2-1"
export OUTPUT_DIR="/projects/static2dynamic/Biel/stablediffusion_nano/outputs/delete_notext"
export WEIGHTS_PATH="/projects/static2dynamic/Biel/stablediffusion_nano/test_output/nano_lora_DMSO/checkpoint-190"
export METHOD="lora"

# Check if GPU IDs are provided
if [ "$#" -eq 0 ]; then
  GPU_IDS="all"
else
  GPU_IDS=$1
fi

# Determine the number of GPUs based on the input
if [ "$GPU_IDS" = "all" ]; then
  NUM_PROCESSES=0
else
  NUM_PROCESSES=$(echo $GPU_IDS | tr -cd ',' | wc -c)
  NUM_PROCESSES=$((NUM_PROCESSES + 1))
fi

# Construct the base accelerate launch command
BASE_CMD="accelerate launch --main_process_port 24591"

# Add multi_gpu option based on the number of GPUs
if [ $NUM_PROCESSES -eq 0 ]; then
  echo "Running on all GPU"
  CMD="$BASE_CMD --multi_gpu "
elif [ $NUM_PROCESSES -eq 1 ]; then
  echo "Running on a single GPU: $GPU_IDS"
  CMD="$BASE_CMD --gpu_ids $GPU_IDS "
else
  echo "Running on multiple GPUs: $GPU_IDS"
  CMD="$BASE_CMD --gpu_ids $GPU_IDS --num_processes $NUM_PROCESSES --multi_gpu "
fi

# Execute the command with the common options
$CMD scripts/fid_test.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --weights_path=$WEIGHTS_PATH \
  --output_dir=$OUTPUT_DIR \
  --finetunning_method=$METHOD \
  --num_images_per_class=20 