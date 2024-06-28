#!/bin/bash
# Usage:
# ./train.sh gpu_list (like 0,1,2 or 0) "data_subfolder" "method" batch_size (like 128) model_index (0 for nano, 1 for stable-diffusion) "prompt"
# ./train.sh 0 "train/DMSO" "attention" 64 0 ""

export DATA_SUBFOLDER=${2:-"train/DMSO"}
export METHOD=${3:-"attention"}
export BATCH_SIZE=${4:-64}

MODEL_NAMES=("bguisard/stable-diffusion-nano-2-1" "stabilityai/stable-diffusion-2-1")
MODEL_INDEX=${5:-0} 
export MODEL_NAME=${MODEL_NAMES[$MODEL_INDEX]}

export PROMPT=${6:-""}

# To remove intermediate folders
BASE_FOLDER_NAME=$(basename $DATA_SUBFOLDER)

MODEL_TYPE=$(echo $MODEL_NAME | cut -d'/' -f2 | cut -d'-' -f3)
export EXPERIMENT_NAME="${MODEL_TYPE}_${METHOD}_${BASE_FOLDER_NAME}"

BASE_DATA_DIR="/projects/static2dynamic/Biel/stablediffusion_nano/data/data/"
export DATA_DIR="${BASE_DATA_DIR}${DATA_SUBFOLDER}"

BASE_OUTPUT_DIR="/projects/static2dynamic/Biel/stablediffusion_nano/test_output/"
export OUTPUT_DIR="${BASE_OUTPUT_DIR}${EXPERIMENT_NAME}"

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
$CMD scripts/accelerate_train.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --data_dir=$DATA_DIR\
  --prompt=$PROMPT \
  --output_dir=$OUTPUT_DIR \
  --resolution=128 \
  --train_batch_size=$BATCH_SIZE \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-3 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=0 \
  --report_to="wandb" \
  --validation_prompt=$PROMPT \
  --validation_epochs=1 \
  --num_validation_images=64 \
  --num_inference_steps=100 \
  --experiment_name=$EXPERIMENT_NAME \
  --num_train_epochs=200 \
  --finetunning_method=$METHOD