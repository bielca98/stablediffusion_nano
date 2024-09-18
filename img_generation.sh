#!/bin/bash

# Usage:
# ./img_generation.sh 3 "generated_images_100_noaug/DMSO" "generated_images_100_noaug/latrunculin_B_high_conc" "checkpoints_jeanzay/svdiff_100_split1" "lora" 200 1600
# To compute fid: metrics_dict=torch_fidelity.calculate_metrics(input1=path1,input2=path2,fid=True,cuda=True)

export OUTPUT_SUBFOLDER1=${2:-"generated_images_100_noaug/DMSO"}
export OUTPUT_SUBFOLDER2=${3:-"generated_images_100_noaug/latrunculin_B_high_conc"}
export WEIGHTS_SUBFOLDER=${4:-"2classes_nano_attention_DMSO_latrunculin_B_high_conc"}
export METHOD=${5:-"lora"}
export BATCH_SIZE=${6:-64}
export NUM_IMG_PER_CLASS=${7:-64}

MODEL_NAMES=("bguisard/stable-diffusion-nano-2-1" "stabilityai/stable-diffusion-2-1")
MODEL_INDEX=${8:-0} 
export MODEL_NAME=${MODEL_NAMES[$MODEL_INDEX]}

export EXPERIMENT_NAME=$(basename $(dirname $WEIGHTS_PATH))

BASE_OUTPUT_DIR="/projects/static2dynamic/Biel/stablediffusion_nano/outputs/"
export OUTPUT_DIR1="${BASE_OUTPUT_DIR}${OUTPUT_SUBFOLDER1}"
export OUTPUT_DIR2="${BASE_OUTPUT_DIR}${OUTPUT_SUBFOLDER2}"


BASE_WEIGHTS_DIR="/projects/static2dynamic/Biel/stablediffusion_nano/test_output/"
export WEIGHTS_PATH="${BASE_WEIGHTS_DIR}${WEIGHTS_SUBFOLDER}"

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
$CMD scripts/img_generation.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --weights_path=$WEIGHTS_PATH \
  --output_dir $OUTPUT_DIR1 $OUTPUT_DIR2\
  --experiment_name=$EXPERIMENT_NAME \
  --batch_size=$BATCH_SIZE \
  --num_images_per_class=$NUM_IMG_PER_CLASS \
  --finetunning_method=$METHOD