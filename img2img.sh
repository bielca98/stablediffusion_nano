#!/bin/bash

# Usage:
# ./img2img.sh 3 "train/DMSO" 0 "BBBC021" "BBBC021_translated" "2classes_nano_lora_attention_DMSO_latrunculin_B_high_conc" "lora" 512 0
# Compute fid: metrics_dict=torch_fidelity.calculate_metrics(input1=path1,input2=path2,fid=True,cuda=True)

export DATA_SUBFOLDER=${2:-"train/DMSO"}
export CLASS_LABEL=${3:-0}
export OUTPUT_SUBFOLDER_ORIGINAL=${4:-"BBBC021"}
export OUTPUT_SUBFOLDER_TRANSLATED=${5:-"BBBC021_translated"}
export WEIGHTS_SUBFOLDER=${6:-"2classes_nano_attention_DMSO_latrunculin_B_high_conc"}
export METHOD=${7:-"lora"}
export BATCH_SIZE=${8:-64}

MODEL_NAMES=("bguisard/stable-diffusion-nano-2-1" "stabilityai/stable-diffusion-2-1")
MODEL_INDEX=${9:-0} 
export MODEL_NAME=${MODEL_NAMES[$MODEL_INDEX]}

export EXPERIMENT_NAME=$(basename $(dirname $WEIGHTS_PATH))

BASE_DATA_DIR="/projects/static2dynamic/Biel/stablediffusion_nano/data/data/"
export DATA_DIR="${BASE_DATA_DIR}${DATA_SUBFOLDER}"

BASE_OUTPUT_DIR="/projects/static2dynamic/Biel/stablediffusion_nano/outputs/"
export OUTPUT_DIR1="${BASE_OUTPUT_DIR}${OUTPUT_SUBFOLDER_ORIGINAL}"
export OUTPUT_DIR2="${BASE_OUTPUT_DIR}${OUTPUT_SUBFOLDER_TRANSLATED}"


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
$CMD scripts/img2img.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --weights_path=$WEIGHTS_PATH \
  --output_dir $OUTPUT_DIR1 $OUTPUT_DIR2\
  --data_dir=$DATA_DIR \
  --experiment_name=$EXPERIMENT_NAME \
  --class_label=$CLASS_LABEL \
  --num_images_per_class=$BATCH_SIZE \
  --finetunning_method=$METHOD