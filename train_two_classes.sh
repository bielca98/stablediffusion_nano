#!/bin/bash

# Script to train a model with two data classes using specified GPUs, data folders, and method. 
# The script allows for flexible GPU assignment, batch sizes, methods, and model selection. 
# 
# Usage: 
# ./train_two_classes.sh <gpu_list> <data_subfolder1> <data_subfolder2> <method> <batch_size> <data_samples> <data_sampling_seed> <model_index> 
# 
# Parameters: 
# 1. <gpu_list>: Comma-separated list of GPU IDs to use (e.g., "0", "0,1,2"). 
# 2. <data_subfolder1>: Path to the first data subfolder (e.g., "train/DMSO"). 
# 3. <data_subfolder2>: Path to the second data subfolder (e.g., "train/latrunculin_B_high_conc"). 
# 4. <method>: Training method to use for generation (e.g., "lora").  Possible elections: "full", "lora", "svdiff", "svdiff_attention", "attention",  "lora_attention", "from_scratch"
# 5. <batch_size>: Batch size for training (e.g., 128). 
# 6. <data_samples>: Number of data samples (e.g., 10). 
# 7. <data_sampling_seed>: Random seed for data sampling (e.g., 43). 
# 8. <model_index>: Index for selecting the model (0 for "nano", 1 for "stable-diffusion"). 
# 
# Example Commands: 
# ./train_two_classes.sh 0 "train/DMSO" "train/latrunculin_B_high_conc" "attention" 64 10 43 0 
# ./train_two_classes.sh 0,1,2,3 "ph3_imr_vs_rsts/IMR" "ph3_imr_vs_rsts/RSTS" "lora" 20 55 43 0 
# ./train_two_classes.sh 0,1,2,3 "org_c2/train/IMR" "org_c2/train/RSTS" "lora" 20 208 43 0 

export DATA_SUBFOLDER1=${2:-"train/DMSO"}
export DATA_SUBFOLDER2=${3:-"train/latrunculin_B_high_conc"}
export METHOD=${4:-"attention"}
export BATCH_SIZE=${5:-64}
export DATA_SAMPLES=${6:-10}
export DATA_SAMPLING_SEED=${7:-43}

MODEL_NAMES=("bguisard/stable-diffusion-nano-2-1" "stabilityai/stable-diffusion-2-1")
MODEL_INDEX=${8:-0} 
export MODEL_NAME=${MODEL_NAMES[$MODEL_INDEX]}

# To remove intermediate folders
BASE_FOLDER_NAME1=$(basename $DATA_SUBFOLDER1)
BASE_FOLDER_NAME2=$(basename $DATA_SUBFOLDER2)

MODEL_TYPE=$(echo $MODEL_NAME | cut -d'/' -f2 | cut -d'-' -f3)
export EXPERIMENT_NAME="golgi_${METHOD}_${DATA_SUBFOLDER1}_${DATA_SUBFOLDER2}_${DATA_SAMPLES}_${DATA_SAMPLING_SEED}"

BASE_DATA_DIR="/projects/static2dynamic/Biel/stablediffusion_nano/data/data/"
export DATA_DIR1="${BASE_DATA_DIR}${DATA_SUBFOLDER1}"
export DATA_DIR2="${BASE_DATA_DIR}${DATA_SUBFOLDER2}"

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
BASE_CMD="accelerate launch --main_process_port 24588"

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
  --data_dir $DATA_DIR1 $DATA_DIR2 \
  --output_dir=$OUTPUT_DIR \
  --resolution=128 \
  --train_batch_size=$BATCH_SIZE \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-3 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=0 \
  --report_to="wandb" \
  --validation_epochs=1000 \
  --num_validation_images=32 \
  --num_inference_steps=100 \
  --experiment_name=$EXPERIMENT_NAME \
  --validation_batch_size=200 \
  --data_sampling_seed=$DATA_SAMPLING_SEED \
  --data_samples=$DATA_SAMPLES \
  --max_train_steps=20000 \
  --finetunning_method=$METHOD
