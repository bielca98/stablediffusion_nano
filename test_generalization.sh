#!/bin/bash

# ./test_generalization.sh gpu_list class_label num_train_images method data_subfolder dataloader_batch_size
#./test_generalization.sh 0 0 100 lora BBBC021_splits/split1_100/DMSO 100 aug_lora1_lora_DMSO_latrunculin_B_high_conc_100_43 aug_lora2_lora_DMSO_latrunculin_B_high_conc_100_43

export MODEL_NAME="bguisard/stable-diffusion-nano-2-1"

export CLASS_LABEL=${2:0}
export NUM_TRAIN_IMAGES=${3:-10}
export METHOD=${4:-"attention"}
export DATA_SUBFOLDER=${5:-"BBBC021_splits/split1_100/DMSO"}
export DATALOADER_BATCH_SIZE=${6:-10}
export WEIGHTS_SUBFOLDER1=${7:-"aug_lora1_lora_DMSO_latrunculin_B_high_conc_100_43"}
export WEIGHTS_SUBFOLDER2=${8:-"aug_lora2_lora_DMSO_latrunculin_B_high_conc_100_43"}

BASE_DATA_DIR="/projects/static2dynamic/Biel/stablediffusion_nano/data/data/"
export DATA_DIR="${BASE_DATA_DIR}${DATA_SUBFOLDER}"


BASE_WEIGHTS_PATH="/projects/static2dynamic/Biel/stablediffusion_nano/test_output/"
export WEIGHTS_PATH1="${BASE_WEIGHTS_PATH}${WEIGHTS_SUBFOLDER1}"
export WEIGHTS_PATH2="${BASE_WEIGHTS_PATH}${WEIGHTS_SUBFOLDER2}"
export EXPERIMENT_NAME="lora_100_1_aug"
# export EXPERIMENT_NAME="200_NOAUG_DMSO"

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
BASE_CMD="accelerate launch --main_process_port 24596"

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
$CMD scripts/test_generalization.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --weights_path=$WEIGHTS_PATH1 \
  --second_weights_path=$WEIGHTS_PATH2 \
  --data_dir=$DATA_DIR \
  --dataloader_batch_size=$DATALOADER_BATCH_SIZE \
  --experiment_name=$EXPERIMENT_NAME \
  --upload_images \
  --resolution=128 \
  --num_images_per_class=100 \
  --data_samples=$NUM_TRAIN_IMAGES \
  --n_close_images_to_upload=10 \
  --finetunning_method=$METHOD