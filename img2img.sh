#!/bin/bash
# Script to perform img2img translation on an entire dataset and save the translated images. # This script allows for translation of a dataset using a pre-trained model with the option to compute FID metrics for image quality. 
# 
# Usage: 
# ./img2img.sh <num_processes> <data_subfolder> <class_label> <output_subfolder_original> <output_subfolder_translated> <weights_subfolder> <method> <batch_size> <gpu_id> 
# 
# Parameters: 
# 1. <num_processes>: Number of processes to run in parallel (e.g., 3). 
# 2. <data_subfolder>: Path to the data subfolder (e.g., "train/DMSO"). 
# 3. <class_label>: Label for the class to translate (e.g., 0 for class 0). 
# 4. <output_subfolder_original>: Folder where original images are stored (e.g., "BBBC021"). 
# 5. <output_subfolder_translated>: Folder where translated images will be saved (e.g., "BBBC021_translated"). 
# 6. <weights_subfolder>: Subfolder containing model weights (e.g., "2classes_nano_lora_attention_DMSO_latrunculin_B_high_conc"). 
# 7. <method>: Training method to use for generation (e.g., "lora").  Possible elections: "full", "lora", "svdiff", "svdiff_attention", "attention",  "lora_attention", "from_scratch"
# 8. <batch_size>: Batch size for processing images (e.g., 512). 
# 9. <gpu_id>: ID of the GPU to use (e.g., 0). 
# 
# Example Command: 
# ./img2img.sh 3 "train/DMSO" 0 "BBBC021" "BBBC021_translated" "2classes_nano_lora_attention_DMSO_latrunculin_B_high_conc" "lora" 512 0 # # To compute FID (Frechet Inception Distance) for image quality: # metrics_dict = torch_fidelity.calculate_metrics(input1=path1, input2=path2, fid=True, cuda=True)

export DATA_SUBFOLDER=${2:-"train/DMSO"}
export CLASS_LABEL=${3:-0}
export OUTPUT_SUBFOLDER_ORIGINAL=${4:-"BBBC021"}
export OUTPUT_SUBFOLDER_TRANSLATED=${5:-"BBBC021_translated"}
export WEIGHTS_SUBFOLDER=${6:-"2classes_nano_attention_DMSO_latrunculin_B_high_conc"}  aug_lora1_lora_DMSO_latrunculin_B_high_conc_100_43/checkpoint-20000
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