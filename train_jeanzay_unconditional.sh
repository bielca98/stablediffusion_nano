#!/bin/bash
# Script to train a model using multiple processes and GPUs, with a specified data folder, method, and batch size. 
# 
# Usage: 
# ./train_two_classes.sh <num_processes> <data_subfolder> <method> <batch_size> <data_samples> 
# 
# Parameters: 
# 1. <num_processes>: Number of processes to run in parallel (e.g., 2). 
# 2. <data_subfolder>: Path to the data subfolder (e.g., "train/DMSO"). 
# 3. <method>: Training method to use for generation (e.g., "lora").  Possible elections: "full", "lora", "svdiff", "svdiff_attention", "attention",  "lora_attention", "from_scratch"
# 4. <batch_size>: Batch size for training (e.g., 128). 
# 5. <data_samples>: Number of data samples (e.g., 10). 
# 
# Example Command: 
# ./train_two_classes.sh 2 "train/DMSO" "attention" 64 10

export NUM_PROCESSES=${1:-2}
export DATA_SUBFOLDER=${2:-"train/DMSO"}
export METHOD=${3:-"attention"}
export BATCH_SIZE=${4:-64}
export DATA_SAMPLES=${5:-10}
export DATA_SAMPLING_SEED=${6:-43}
export MAIN_PROCESS_PORT=${7:-24591}
MODEL_NAMES=("/lustre/fsn1/projects/rech/arj/uni93xz/base_outputs/stable-diffusion-nano-2-1/local/checkpoints/" "/lustre/fsn1/projects/rech/arj/uni93xz/base_outputs/stable-diffusion-2-1")

MODEL_INDEX=${8:-0} 
export MODEL_NAME=${MODEL_NAMES[$MODEL_INDEX]}

# To remove intermediate folders
BASE_FOLDER_NAME=$(basename $DATA_SUBFOLDER)

MODEL_TYPE=$(echo $MODEL_NAME | cut -d'/' -f2 | cut -d'-' -f3)
export EXPERIMENT_NAME="unconditional_${MODEL_TYPE}_${METHOD}_${BASE_FOLDER_NAME}_${DATA_SAMPLES}_${DATA_SAMPLING_SEED}"

BASE_DATA_DIR="/gpfswork/rech/arj/uni93xz/stablediffusion_nano/data/data/"
export DATA_DIR="${BASE_DATA_DIR}${DATA_SUBFOLDER}"

BASE_OUTPUT_DIR="/lustre/fsn1/projects/rech/arj/uni93xz/svdiff_outputs/"
export OUTPUT_DIR="${BASE_OUTPUT_DIR}${EXPERIMENT_NAME}"


# Construct the base accelerate launch command
CMD="accelerate launch --main_process_port ${MAIN_PROCESS_PORT} --num_processes $NUM_PROCESSES --multi_gpu --num_machines 1 --mixed-precision no --dynamo_backend no"

# Execute the command with the common options
$CMD /gpfswork/rech/arj/uni93xz/stablediffusion_nano/scripts/accelerate_train.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --data_dir=$DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  --resolution=128 \
  --train_batch_size=$BATCH_SIZE \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-3 \
  --lr_scheduler="cosine" \
  --checkpointing_steps=10000 \
  --lr_warmup_steps=0 \
  --report_to="wandb" \
  --validation_epochs=1000 \
  --num_validation_images=32 \
  --num_inference_steps=100 \
  --experiment_name=$EXPERIMENT_NAME \
  --validation_batch_size=200 \
  --data_sampling_seed=$DATA_SAMPLING_SEED \
  --data_samples=$DATA_SAMPLES \
  --max_train_steps=30000 \
  --use_local_checkpoints \
  --upload_training_images \
  --inception_weights_path="/lustre/fsn1/projects/rech/arj/uni93xz/base_outputs/inception/weights-inception-2015-12-05-6726825d.pth" \
  --finetunning_method=$METHOD
