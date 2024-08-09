#!/bin/bash
# Usage:
# ./train_two_classes.sh gpu_list (like 0,1,2 or 0) "data_subfolder1" "data_subfolder2" "method" batch_size (like 128) model_index (0 for nano, 1 for stable-diffusion)
# ./train_two_classes.sh 2 "train/DMSO" "train/latrunculin_B_high_conc" "attention" 64 10 43 0

export NUM_PROCESSES=${1:-2}
export DATA_SUBFOLDER1=${2:-"train/DMSO"}
export DATA_SUBFOLDER2=${3:-"train/latrunculin_B_high_conc"}
export METHOD=${4:-"attention"}
export BATCH_SIZE=${5:-64}
export DATA_SAMPLES=${6:-10}
export DATA_SAMPLING_SEED=${7:-43}
export MAIN_PROCESS_PORT=${8:-24591}

MODEL_NAMES=("/lustre/fsn1/projects/rech/arj/uni93xz/base_outputs/stable-diffusion-nano-2-1/local/checkpoints/" "/lustre/fsn1/projects/rech/arj/uni93xz/base_outputs/stable-diffusion-2-1")

MODEL_INDEX=${9:-0} 
export MODEL_NAME=${MODEL_NAMES[$MODEL_INDEX]}

# To remove intermediate folders
BASE_FOLDER_NAME1=$(basename $DATA_SUBFOLDER1)
BASE_FOLDER_NAME2=$(basename $DATA_SUBFOLDER2)

MODEL_TYPE=$(echo $MODEL_NAME | cut -d'/' -f2 | cut -d'-' -f3)
export EXPERIMENT_NAME="2classes_${MODEL_TYPE}_${METHOD}_${BASE_FOLDER_NAME1}_${BASE_FOLDER_NAME2}_${DATA_SAMPLES}_${DATA_SAMPLING_SEED}"

BASE_DATA_DIR="/gpfswork/rech/arj/uni93xz/stablediffusion_nano/data/data/"
export DATA_DIR1="${BASE_DATA_DIR}${DATA_SUBFOLDER1}"
export DATA_DIR2="${BASE_DATA_DIR}${DATA_SUBFOLDER2}"

BASE_OUTPUT_DIR="/lustre/fsn1/projects/rech/arj/uni93xz/svdiff_outputs/"
export OUTPUT_DIR="${BASE_OUTPUT_DIR}${EXPERIMENT_NAME}"


# Construct the base accelerate launch command
CMD="accelerate launch --main_process_port ${MAIN_PROCESS_PORT} --num_processes $NUM_PROCESSES --multi_gpu --num_machines 1 --mixed-precision no --dynamo_backend no"

# Execute the command with the common options
$CMD /gpfswork/rech/arj/uni93xz/stablediffusion_nano/scripts/accelerate_train.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --data_dir $DATA_DIR1 $DATA_DIR2 \
  --output_dir=$OUTPUT_DIR \
  --resolution=128 \
  --train_batch_size=$BATCH_SIZE \
  --checkpointing_steps=10000 \
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
  --max_train_steps=30000 \
  --use_local_checkpoints \
  --inception_weights_path="/lustre/fsn1/projects/rech/arj/uni93xz/base_outputs/inception/weights-inception-2015-12-05-6726825d.pth" \
  --finetunning_method=$METHOD



