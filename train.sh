#!/bin/bash
# ./train.sh  0,1,2

export MODEL_NAME="bguisard/stable-diffusion-nano-2-1"
#export MODEL_NAME="stabilityai/stable-diffusion-2-1"
export OUTPUT_DIR="/projects/static2dynamic/Biel/stablediffusion_nano/test_output/nano_two_classess"
export EXPERIMENT_NAME="nano_two_classess"
export DATA_DIR1="/projects/static2dynamic/Biel/stablediffusion_nano/data/data/train/DMSO_1"
export DATA_DIR2="/projects/static2dynamic/Biel/stablediffusion_nano/data/data/train/DMSO_2"
export PROMPT1="rkm"
export PROMPT2="kle"

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
  --data_dir $DATA_DIR1 $DATA_DIR2\
  --prompt $PROMPT1 $PROMPT2 \
  --output_dir=$OUTPUT_DIR \
  --resolution=128 \
  --train_batch_size=64 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-3 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=0 \
  --report_to="wandb" \
  --validation_prompt $PROMPT1 $PROMPT2 \
  --validation_epochs=1 \
  --num_validation_images=32 \
  --num_inference_steps=100 \
  --experiment_name=$EXPERIMENT_NAME \
  --num_train_epochs=200 \
  --finetunning_method="lora" 
