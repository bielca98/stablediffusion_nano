#!/bin/bash

GPU_LIST=$1
MODEL_TYPE=$2
BATCH_SIZE=$3

./train_two_classes.sh 0 "train/DMSO" "train/latrunculin_B_high_conc" "$MODEL_TYPE" $BATCH_SIZE
./train_two_classes.sh 0 "train/DMSO_1_100" "train/latrunculin_B_high_conc_100" "$MODEL_TYPE" $BATCH_SIZE
./train_two_classes.sh 0 "train/DMSO_1_10" "train/latrunculin_B_high_conc_10" "$MODEL_TYPE" $BATCH_SIZE