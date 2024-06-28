#!/bin/bash

GPU_LIST=$1

./train.sh $GPU_LIST "train/DMSO_1_10" "from_scratch" 10
./train.sh $GPU_LIST "train/DMSO_2_10" "from_scratch" 10
./train.sh $GPU_LIST "train/DMSO_1_100" "from_scratch" 64
./train.sh $GPU_LIST "train/DMSO_2_100" "from_scratch" 64