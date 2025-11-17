#!/bin/bash

export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:/usr/local/cuda-12.2/extras/CUPTI/lib64:$LD_LIBRARY_PATH
export HF_ENDPOINT=https://hf-mirror.com

# Define variables for multi-modal training
TASK="Eye_9tasks" #2, 7, 9
MODE="multimodal" #multimodal_madeleine or multimodal_chief or multimodal_feather  or multimodal
MODEL="our_clam_sb" #our_clam_sb or our_clam_sb_custom or our_focus
MODEL_NAME=${MODEL}

PX="512"
DIM="768"
if [ "$MODE" == "multimodal_madeleine" ]; then
    DIM="512"
fi
TASK_NAME="${TASK}_${MODE}"  # Multi-modal task name

EXP_CODE="${TASK_NAME}_${MODEL_NAME}_${PX}_1061_v1"
SPLIT_DIR="${TASK}_100"
DATA_ROOT_DIR="/data/zzh/WSI_zzh/ELSiP/Dataset"

LR="1e-4" # 2e-4 or 5e-6
ENHANCE="--enhance"
# ENHANCE=""

# Run the training script with GPU 3
CUDA_VISIBLE_DEVICES=3 python main.py \
    --drop_out 0.25 \
    --early_stopping \
    --lr $LR \
    --k 10 \
    --exp_code "$EXP_CODE" \
    --weighted_sample \
    --bag_loss ce \
    --inst_loss svm \
    --task "$TASK_NAME" \
    --model_type ${MODEL} \
    --log_data \
    --data_root_dir "$DATA_ROOT_DIR" \
    --split_dir "$SPLIT_DIR" \
    --embed_dim $DIM \
    $ENHANCE