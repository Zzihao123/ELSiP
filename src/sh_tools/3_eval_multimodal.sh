 #!/bin/bash

export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:/usr/local/cuda-12.2/extras/CUPTI/lib64:$LD_LIBRARY_PATH
export HF_ENDPOINT=https://hf-mirror.com

TASK="Eye_9tasks"
MODE="multimodal" 
MODEL="our_clam_sb" 
MODEL_NAME=${MODEL}

PX="512"
DIM="768"

TASK_NAME="${TASK}_${MODE}"  
EXP_CODE="${TASK_NAME}_${MODEL_NAME}_${PX}_1061_v1_s1"
SAVE_EXP_CODE="${EXP_CODE}_cv" 
SPLIT_DIR="splits/${TASK}_100"
DATA_ROOT_DIR="/data/zzh/WSI_zzh/ELSiP/Dataset"

ENHANCE="--enhance"
SPLIT="test"

CUDA_VISIBLE_DEVICES=0 python eval.py \
    --k 10 \
    --models_exp_code "$EXP_CODE" \
    --save_exp_code "$SAVE_EXP_CODE" \
    --task "$TASK_NAME" \
    --model_type ${MODEL} \
    --results_dir results \
    --data_root_dir "$DATA_ROOT_DIR" \
    --splits_dir "$SPLIT_DIR" \
    --embed_dim $DIM \
    --split $SPLIT \
    $ENHANCE

echo "Evaluation completed!"
echo "Results saved in: ${EXP_CODE}/" 