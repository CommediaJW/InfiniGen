#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_pred.py \
    --model_name longchat-7b-v1.5-32k \
    --model_name_or_path  /nfs/shared_LLM_model/lmsys/longchat-7b-v1.5-32k \
    --model_maxlen 31500 \
    --dataset_path /nfs/shared_LLM_dataset/LongBench/ \
    --dataset_name longbench \
    --output_dir ./preds/pred_longbench_longchat-7b-v1.5-32k_0 \
    --partial_weight_path ../setup/weights/longchat-7b-v1.5-32k_0.2 \
    --skewing_matrix_path ../setup/skewing_matrix/longchat-7b-v1.5-32k.pt \
    --partial_weight_ratio 0.2 --alpha 4 --capacity 1.0 --budget 0.2 \
    --write_in_time --mp_num 1 --e --min_seq_len 0
