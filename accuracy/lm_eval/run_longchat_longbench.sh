#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_pred.py \
    --model_name Llama-2-7b-chat-hf \
    --model_name_or_path  /nfs/shared_LLM_model/meta-llama/Llama-2-7b-chat-hf \
    --model_maxlen 3500 \
    --dataset_path /nfs/shared_LLM_dataset/LongBench/ \
    --dataset_name longbench \
    --output_dir ./preds/pred_longbench_Llama-2-7b-chat-hf \
    --partial_weight_path ../setup/weights/Llama-2-7b-chat-hf_0.2 \
    --skewing_matrix_path ../setup/skewing_matrix/Llama-2-7b-chat-hf.pt \
    --partial_weight_ratio 0.2 --alpha 4 --capacity 1.0 --budget 0.2 \
    --write_in_time --mp_num 8 --e --min_seq_len 0
