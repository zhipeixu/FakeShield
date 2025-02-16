#!/bin/bash
OUTPUT_DIR=./playground/DTE-FDM_train_result
DATA_PATH=path_to_your_train_data
WEIGHT_PATH=./weight/fakeshield-v1-22b/DTE-FDM

pip install transformers==4.37.2  > /dev/null 2>&1
mkdir -p $OUTPUT_DIR
deepspeed --include localhost:0,1,2,3  --master_port=29501  ./DTE-FDM/llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/DTE-FDM/zero3.json \
    --model_name_or_path $WEIGHT_PATH \
    --version v1 \
    --data_path $DATA_PATH \
    --image_folder / \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 6 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "epoch"\
    --save_steps 800 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
