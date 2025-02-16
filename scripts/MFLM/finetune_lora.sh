#!/bin/sh

pip install transformers==4.28.0  > /dev/null 2>&1

export PATH=/usr/local/cuda/bin:$PATH
export MASTER_PORT=$(shuf -i 2000-65000 -n 1)

export WEIGHT_PATH=./weight/fakeshield-v1-22b/MFLM
export OUTPUT_DIR=./playground/MFLM_train_result
export DATA_PATH=./dataset
export TRAIN_DATA_CHOICE=ps|CASIA2
export VAL_DATA_CHOICE=ps|CASIA1+

 
deepspeed --include localhost:0,1,2,3 --master_port $MASTER_PORT ./MFLM/train_ft.py \
  --version $WEIGHT_PATH \
  --DATA_PATH $DATA_PATH \
  --vision_pretrained ./weight/sam_vit_h_4b8939.pth \
  --vision-tower openai/clip-vit-large-patch14-336 \
  --exp_name $OUTPUT_DIR \
  --lora_r 8 \
  --lora_alpha 16 \
  --lr 3e-4 \
  --batch_size 6 \
  --pretrained \
  --use_segm_data \
  --tamper_segm_data "ps|CASIA2" \
  --val_tamper_dataset "ps|CASIA1+" \
  --epochs 200 \
  --mask_validation \
  --wandb   \
  --auto_resume \
