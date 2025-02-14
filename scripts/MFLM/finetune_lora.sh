#!/bin/sh

pip install transformers==4.28.0

export PATH=/usr/local/cuda/bin:$PATH
export MASTER_PORT=$(shuf -i 2000-65000 -n 1)

export CKPT_PATH=./weight/MFLM
export OUTPUT_DIR_PATH=./playground/MFLM_train_result

deepspeed --include localhost:0,1,2,3 --master_port $MASTER_PORT ./MFLM/train_ft.py \
  --version $CKPT_PATH \
  --dataset_dir path_to_your_dataset \
  --vision_pretrained ./weight/sam_vit_h_4b8939.pth \
  --vision-tower openai/clip-vit-large-patch14-336 \
  --exp_name $OUTPUT_DIR_PATH \
  --lora_r 8 \
  --lora_alpha 16 \
  --lr 3e-4 \
  --batch_size 6 \
  --pretrained \
  --use_segm_data \
  --seg_dataset "Tamper_Segm" \
  --segm_sample_rates "1" \
  --tamper_segm_data "ps|CASIA2" \
  --val_dataset "TamperSegmVal" \
  --val_tamper_dataset "ps|CASIA1+" \
  --epochs 200 \
  --mask_validation \
  --wandb   \
  --auto_resume \
