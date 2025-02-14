#!/bin/sh

pip install transformers==4.28.0

export PATH=/usr/local/cuda/bin:$PATH
export MASTER_PORT=$(shuf -i 2000-65000 -n 1)

export CKPT_PATH="./weight/MFLM"
export OUTPUT_DIR_PATH=./playground/MFLM_test_result

deepspeed --include localhost:2,3,4,5 --master_port $MASTER_PORT ./MFLM/test.py \
  --version $CKPT_PATH \
  --dataset_dir path_to_your_dataset \
  --vision_pretrained ./weight/sam_vit_h_4b8939.pth \
  --vision-tower openai/clip-vit-large-patch14-336 \
  --pretrained \
  --val_dataset "TamperSegmVal" \
  --val_tamper_dataset "ps|CASIA1+" \
  --mask_validation \
  --eval_only \
  --eval_output_dir $OUTPUT_DIR_PATH \
