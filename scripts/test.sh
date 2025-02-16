WEIGHT_PATH=./weight/fakeshield-v1-22b
QUESTION_PATH=./playground/test.jsonl
DTE_FDM_OUTPUT=./playground/DTE-FDM_output.jsonl
MFLM_OUTPUT=./playground/MFLM_output

pip install -q transformers==4.37.2  > /dev/null 2>&1
CUDA_VISIBLE_DEVICES=0 \
python ./DTE-FDM/llava/eval/model_vqa.py \
    --model-path ${WEIGHT_PATH}/DTE-FDM  \
    --DTG-path ${WEIGHT_PATH}/DTG.pth \
    --question-file ${QUESTION_PATH} \
    --image-folder / \
    --answers-file ${DTE_FDM_OUTPUT} \

pip install -q transformers==4.28.0  > /dev/null 2>&1
CUDA_VISIBLE_DEVICES=0 \
python ./MFLM/test.py \
    --version ${WEIGHT_PATH}/MFLM \
    --DTE-FDM-output ${DTE_FDM_OUTPUT} \
    --MFLM-output ${MFLM_OUTPUT}

