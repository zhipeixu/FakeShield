pip install transformers==4.37.2

CUDA_VISIBLE_DEVICES=1 \
python ./DTE-FDM/llava/eval/model_vqa.py \
    --model-path ./weight/DTE-FDM  \
    --DTG-path ./weight/DTG.pth \
    --question-file ./playground/DTE-FDM_eval_questions.jsonl \
    --image-folder / \
    --answers-file ./playground/test_answers.jsonl \
