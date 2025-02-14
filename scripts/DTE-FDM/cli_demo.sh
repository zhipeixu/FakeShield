pip install transformers==4.37.2

CUDA_VISIBLE_DEVICES=3 \
python -m llava.serve.cli \
    --model-path  ./weight/DTE-FDM \
    --DTG-path ./weight/DTG.pth \
