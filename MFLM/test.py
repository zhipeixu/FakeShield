import sys
import cv2
import random
import argparse
import gradio as gr
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPImageProcessor

from model.GLaMM import GLaMMForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.SAM.utils.transforms import ResizeLongestSide
from tools.generate_utils import center_crop, create_feathered_mask
from tools.utils import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from tools.markdown_utils import (markdown_default, examples, title, description, article, process_markdown, colors,
                                  draw_bbox, ImageSketcher)
import os
import json

def parse_args(args):
    parser = argparse.ArgumentParser(description="FakeShield Model Demo")
    parser.add_argument("--version", default="./weight/fakeshield-v1-22b/MFLM")
    parser.add_argument("--DTE-FDM-output", type=str)
    parser.add_argument("--MFLM-output", type=str)
    parser.add_argument("--precision", default='bf16', type=str)
    parser.add_argument("--image_size", default=1024, type=int, help="Image size for grounding image encoder")
    parser.add_argument("--model_max_length", default=1536, type=int)
    parser.add_argument("--vision-tower", default="openai/clip-vit-large-patch14-336", type=str)
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--conv_type", default="llava_v1", type=str, choices=["llava_v1", "llava_llama_2"])

    return parser.parse_args(args)

EXPLANATORY_QUESTION_LIST = [
    "Based on the description of the tampered area, please give the mask of the tampered area.",
    "Please provide a mask of the tampered region based on the description of the tampered region given above.",
    "Please give the mask of the tampered area based on the description of the tampered area given above."
]

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                image_path = record.get("image", "")
                output_text = record.get("outputs", "")
                data.append((image_path, output_text))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
    return data

def setup_tokenizer_and_special_tokens(args):
    """ Load tokenizer and add special tokens. """
    tokenizer = AutoTokenizer.from_pretrained(
        args.version, model_max_length=args.model_max_length, padding_side="right", use_fast=False
    )
    print('\033[92m' + "---- Initialized tokenizer from: {} ----".format(args.version) + '\033[0m')
    tokenizer.pad_token = tokenizer.unk_token
    args.bbox_token_idx = tokenizer("<bbox>", add_special_tokens=False).input_ids[0]
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    args.bop_token_idx = tokenizer("<p>", add_special_tokens=False).input_ids[0]
    args.eop_token_idx = tokenizer("</p>", add_special_tokens=False).input_ids[0]

    return tokenizer


def initialize_model(args, tokenizer):
    """ Initialize the GLaMM model. """
    model_args = {k: getattr(args, k) for k in
                  ["seg_token_idx", "bbox_token_idx", "eop_token_idx", "bop_token_idx"]}

    model = GLaMMForCausalLM.from_pretrained(
        args.version, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, **model_args)
    print('\033[92m' + "---- Initialized model from: {} ----".format(args.version) + '\033[0m')

    # Configure model tokens
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    return model


def prepare_model_for_inference(model, args):
    # Initialize vision tower
    print(
        '\033[92m' + "---- Initialized Global Image Encoder (vision tower) from: {} ----".format(
            args.vision_tower
        ) + '\033[0m'
    )
    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch.bfloat16, device=args.local_rank)
    model = model.bfloat16().cuda()
    return model


def grounding_enc_processor(x: torch.Tensor) -> torch.Tensor:
    IMG_MEAN = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    IMG_STD = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    IMG_SIZE = 1024
    x = (x - IMG_MEAN) / IMG_STD
    h, w = x.shape[-2:]
    x = F.pad(x, (0, IMG_SIZE - w, 0, IMG_SIZE - h))
    return x


def region_enc_processor(orig_size, post_size, bbox_img):
    orig_h, orig_w = orig_size
    post_h, post_w = post_size
    y_scale = post_h / orig_h
    x_scale = post_w / orig_w

    bboxes_scaled = [[bbox[0] * x_scale, bbox[1] * y_scale, bbox[2] * x_scale, bbox[3] * y_scale] for bbox in bbox_img]

    tensor_list = []
    for box_element in bboxes_scaled:
        ori_bboxes = np.array([box_element], dtype=np.float64)
        # Normalizing the bounding boxes
        norm_bboxes = ori_bboxes / np.array([post_w, post_h, post_w, post_h])
        # Converting to tensor, handling device and data type as in the original code
        tensor_list.append(torch.tensor(norm_bboxes, device='cuda').half().to(torch.bfloat16))

    if len(tensor_list) > 1:
        bboxes = torch.stack(tensor_list, dim=1)
        bboxes = [bboxes.squeeze()]
    else:
        bboxes = tensor_list
    return bboxes


def prepare_mask(pred_masks, text_output):
    if not pred_masks:
        return None, None
    
    seg_count = text_output.count("[SEG]")
    mask_list = [pred_masks[i].detach().cpu().numpy() for i in range(len(pred_masks))]
    mask_list = mask_list[-seg_count:]
    
    if not mask_list:
        return None, None
    
    final_mask = np.zeros_like(mask_list[0], dtype=np.uint8)
    for curr_mask in mask_list:
        final_mask[curr_mask > 0] = 255 
    
    if final_mask.ndim == 3:
        final_mask = final_mask.squeeze(0)
    
    seg_mask = Image.fromarray(final_mask, mode='L')
    
    
    return seg_mask, seg_mask


def inference(input_str, all_inputs, follow_up, generate):
    bbox_img = all_inputs['boxes']
    input_image = all_inputs['image']

    if not follow_up:
        conv = conversation_lib.conv_templates[args.conv_type].copy()
        conv.messages = []
        conv_history = {'user': [], 'model': []}
        conv_history["user"].append(input_str)

    input_str = input_str.replace('&lt;', '<').replace('&gt;', '>')
    prompt = input_str
    prompt = f"The {DEFAULT_IMAGE_TOKEN} provides an overview of the picture." + "\n" + prompt
    if args.use_mm_start_end:
        replace_token = (DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN)
        prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

    if not follow_up:
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], "")
    else:
        conv.append_message(conv.roles[0], input_str)
        conv.append_message(conv.roles[1], "")
    prompt = conv.get_prompt()

    image_np = cv2.imread(input_image)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image_np.shape[:2]
    original_size_list = [image_np.shape[:2]]

    # Prepare input for Global Image Encoder
    global_enc_image = global_enc_processor.preprocess(
        image_np, return_tensors="pt")["pixel_values"][0].unsqueeze(0).cuda()
    global_enc_image = global_enc_image.bfloat16()

    # Prepare input for Grounding Image Encoder
    image = transform.apply_image(image_np)
    resize_list = [image.shape[:2]]
    grounding_enc_image = (grounding_enc_processor(torch.from_numpy(image).permute(2, 0, 1).
                                                   contiguous()).unsqueeze(0).cuda())
    grounding_enc_image = grounding_enc_image.bfloat16()

    # Prepare input for Region Image Encoder
    post_h, post_w = global_enc_image.shape[1:3]
    bboxes = None
    if len(bbox_img) > 0:
        bboxes = region_enc_processor((orig_h, orig_w), (post_h, post_w), bbox_img)

    input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
    input_ids = input_ids.unsqueeze(0).cuda()

    # Pass prepared inputs to model
    output_ids, pred_masks = model.evaluate(
        global_enc_image, grounding_enc_image, input_ids, resize_list, original_size_list, max_tokens_new=512,
        bboxes=bboxes)
    output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]

    text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
    text_output = text_output.replace("\n", "").replace("  ", " ")
    text_output = text_output.split("ASSISTANT: ")[-1]

    # For multi-turn conversation
    conv.messages.pop()
    conv.append_message(conv.roles[1], text_output)
    conv_history["model"].append(text_output)
    color_history = []
    save_img = None
    if "[SEG]" in text_output:
        save_img, seg_mask = prepare_mask(pred_masks, text_output)

    output_str = text_output  # input_str

    return seg_mask, output_str


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])    
    print("======== MFLM Model Loading ========")

    tokenizer = setup_tokenizer_and_special_tokens(args)
    model = initialize_model(args, tokenizer)
    model = prepare_model_for_inference(model, args)
    global_enc_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    transform = ResizeLongestSide(args.image_size)
    model.eval()

    print("======== DTE_FDM Model Loaded ========")

    output_path = args.MFLM_output
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    print("======== MLFM Localization Begin ========")

    DTE_FDM_output = read_jsonl(args.DTE_FDM_output)
    for input_image, input_text in DTE_FDM_output:
        if "has not been tampered with" in input_text:
            print("The image has not been tampered with.")
            print("No mask generated.")
            continue

        input_text = (
            'This photo has been tampered. The following is a description of the tampering area and the basis for judgment: \n' +
            input_text +
            '\n' +
            " {}".format(random.choice(EXPLANATORY_QUESTION_LIST))
        )

        conv = None
        # Only to Display output
        conv_history = {'user': [], 'model': []}
        mask_path = None

        filename = os.path.basename(input_image)
        output_image, markdown_out = inference(input_text, {'image': input_image, 'boxes': []}, False, False)
        # output_image.show()
        save_path = os.path.join(output_path, filename)
        output_image.save(save_path)
        print("======== Mask saved to: ", save_path, " ========\n")

