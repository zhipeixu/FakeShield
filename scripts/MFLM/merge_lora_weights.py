import os
import torch
import argparse
from peft import get_peft_model
# from train import setup_tokenizer_and_special_tokens, initialize_model, prepare_model_for_training, setup_lora_config
import wandb
import os
import sys
import time
import tqdm
import random
import torch
import argparse
import deepspeed
import numpy as np
import shutil
from PIL import Image
import transformers
from functools import partial
from torch.utils.data import ConcatDataset
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter

from model.GLaMM import GLaMMForCausalLM
from model.llava import conversation as conversation_lib

from dataset.dataset import custom_collate_fn
from tools.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, AverageMeter, ProgressMeter, dict_to_cuda,
                         Summary, intersectionAndUnionGPU)
from dataset.Tamper_PS_Segm_ds import TamperSegmDataset
import transformers

def setup_lora_config(model, args):
    """ Configure LoRA settings for the model. """

    def find_proj_layers(model, target_modules):
        """ Identify projection layers in the model for LoRA adaptation. """
        linear_cls = torch.nn.Linear
        lora_module_names = set()
        for name, module in model.named_modules():
            if (isinstance(module, linear_cls) and all(
                    x not in name for x in ["grounding_encoder", "vision_tower", "mm_projector", "text_hidden_fcs"]
            ) and any(x in name for x in target_modules)):
                lora_module_names.add(name)
        return sorted(list(lora_module_names))

    # Extracting LoRA target modules
    lora_target_modules = args.lora_target_modules.split(",")
    lora_module_names = find_proj_layers(model, lora_target_modules)

    # Configuring LoRA
    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=lora_module_names, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM"
    )
    return lora_config



def prepare_model_for_training(model, tokenizer, args):
    # Enable input gradients
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    # Initialize vision tower，初始化CLIP
    print(
        '\033[92m' + "---- Initialized Global Image Encoder (vision tower) from: {} ----".format(
            args.vision_tower
        ) + '\033[0m'
    )
    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch.bfloat16, device=args.local_rank)

    # Initialize GLaMM model and adjust requires_grad
    if not args.pretrained:
        model.get_model().initialize_glamm_model(model.get_model().config)
    else:   #如果是微调模式
        #对于grounding_encoder和mask_decoder，设置requires_grad为False，不微调，也就是不微调sam
        for param in model.get_model().grounding_encoder.parameters():
            param.requires_grad = False
        #如果开了train_mask_decoder，那就再把mask_decoder的requires_grad设置为True
        if model.get_model().config.train_mask_decoder:
            model.get_model().grounding_encoder.mask_decoder.train()
            for param in model.get_model().grounding_encoder.mask_decoder.parameters():
                param.requires_grad = True

        # Projection layer 设置第一个映射层
        model.get_model().text_hidden_fcs.train()
        for param in model.get_model().text_hidden_fcs.parameters():
            param.requires_grad = True

    # Set requires_grad for vision tower and mm projector 
    #为CLIP和映射层设置requires_grad为False，不微调
    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False

    # Set requires_grad based on LoRA training
    lora_r = args.lora_r
    if lora_r == 0:
        for p in model.get_model().layers.parameters():
            p.requires_grad = True
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True

    # Configure conversation library
    conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_type]

    # Configure LoRA if applicable
    if lora_r > 0:
        #启动lora训练
        lora_config = setup_lora_config(model, args)
        model = get_peft_model(model, lora_config)

    # Resize token embeddings
    model.resize_token_embeddings(len(tokenizer))

    # Make certain modules trainable
    set_trainable_modules(model)

def initialize_model(args, tokenizer):
    """ Initialize the GLaMM model. """

    # 从 args 对象中提取多个参数，并存储在 model_args 字典中。这里提取的参数包括与模型训练和配置相关的各种设置。
    model_args = {k: getattr(args, k) for k in
                  ["train_mask_decoder", "out_dim", "ce_loss_weight", "dice_loss_weight", "bce_loss_weight",
                   "seg_token_idx", "vision_pretrained", "vision_tower", "use_mm_start_end", "mm_vision_select_layer",
                   "pretrain_mm_mlp_adapter", "tune_mm_mlp_adapter", "freeze_mm_mlp_adapter", "mm_use_im_start_end",
                   "with_region", "bbox_token_idx", "eop_token_idx", "bop_token_idx"]}
    # 向 model_args 字典中添加一个名为 num_level_reg_features 的参数，并将其值设为 4。这个其实就是区域特征里面的特征金字塔层数
    model_args["num_level_reg_features"] = 4

    #创建模型
    model = GLaMMForCausalLM.from_pretrained(
        args.version, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, **model_args
    )
    print('\033[92m' + "---- Initialized model from: {} ----".format(args.version) + '\033[0m')

    # Configure model tokens 配置模型的特殊 token ID，包括结束 token (eos_token_id)、开始 token (bos_token_id) 和填充 token (pad_token_id)
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    return model


def setup_tokenizer_and_special_tokens(args):
    """ Load tokenizer and add special tokens. """
    # 从预训练模型中加载一个 AutoTokenizer 实例，使用的模型版本和最大长度由 args.version 和 args.model_max_length 指定。padding_side="right" 指定填充在右边
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version, model_max_length=args.model_max_length, padding_side="right", use_fast=False
    )
    print('\033[92m' + "---- Initialized tokenizer from: {} ----".format(args.version) + '\033[0m')
    tokenizer.pad_token = tokenizer.unk_token  #设置 pad_token 为 unk_token，即未识别 token。

    if not args.pretrained:
        #没有pretrained的情况下，才会进入这里
        if args.use_mm_start_end:
            tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
            )
        #定义三种特殊的token
        # modifications specific for regions 针对具体区域的修改
        reg_tokens = ['<bbox>', '<point>'] #定义区域标记 token 列表，包括 '<bbox>' 和 '<point>'。
        # Adding special tokens for pixel grounding 添加pixel grounding的特殊token
        segmentation_tokens = ['[SEG]'] #定义分割标记 token 列表，包括 '[SEG]'。
        # Adding tokens for GCG
        phrase_tokens = ['<p>', '</p>'] #定义短语标记 token 列表，包括 '<p>' 和 '</p>'。用于GCG任务
        special_tokens = reg_tokens + segmentation_tokens + phrase_tokens
        tokenizer.add_tokens(special_tokens, special_tokens=True)

    #将每个特殊 token 编码为输入 ID，并存储在 args 对应的属性中。
    args.bbox_token_idx = tokenizer("<bbox>", add_special_tokens=False).input_ids[0]
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    args.bop_token_idx = tokenizer("<p>", add_special_tokens=False).input_ids[0]
    args.eop_token_idx = tokenizer("</p>", add_special_tokens=False).input_ids[0]

    return tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="GLaMM: Merge lora weights and save model in hf format")

    parser.add_argument("--version", help='Path to the base model.')
    parser.add_argument("--vision_pretrained", type=str)
    parser.add_argument("--weight", type=str, help="Path to the .bin model "
                                                                  "(generated using the script zero_to_fp32.py)")
    parser.add_argument("--save_path", type=str, help="Path to save the hf model.")

    # Model-specific settings
    parser.add_argument("--vision-tower", default="openai/clip-vit-large-patch14-336", type=str)
    parser.add_argument("--conv_type", default="llava_v1", type=str, choices=["llava_v1", "llava_llama_2"])
    parser.add_argument("--tune_mm_mlp_adapter", action="store_true", default=False)
    parser.add_argument("--mm_use_im_start_end", action="store_true", default=True)
    parser.add_argument("--freeze_mm_mlp_adapter", action="store_true", default=False)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=1536, type=int)
    parser.add_argument("--with_region", action="store_true", default=True)
    parser.add_argument("--mm_vision_select_layer", default=-2, type=int)
    parser.add_argument("--pretrain_mm_mlp_adapter", default="", type=str)
    parser.add_argument("--pretrained", action="store_true", default=True)
    # Training settings
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=0.5, type=float)
    parser.add_argument("--bce_loss_weight", default=2.0, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")

    return parser.parse_args()


def main():
    args = parse_args()

    # Create output directory if not exists already
    os.makedirs(args.save_path, exist_ok=True)

    # Initialize the tokenizer and model
    tokenizer = setup_tokenizer_and_special_tokens(args)
    model = initialize_model(args, tokenizer)
    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch.bfloat16)
    model.get_model().initialize_glamm_model(model.get_model().config)
    lora_r = args.lora_r
    if lora_r > 0:
        lora_config = setup_lora_config(model, args)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    model.resize_token_embeddings(len(tokenizer))

    # Load the state-dict from --weights
    state_dict = torch.load(args.weight, map_location="cpu")
    updated_state_dict = {}
    for key in state_dict.keys():
        updated_key = f"base_model.model.{key}"
        updated_state_dict[updated_key] = state_dict[key]
    model.load_state_dict(updated_state_dict, strict=True)

    # Merge and save
    model = model.merge_and_unload()
    state_dict = {}
    for k, v in model.state_dict().items():
        if "vision_tower" not in k:
            state_dict[k] = v
    model.save_pretrained(args.save_path, state_dict=state_dict)
    tokenizer.save_pretrained(args.save_path)


if __name__ == "__main__":
    main()