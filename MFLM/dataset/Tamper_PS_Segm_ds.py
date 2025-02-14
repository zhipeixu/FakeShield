import os
import cv2
import random
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask
from transformers import CLIPImageProcessor
from model.llava import conversation as conversation_lib
from model.SAM.utils.transforms import ResizeLongestSide
from tools.utils import DEFAULT_IMAGE_TOKEN
from dataset.utils.utils import ANSWER_LIST, SEG_QUESTIONS, IML_TAMPER_QUESTION_LIST, LONG_QUESTION_LIST
import glob
import json


class TamperSegmDataset(torch.utils.data.Dataset):
    CLASSES = ('object',)
    IMG_MEAN = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    IMG_STD = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    IMG_SIZE = 1024
    IGNORE_LABEL = 255

    def __init__(self, dataset_dir, tokenizer, global_image_encoder, epoch_samples=500 * 8 * 2 * 10,
                 precision: str = "fp32", image_size: int = 224, num_classes_per_sample: int = 3,
                 tamper_segm_data="ps|CASIA1,CASIA2,CASIA2_AU", validation=False, split='train', val_prompt_json=None, gt_train=False, gt_val=False,
                 random_sampling=True, inference=False):
        self.epoch_samples = epoch_samples
        self.num_classes_per_sample = num_classes_per_sample

        self.dataset_dir = dataset_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.global_enc_processor = CLIPImageProcessor.from_pretrained(global_image_encoder)

        self.question_templates = SEG_QUESTIONS 
        self.answer_list = ANSWER_LIST 
        self.begin_str = f"""The {DEFAULT_IMAGE_TOKEN} provides an overview of the picture.\n"""
        self.long_question_list = LONG_QUESTION_LIST
        self.validation = validation
        self.split = split
        self.val_prompt_json = val_prompt_json
        self.random_sampling = random_sampling
        
        iml_det_seg_data, splits = tamper_segm_data.split("|")  
        splits = splits.split(",")
        base_image_dir = self.dataset_dir
        print("iml_det_seg_data: ", iml_det_seg_data)
        print("splits: ", splits)
        
        images = []
        masks = []

        for split in splits:
            images_split = sorted(glob.glob(
                os.path.join(
                    base_image_dir, iml_det_seg_data, split, "image" ,"*.jpg"
                )
            ))
            images.extend(images_split)
            masks_split = sorted(glob.glob(
                os.path.join(
                    base_image_dir, iml_det_seg_data, split, "mask" ,"*.png"
                )
            ))
            masks.extend(masks_split)

        self.iml_det_seg_data = (images, masks)
        self.images = images
        self.masks = masks

        print("number of reason_seg samples: ", len(images))

        EXPLANATORY_QUESTION_LIST = [
            "Based on the description of the tampered area, please give the mask of the tampered area.",
            "Please provide a mask of the tampered region based on the description of the tampered region given above.",
            "Please give the mask of the tampered area based on the description of the tampered area given above."
        ]

        self.explanatory_question_list = EXPLANATORY_QUESTION_LIST
        self.img_to_explanation = {}

        if validation == False:
            for split in splits:
                if gt_train:
                    print("gt_train")
                    with open(
                        os.path.join(
                            base_image_dir, iml_det_seg_data, split, "lisa_train_gt.json"
                        )
                    ) as f:
                        items = json.load(f)
                else:
                    print("not gt_train")
                    with open(
                        os.path.join(
                            base_image_dir, iml_det_seg_data, split, "lisa_train_output.json"
                        )
                    ) as f:
                        items = json.load(f)
                for item in items:
                    img_name = item["image"]
                    self.img_to_explanation[img_name] = {
                        "query": item["query"],
                        "outputs": item["outputs"],
                    }
            print("train: len(self.img_to_explanation): ", len(self.img_to_explanation))
        
        else:
            for split in splits:
                if gt_val:
                    print("gt_val")
                    with open(
                        os.path.join(
                            base_image_dir, iml_det_seg_data, split, "lisa_val_gt.json"
                        )
                    ) as f:
                        items = json.load(f)
                else:
                    print("not gt_val")
                    with open(
                        os.path.join(
                            base_image_dir, iml_det_seg_data, split, "lisa_val_output.json"
                        )
                    ) as f:
                        items = json.load(f)
                for item in items:
                    img_name = item["image"]
                    self.img_to_explanation[img_name] = {
                        "query": item["query"],
                        "outputs": item["outputs"],
                    }

            print("train: len(self.img_to_explanation): ", len(self.img_to_explanation))

    def __len__(self):
        return self.epoch_samples

    def _set_len(self, length):
        self.epoch_samples = length

    def grounding_enc_processor(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.IMG_MEAN) / self.IMG_STD
        h, w = x.shape[-2:]
        x = F.pad(x, (0, self.IMG_SIZE - w, 0, self.IMG_SIZE - h))
        return x

    def __getitem__(self, idx):
        images, masks = self.iml_det_seg_data
        image_path = images[idx]
        mask_path = masks[idx]
        print(image_path)
        print(mask_path)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        global_enc_img = self.global_enc_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        image = self.transform.apply_image(image)
        image_resize = image.shape[:2]
        grounding_enc_img = self.grounding_enc_processor(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255.

        sents = IML_TAMPER_QUESTION_LIST
        is_sentence = True
        if len(sents) >= self.num_classes_per_sample:
            sampled_inds = np.random.choice(
                list(range(len(sents))), size=self.num_classes_per_sample, replace=False
            )
        else:
            sampled_inds = list(range(len(sents)))
        sampled_sents = np.vectorize(sents.__getitem__)(sampled_inds).tolist()
        selected_labels = sampled_sents
        sampled_masks = [
            (mask == 1).astype(np.float32) for _ in range(len(sampled_inds))
        ]
        image_name = image_path.split("/")[-1]
        if image_name in self.img_to_explanation:
            choice = 1
        else:
            choice = 0
        questions = []
        answers = []
        for text in sampled_sents:
            question_template = random.choice(self.long_question_list)
            questions.append(question_template)
            # add explanation if applicable
            img_name = image_path.split("/")[-1]
            if img_name in self.img_to_explanation:
                if choice == 0:  
                    answers.append(random.choice(self.answer_list))
                elif choice == 1: 
                    image_name = image_path.split("/")[-1]
                    answer = self.img_to_explanation[image_name]["outputs"]
                    questions[-1] = (
                        'This photo has been tampered. The following is a description of the tampering area and the basis for judgment: \n' +
                        answer +
                        '\n' +
                        " {}".format(random.choice(self.explanatory_question_list))
                    )
                    answers.append(random.choice(self.answer_list))
                else:
                    raise ValueError("Not implemented yet.")
            else:
                answers.append(random.choice(self.answer_list))
            conversations = []
            conv = conversation_lib.default_conversation.copy()
            conv.messages = []
            for i, (question, answer) in enumerate(zip(questions, answers)):
                if i == 0:
                    question = self.begin_str + question
                conv.append_message(conv.roles[0], question)
                conv.append_message(conv.roles[1], answer)
            conversations.append(conv.get_prompt())
        image_name = image_path.split("/")[-1]
        masks = np.stack(sampled_masks, axis=0)
        masks = torch.from_numpy(masks)
        label = torch.ones(masks.shape[1], masks.shape[2]) * self.IGNORE_LABEL
        bboxes = None

        return (image_path, global_enc_img, grounding_enc_img, bboxes, conversations, masks, label,
                image_resize, questions, selected_labels)

