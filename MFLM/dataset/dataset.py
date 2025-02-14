import numpy as np
import torch

from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from tools.utils import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN


def custom_collate_fn(batch, tokenizer=None, use_mm_start_end=True, inference=False, local_rank=-1):
    # Initializing lists and counters
    image_path_list, global_enc_image_list, grounding_enc_image_list = [], [], []
    bboxes_list, conversation_list, masks_list = [], [], []
    label_list, resize_list, questions_list = [], [], []
    selected_labels_list, offset_list, inferences = [], [0], []
    cnt = 0

    # Iterating through the batch
    for (image_path, global_enc_image, grounding_enc_image, bboxes, conversations, masks, label, resize, questions,
         sampled_classes) in batch:
        image_path_list.append(image_path)
        global_enc_image_list.append(global_enc_image)
        grounding_enc_image_list.append(grounding_enc_image)
        bboxes_list.append(bboxes)
        conversation_list.extend(conversations)
        masks_list.append([] if masks is None else masks.float())
        label_list.append(label)
        resize_list.append(resize)
        questions_list.append(questions)
        selected_labels_list.append(sampled_classes)
        offset_list.append(cnt := cnt + len(conversations))
        inferences.append(inference)

    # Handling the conversation list
    if use_mm_start_end:
        replace_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        conversation_list = [conv.replace(DEFAULT_IMAGE_TOKEN, replace_token) for conv in conversation_list]

    # Tokenizing and padding input ids
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversation_list],
        batch_first=True, padding_value=tokenizer.pad_token_id
    )
    # print('input_ids111:', input_ids)
    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    # print('input_ids:', input_ids)
    # print('conversation_list:', conversation_list)


    # Preparing targets and handling conversation types
    conv = conversation_lib.default_conversation.copy()
    targets = input_ids.clone()
    # conv_type == "llava_v1"
    sep = conv.sep + conv.roles[1] + ": "
    sep2 = conv.sep2

    for conversation, target in zip(conversation_list, targets):
        _process_conversation(conversation, target, tokenizer, sep, sep2)

    # Adjusting for inferences
    if not inferences[0]:
        truncate_len = tokenizer.model_max_length - 575
        if input_ids.shape[1] > truncate_len:
            input_ids, targets, attention_masks = map(
                lambda x: x[:, :truncate_len], [input_ids, targets, attention_masks]
                )

    return {
        "image_paths": image_path_list,
        "global_enc_images": torch.stack(global_enc_image_list, dim=0),
        "grounding_enc_images": None if grounding_enc_image_list[0] is None else torch.stack(grounding_enc_image_list, dim=0),
        "bboxes": None if bboxes_list[0] is None else bboxes_list,
        "input_ids": input_ids,
        "labels": targets,
        "attention_masks": attention_masks,
        "masks_list": None if masks_list[0] is None else masks_list,
        "label_list": None if label_list[0] is None else label_list,
        "resize_list": None if resize_list[0] is None else resize_list,
        "offset": torch.LongTensor(offset_list),
        "questions_list": questions_list,
        "sampled_classes_list": selected_labels_list,
        "inference": inferences[0],
        "conversation_list": conversation_list,
    }


def _process_conversation(conversation, target, tokenizer, sep, sep2):
    total_len = target.ne(tokenizer.pad_token_id).sum().item()
    rounds = conversation.split(sep2)
    cur_len = 1
    target[:cur_len] = IGNORE_INDEX

    for rou in rounds:
        if not rou:
            break

        parts = rou.split(sep)
        assert len(parts) == 2, (len(parts), rou)
        parts[0] += sep

        if DEFAULT_IMAGE_TOKEN in conversation:
            round_len = len(tokenizer_image_token(rou, tokenizer))
            instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
        else:
            round_len = len(tokenizer(rou).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

        target[cur_len: cur_len + instruction_len] = IGNORE_INDEX
        cur_len += round_len

    target[cur_len:] = IGNORE_INDEX
    if cur_len < tokenizer.model_max_length:
        assert cur_len == total_len
