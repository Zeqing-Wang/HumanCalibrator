from tqdm import tqdm
from io import BytesIO
import requests
import json
import random
import sys
import re
from copy import deepcopy
import os
import cv2
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image,ImageDraw
import torch
import numpy as np
from pathlib import Path
from inference import cal_ratio, cal_ratio_size, get_prompt, rmv_duplicates, get_text_list, adjust
# 添加 GroundingDINO 和 Molding_human 模块的路径到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))

# 添加上一级目录的路径
sys.path.append(os.path.join(current_dir, '..', 'GroundingDINO'))
sys.path.append(os.path.join(current_dir, '..', 'MoldingHuman'))
from grounding_for_redundant import grounding
from infer_llava_ours_mask_only import find_word_in_string, natural_sort_key, inference, prepare_conv, load_model_llava, load_image
from load_args import parse_args

def gen_mask(args, pos, size):
    width, height = size
    pos = adjust(args, pos, width, height)
    mask_image = Image.new('RGB', (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(mask_image)
    draw.rectangle(pos, fill="white")
    return mask_image, pos

def inpainting(masked_image, mask, find_word, task, output_path, pipe):
    prompt, negative_prompt = get_prompt(find_word)
    inpainted_images = []
    inpainted_path = []
    if isinstance(masked_image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB and then to PIL
    else:
        image = masked_image.convert("RGB")  
    if isinstance(masked_image, np.ndarray):
        mask_image = Image.fromarray(mask).convert("L")
    else: 
        mask_image = mask.convert("L")

    original_size = image.size
    # Perform inpainting using the pipeline
    result_image = pipe(
        prompt=prompt, 
        negative_prompt=negative_prompt, 
        image=image, 
        mask_image=mask_image
    ).images[0]

    result_image = result_image.resize(original_size, Image.LANCZOS)
    temp_path = os.path.join(output_path, f'{task}_{find_word}.png')
    result_image.save(temp_path)
    return result_image, temp_path

def compare(image, part_dict, pos_list, path_list, output_path, i, text_list):
    label = part_dict['label']
    pos = part_dict['pos']
    for p, f in zip(pos_list, path_list):
        ratio = cal_ratio(pos, p)
        l = find_word_in_string(text_list, f.split('/')[-1])
        print(l, label, ratio)
        if ratio is not None and ratio > 0.4 and label == l:
            return True
    print(f'The original image appears to have a redundant {label} at {pos}.')
    mark_image = image.copy()

    draw_bbox = ImageDraw.Draw(mark_image)
    draw_bbox.rectangle(pos, outline='red', width=2)
    text_pos = (pos[0] + 2, pos[1])
    draw_bbox.text(text_pos, f'redundant {label}', fill="red")

    mark_path = os.path.join(output_path, f'marked_redundant_{label}_{i}.png')
    mark_image.save(mark_path)

    return False

def process_image(args, image_path, question, mode, pipe, text_list):
    # 创建输出路径 output_path
    img_folder = os.path.dirname(image_path)
    img_name = image_path.split('/')[-1]
    output_path = os.path.join(img_folder, f'redundant_res_{img_name}')
    os.makedirs(output_path, exist_ok = True)

    _, _, masked_path, _, all_masked_image_path, _, position_of_masks, _ = grounding(image_path, output_path)
    position_of_masks, masked_path = rmv_duplicates(position_of_masks, masked_path) 
    if all_masked_image_path is None:
        print(f'{image_path}:\nCould not ground anything.\n')
        return
    
    image = Image.open(image_path)
    results = []
    json_output_path = os.path.join(img_folder, "redundant_results.json")
    for i, pos in enumerate(position_of_masks):
        temp_phrase = find_word_in_string(text_list, masked_path[i].split('/')[-1])

        temp_dict = {
            'label' : temp_phrase,
            'pos' : pos,
            'condition' : 'normal',
            'path' : None
        }
        
        if temp_phrase is not None:
            ori_image = image.copy()
            mask, new_pos = gen_mask(args, pos, ori_image.size)
            temp_inpainted_image, temp_inpainted_path = inpainting(ori_image, mask, f'human {temp_phrase}', f'inpainted_mask_{i}', output_path, pipe) # pil
            new_grounding_path = os.path.join(output_path, f'{temp_phrase}_{i}')
            os.makedirs(new_grounding_path, exist_ok = True)

            temp_image = Image.open(temp_inpainted_path)
            _, _, masked_path_new, _, _, _, position_of_masks_new, _= grounding(temp_inpainted_path, new_grounding_path)
            position_of_masks_new, masked_path_new = rmv_duplicates(position_of_masks_new, masked_path_new)

            if not compare(image, temp_dict, position_of_masks_new, masked_path_new, output_path, i, text_list):
                temp_dict['condition'] = 'redundant'
                image = Image.open(image_path)
                final_image = image.copy()
                cropped_regen_image = temp_image.crop(new_pos)
                final_image.paste(cropped_regen_image, new_pos[:2])
                final_path = os.path.join(img_folder, f'handled_redundant_{i}.jpg')
                temp_dict['path'] = final_path
                final_image.save(final_path)
            results.append(temp_dict)
    
    with open(json_output_path, "w") as json_file:
        json.dump(results, json_file, indent=4)
    print(f'Results saved to {json_output_path}')
                


def redundant_sol(pipe):
    args = parse_args()
    text_list = get_text_list(args.text_list)
    # get arguments
    question = args.question
    mode = args.mode
    image_path = args.image_dir
    # 单个文件测试或批量测试
    image_dir = Path(args.image_dir)
    if image_dir.is_file():
        process_image(args, image_path, question, mode, pipe, text_list)
    else:
        for folder in image_dir.iterdir():
            for img_file in tqdm(folder.iterdir()):
                if img_file.is_file() and not img_file.name.startswith('handle') and img_file.suffix == '.jpg':
                    process_image(args, str(img_file), question, mode, pipe, text_list)


if __name__ == "__main__":
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "/mnt/SSD4_7T/keze/kz/model/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16,
    )
    pipe.to("cuda")
    redundant_sol(pipe)

    
