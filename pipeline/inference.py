import argparse
from llava.constants import (IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN,
                             DEFAULT_IM_END_TOKEN, IMAGE_PLACEHOLDER, )
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
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
import argparse
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image,ImageDraw
import torch
import numpy as np
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..', 'MoldingHuman'))
from infer_llava_ours_mask_only import natural_sort_key, inference, prepare_conv, load_model_llava, load_image


def rmv_duplicates(pos_list, threshold):
    if len(pos_list) < 1:
        return pos_list

    labels = [l for l, p in pos_list]
    upper_left = [(p[0], p[1]) for l, p in pos_list]
    lower_right = [(p[2], p[3]) for l, p in pos_list]
    center = [get_center(tl, br) for tl, br in zip(upper_left, lower_right)]
    sorted_indices = sort_by_distance(center)

    remove_idx = set()
    pos = []

    for i in range(1, len(sorted_indices)):
        
        idx1, idx2 = sorted_indices[i-1], sorted_indices[i]
        ratio = cal_ratio(pos_list[idx1][1], pos_list[idx2][1])
        ratio_1 = cal_ratio(pos_list[idx2][1], pos_list[idx1][1])


        if ((ratio is not None and threshold < ratio < 1/threshold) or (ratio_1 is not None and threshold < ratio_1 < 1/threshold)) and pos_list[idx1][0] == pos_list[idx2][0]:
            remove_idx.add(idx1)
            # print(f'removing {pos_list[idx1]}')

    for idx in range(len(pos_list)):
        if idx not in remove_idx:
            pos.append(pos_list[idx])

    return pos
    
def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def sort_by_distance(centers):
    n = len(centers)
    visited = [False] * n
    sorted_centers = []
    
    # 从第一个点开始
    current_index = 0
    for _ in range(n):
        sorted_centers.append(current_index)
        visited[current_index] = True
        
        # 找到最近的未访问点
        next_index = None
        min_distance = float('inf')
        for i in range(n):
            if not visited[i]:
                distance = euclidean_distance(centers[current_index], centers[i])
                if distance < min_distance:
                    min_distance = distance
                    next_index = i
        
        current_index = next_index
    
    return sorted_centers

def get_center(top_left, bottom_right):
    return ((top_left[0] + bottom_right[0]) / 2, (top_left[1] + bottom_right[1]) / 2)

def rmv_duplicates(pos_list, path_list):
    upper_left = [(p[0], p[1]) for p in pos_list]
    lower_right = [(p[2], p[3]) for p in pos_list]
    center = [get_center(tl, br) for tl, br in zip(upper_left, lower_right)]
    order = sort_by_distance(center)
    remove_idx = []
    pos = []
    path = []
    for i in order:
        ratio = cal_ratio(pos_list[order[i-1]], pos_list[order[i]])
        ratio_size = cal_ratio_size(pos_list[order[i-1]], pos_list[order[i]])
        if ratio is not None and 0.8 < ratio < 1.25 and ratio_size is not None and 0.8 < ratio_size < 1.25:
            print(ratio, ratio_size)
            # print(f'removing position{path_list[order[i-1]]}')
            remove_idx.append(order[i-1])
            continue
    for idx in range(len(pos_list)):
        if idx not in remove_idx:
            pos.append(pos_list[idx])
            path.append(path_list[idx])
    return pos, path

def cal_ratio(pos_1, pos_2):
    x1_ = max(pos_1[0], pos_2[0])
    y1_ = max(pos_1[1], pos_2[1])
    x2_ = min(pos_1[2], pos_2[2])
    y2_ = min(pos_1[3], pos_2[3])
    if x2_ < x1_ or y2_ < y1_:
        return None
    return abs((x2_ - x1_)*(y2_ - y1_)/((pos_1[3] - pos_1[1]) * (pos_1[2] - pos_1[0])))

def cal_ratio_size(pos_1, pos_2):
    x1_ = max(pos_1[0], pos_2[0])
    y1_ = max(pos_1[1], pos_2[1])
    x2_ = min(pos_1[2], pos_2[2])
    y2_ = min(pos_1[3], pos_2[3])
    return abs(((pos_2[3] - pos_2[1]) * (pos_2[2] - pos_2[0]))/((pos_1[3] - pos_1[1]) * (pos_1[2] - pos_1[0])))

def detect_duplicate(label, pos, position_of_masks, masked_path, text_list):
    for p, f in zip(position_of_masks, masked_path):
        # text_list = ['head', 'arm', 'leg', 'foot', 'hand', 'ear', 'eye']
        ratio = cal_ratio(pos, p)
        l = find_word_in_string(text_list, f.split('/')[-1])
        print(l, label, ratio)
        if ratio is not None and ratio > 0.1 and label == l:
            print(f'Incorrect detection occurred: {label} at {pos}.')
            return True
    
    return False

def find_word_in_string(word_list, input_string):
    for word in word_list:
        if word in input_string:
            return word  
    return None 

def get_text_list(text):
    result = [item.strip() for item in text.split(',')]
    # print(result)
    return result

def compare(list_1, list_2, text_list):
    counter = {}
    record = {}
    record['missing'] = []
    record['redundant'] = []
   
    # count body parts in list_1
    for l in list_1:
        file = l['image_path'].split('/')[-1]
        if len(file.split(' ')) > 1:
            continue
        find_word = find_word_in_string(text_list, file)
        if find_word not in counter:
            counter[find_word] = 0
        counter[find_word] += 1
    
    for l in list_2:
        file = l['image_path'].split('/')[-1]
        # print(file)
        if len(file.split(' ')) > 1:
            continue
        find_word = find_word_in_string(text_list, file)
        if find_word not in counter:
            counter[find_word] = 0
        counter[find_word] -= 1

    for key, value in counter.items():
        if value < 0:
            record['missing'].append(key)
            print(f'The original image appears to have {abs(value)} missing {key}(s).')
        # if value > 0:
        #     record['redundant'].append(key)
        #     print(f'The original image appears to have {value} redundant {key}(s).')
    return record


def enlarge(proportion, position, image_width, image_height):
    box_width = position[2] - position[0]
    box_height = position[3] - position[1]
    position = [int(max(position[0] - proportion * box_width, 0)), int(max(position[1] - proportion * box_height, 0)), int(min(position[2] + proportion * box_width, image_width)), int(min(position[3] + proportion * box_height, image_height))]
    return position

def get_prompt(phrase):
    prompt = f"{phrase}, masterpiece, best quality, high quality ultrarealistic"
    negative_prompt = f"bad {phrase}, no {phrase}, monster, nonhuman, digital, machine, cartoon style, comic, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username"
    return prompt, negative_prompt



def cal_size(position):
    return (position[2]-position[0]) * (position[3]-position[1])

def adjust(args,position, image_width, image_height):
    # 0919 mqy
    # 问题：虽然有些ours给出的bbox有点大，但是直接原地缩小可能导致新的bbox和原本的身体没有交际——影响inpainting的效果
    # 解决方案：参考“切线”mask的生成方式？
    size = cal_size(position)
    upper_bound = args.size_upper_bound
    lower_bound = args.size_lower_bound
    proportion = args.proportion
    while size < lower_bound :
        proportion = abs(proportion)
        position = enlarge(proportion, position, image_width, image_height)
        size = cal_size(position)
    while size > upper_bound:
        proportion = abs(proportion) * (-1)
        position = enlarge(proportion, position, image_width, image_height)
        size = cal_size(position)
    return position

def gen_reasoning_prompt(data):
    redundancy = ''
    deprivation = ''

    for d in data:
        if d['condition'] == 'redundant':
            if redundancy:
                redundancy += ', '
            redundancy += d['label']
        elif d['condition'] == 'deprivative':
            if deprivation:
                deprivation += ', '
            deprivation += d['label']
    if len(redundancy) > 0 and len(deprivation) > 0:
        prompt_template = "The person in this picture has extra {redundancy} and is missing {deprivation}. Based on the image, is this assessment correct? Please explain your reasoning."
    elif len(redundancy) > 0:
        prompt_template = "The person in this picture has extra {redundancy}. Based on the image, is this assessment correct? Please explain your reasoning."
    elif len(deprivation) > 0:
        prompt_template = "The person in this picture is missing {deprivation}. Based on the image, is this assessment correct? Please explain your reasoning."
    else:
        prompt_template = "The person in this picture does not show any obvious abnormalities. Based on the image, is this assessment correct? Please explain your reasoning."

def get_SD_prompt(phrase):
    prompt = f"{phrase}, masterpiece, best quality, high quality ultrarealistic"
    negative_prompt = f"bad {phrase}, no {phrase}, monster, nonhuman, digital, machine, cartoon style, comic, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username"
    # realistic inpainting of a missing {phrase}, human anatomy, seamless integration, natural skin tone, accurate lighting, {phrase} positioned naturally with body, highly detailed, smooth skin texture, realistic muscle definition, soft shadows, perfect alignment with body proportions, precise {phrase} position, photorealistic, attention to joint alignment, clear and defined hand and fingers, natural {phrase} movement, subtle highlights, skin shading consistent with existing lighting, (masterpiece: 2), best quality, ultra highres, original, extremely detailed, perfect lighting
    # NSFW, (worst quality:2), (low quality:2), (normal quality:2), (monochrome), (grayscale), (skin blemishes:1.331), (acne:1.331), (age spots:1.331), (extra fingers:1.61051), (deformed limbs:1.331), (malformed limbs:1.331), (ugly:1.331), (poorly drawn hands:1.5), (poorly drawn feet:1.5), (poorly drawn face:1.5), (mutated hands:1.331), (bad anatomy:1.21), (distorted face:1.331), (disfigured:1.331), (low contrast), (underexposed), (overexposed), (amateur), (blurry), (bad proportions:1.331), (extra limbs:1.331), (fused fingers:1.61051), (unclear eyes:1.331)

    return prompt, negative_prompt

def get_SD_prompt(phrase):

    prompts = {
        "head": "realistic inpainting of a human head, human anatomy, seamless integration with the body, natural skin tone, accurate head shape, realistic hair, smooth skin texture, detailed facial features, lifelike expressions, proper alignment with neck and shoulders, photorealistic, consistent lighting, perfect anatomy, attention to details like eyes, nose, lips, ears, highly detailed, (masterpiece: 2), best quality, ultra highres, original, extremely detailed, perfect lighting",
        
        "foot": "realistic inpainting of a human foot, human anatomy, seamless integration with leg, natural skin tone, accurate foot shape, realistic toes and arch, smooth skin texture, photorealistic detail, proper foot alignment, attention to ankle joint, consistent lighting and shadows, lifelike muscle definition, highly detailed, soft highlights, natural toe positioning, (masterpiece: 2), best quality, ultra highres, original, extremely detailed, perfect lighting",
        
        "leg": "realistic inpainting of a human leg, human anatomy, seamless integration with body, natural skin tone, accurate muscle and bone structure, smooth texture, realistic leg proportions, soft shadows, consistent with existing lighting, high detail, photorealistic, proper alignment with hips and feet, natural stance, attention to knee joint, realistic thigh and calf definition, (masterpiece: 2), best quality, ultra highres, original, extremely detailed, perfect lighting",
        
        "arm": "realistic inpainting of a human arm, human anatomy, seamless integration, natural skin tone, accurate lighting, arm positioned naturally with body, highly detailed, smooth skin texture, realistic muscle definition, soft shadows, perfect alignment with body proportions, precise arm position, photorealistic, attention to joint alignment, clear and defined hand and fingers, natural arm movement, subtle highlights, skin shading consistent with existing lighting, (masterpiece: 2), best quality, ultra highres, original, extremely detailed, perfect lighting",
        
        "ear": "realistic inpainting of a human ear, human anatomy, seamless integration, natural skin tone, accurate ear shape and positioning, smooth skin texture, realistic cartilage definition, subtle shadows and highlights, consistent with existing lighting, precise ear contours, high detail, photorealistic, perfect alignment with the head, natural ear proportions, (masterpiece: 2), best quality, ultra highres, original, extremely detailed, perfect lighting",
        
        "hand": "realistic inpainting of a human hand, human anatomy, seamless integration with arm, natural skin tone, accurate finger positioning, realistic palm and knuckles, smooth skin texture, lifelike muscle and vein details, soft highlights and shadows, consistent with existing lighting, photorealistic, high detail, precise joint alignment, proper proportions, natural hand posture, (masterpiece: 2), best quality, ultra highres, original, extremely detailed, perfect lighting",
        
        "eye": "realistic inpainting of a human eye, human anatomy, seamless integration with face, natural eye shape, accurate iris color, realistic eyelashes, smooth skin around eye, detailed pupil and cornea, lifelike reflections, subtle shading, consistent lighting, photorealistic detail, proper alignment with other facial features, highly detailed, natural expression, (masterpiece: 2), best quality, ultra highres, original, extremely detailed, perfect lighting",
        
        "knee": "realistic inpainting of a human knee, human anatomy, seamless integration with leg, natural skin tone, realistic knee cap and muscle definition, smooth texture, consistent lighting, soft shadows, high detail, photorealistic, proper alignment with thigh and calf, lifelike knee joint structure, attention to skin folds, accurate proportions, subtle highlights, (masterpiece: 2), best quality, ultra highres, original, extremely detailed, perfect lighting"
    }
    
    neg_dict = {
        "ear": "NSFW, (worst quality:2), (low quality:2), (normal quality:2), (monochrome), (grayscale), (skin blemishes:1.331), (acne:1.331), (age spots:1.331), (deformed ears:1.331), (malformed ears:1.331), (ugly:1.331), (poorly drawn ears:1.5), (bad anatomy:1.21), (disfigured:1.331), (low contrast), (underexposed), (overexposed), (amateur), (blurry), (bad proportions:1.331)",
        "head": "NSFW, (worst quality:2), (low quality:2), (normal quality:2), (monochrome), (grayscale), (skin blemishes:1.331), (acne:1.331), (age spots:1.331), (deformed heads:1.331), (malformed heads:1.331), (ugly:1.331), (poorly drawn heads:1.5), (bad anatomy:1.21), (disfigured:1.331), (low contrast), (underexposed), (overexposed), (amateur), (blurry), (bad proportions:1.331)",
        "leg": "NSFW, (worst quality:2), (low quality:2), (normal quality:2), (monochrome), (grayscale), (skin blemishes:1.331), (acne:1.331), (age spots:1.331), (deformed legs:1.331), (malformed legs:1.331), (ugly:1.331), (poorly drawn legs:1.5), (bad anatomy:1.21), (disfigured:1.331), (low contrast), (underexposed), (overexposed), (amateur), (blurry), (bad proportions:1.331)",
        "foot": "NSFW, (worst quality:2), (low quality:2), (normal quality:2), (monochrome), (grayscale), (skin blemishes:1.331), (acne:1.331), (age spots:1.331), (deformed feet:1.331), (malformed feet:1.331), (ugly:1.331), (poorly drawn feet:1.5), (bad anatomy:1.21), (disfigured:1.331), (low contrast), (underexposed), (overexposed), (amateur), (blurry), (bad proportions:1.331)",
        "eye": "NSFW, (worst quality:2), (low quality:2), (normal quality:2), (monochrome), (grayscale), (deformed eyes:1.331), (malformed eyes:1.331), (ugly:1.331), (poorly drawn eyes:1.5), (bad anatomy:1.21), (disfigured:1.331), (low contrast), (underexposed), (overexposed), (amateur), (blurry), (bad proportions:1.331), (unclear eyes:1.331)",
        "hand": "NSFW, (worst quality:2), (low quality:2), (normal quality:2), (monochrome), (grayscale), (skin blemishes:1.331), (acne:1.331), (age spots:1.331), (extra fingers:1.61051), (deformed hands:1.331), (malformed hands:1.331), (ugly:1.331), (poorly drawn hands:1.5), (bad anatomy:1.21), (disfigured:1.331), (low contrast), (underexposed), (overexposed), (amateur), (blurry), (bad proportions:1.331), (fused fingers:1.61051)",
        "knee": "NSFW, (worst quality:2), (low quality:2), (normal quality:2), (monochrome), (grayscale), (skin blemishes:1.331), (acne:1.331), (age spots:1.331), (deformed knees:1.331), (malformed knees:1.331), (ugly:1.331), (poorly drawn knees:1.5), (bad anatomy:1.21), (disfigured:1.331), (low contrast), (underexposed), (overexposed), (amateur), (blurry), (bad proportions:1.331)",
        "arm": "NSFW, (worst quality:2), (low quality:2), (normal quality:2), (monochrome), (grayscale), (skin blemishes:1.331), (acne:1.331), (age spots:1.331), (extra fingers:1.61051), (deformed limbs:1.331), (malformed limbs:1.331), (ugly:1.331), (poorly drawn hands:1.5), (poorly drawn feet:1.5), (poorly drawn face:1.5), (mutated hands:1.331), (bad anatomy:1.21), (distorted face:1.331), (disfigured:1.331), (low contrast), (underexposed), (overexposed), (amateur), (blurry), (bad proportions:1.331), (extra limbs:1.331), (fused fingers:1.61051), (unclear eyes:1.331)"
    }

    if phrase not in prompts:
        return None
    else:
        return prompts[phrase], neg_dict[phrase]