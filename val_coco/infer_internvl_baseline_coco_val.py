# import os
# import json
# from PIL import Image
# from tqdm import tqdm
# from lmdeploy.vl import load_image
# from lmdeploy import pipeline, TurbomindEngineConfig, ChatTemplateConfig

import os
import json
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import math
# from transformers import modeling_phi3
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        'InternVL2-1B': 24, 'InternVL2-2B': 24, 'InternVL2-4B': 32, 'InternVL2-8B': 32,
        'InternVL2-26B': 48, 'InternVL2-40B': 60, 'InternVL2-Llama3-76B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def find_word_in_string(word_list, input_string):
    for word in word_list:
        if word in input_string:
            return word  
    return None 


def main():
    val_image_dir = './MoldingHuman/data_coco/coco_val_replace_person_mask_filter'
    output_dir = './MoldingHuman/baseline/InternVL2-1B'
    os.makedirs(output_dir, exist_ok=True)
    

    
    path = 'InternVL2-1B'
    device_map = split_model('InternVL2-1B')


    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        # use_fast=False,
        device_map = device_map).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

    generation_config = dict(max_new_tokens=1024, do_sample=False)
    
    question = "Describe the image in detail."  # or any other specific question
    questions = [
        'Are there any missing parts on the person shown in the image?'
        ]
    questions = ['Are there any missing parts on the person shown in the image? If yes, please answer from \'head\', \'arm\', \'leg\', \'foot\', \'hand\', \'ear\', \'eye\', \'knee\'; otherwise, please answer \'no\'.']
    questions = ['Are there any absent body parts in the person shown in the image? If yes, please answer from \'head\', \'arm\', \'leg\', \'foot\', \'hand\', or \'ear\'; otherwise, please answer \'no\'. Answer the question using a single word:']
    res = {}
    text_list = ['head', 'arm', 'leg', 'foot', 'hand', 'ear']
    idx_img = 0
    for folder in tqdm(os.listdir(val_image_dir)):
        folder_path = os.path.join(val_image_dir, folder)
        res[folder] = {}
        for image_file in os.listdir(folder_path):
            idx_img += 1
            total_acc = 0
            flag_ori = False
            if image_file.startswith('ori_image'):
                flag_ori = True
            else:
                image_label = find_word_in_string(word_list=text_list, input_string=image_file)
                
                
            res[folder][image_file] = {}
            res[folder][image_file]["acc"] = -1
            res[folder][image_file]["iou_loss"] = -1
            res[folder][image_file]["question_ans"] = {}
            for idx, question in enumerate(questions):
                image_path = os.path.join(folder_path, image_file)
                # image = Image.open(image_path).convert("RGB")
                # intern_image = load_image(image_path)
                pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
                question_input = '<image>\n{}'.format(question)
                response = model.chat(tokenizer, pixel_values, question_input, generation_config)
                print(f'User: {question_input}\nAssistant: {response}')
                
                
                # response = pipe((question, intern_image))
                found_word = find_word_in_string(word_list=text_list, input_string=response)
                res[folder][image_file]["question_ans"][question] = response
                if flag_ori:
                    if found_word == None:
                        total_acc += 1
                else:
                    if found_word == image_label:
                        total_acc +=1
    # Save results to JSON
            if idx_img % 10 == 0:
                json.dump(res, open(os.path.join(output_dir, 'Intern_VL_infer.json'), "w"))

if __name__ == "__main__":
    main()
