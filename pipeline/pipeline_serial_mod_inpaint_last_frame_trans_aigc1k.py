from load_args import parse_args
from llava.constants import (IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN,DEFAULT_IM_END_TOKEN, IMAGE_PLACEHOLDER, )
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
import cv2
import glob
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from tqdm import tqdm
from io import BytesIO
import requests
import json
import math
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
sys.path.append(os.path.join(current_dir, '..', 'GroundingDINO'))
sys.path.append(os.path.join(current_dir, '..', 'MoldingHuman'))
from grounding_for_pipeline_v2 import grounding_new, load_model
from infer_llava_ours_mask_only import find_word_in_string, natural_sort_key, inference, prepare_conv, load_model_llava, load_image

from inference import cal_ratio, get_center, sort_by_distance, euclidean_distance, find_word_in_string, get_text_list, compare, enlarge, get_prompt, cal_size, adjust, cal_ratio_size, gen_reasoning_prompt
from redundant_solution import redundant_sol

# model_path = './Real-ESRGAN/weights/RealESRGAN_x4plus.pth'
# if 'x4plus' in model_path:
#     outscale = 4
#     netscale = 4
# dni_weight = [0.5, 0.5]
# tile = 0
# tile_pad = 10
# pre_pad = 0

# model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

# upsampler = RealESRGANer(
# scale=netscale,
# model_path=model_path,
# dni_weight=dni_weight,
# model=model,
# tile=tile,
# tile_pad=tile_pad,
# pre_pad=pre_pad,
# half=False,
# gpu_id=0)

def single_image_super_resolution(upsampler, img, outscale):
    output, _ = upsampler.enhance(img, outscale=4)
    return output
    pass
    

# NMS变体
def merge_overlapping_boxes(boxes, types, classes, iou_threshold=0.5):
    def calculate_iou(box1, box2):
        # 计算两个bbox的交并比（IoU）
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0

    def merge_boxes(box1, box2):
        # 合并两个bbox
        return [
            min(box1[0], box2[0]),
            min(box1[1], box2[1]),
            max(box1[2], box2[2]),
            max(box1[3], box2[3])
        ]

    # 将具有相同type和class的bbox分组
    groups = {}
    for box, t, c in zip(boxes, types, classes):
        key = (t, c)
        if key not in groups:
            groups[key] = []
        groups[key].append(box)

    merged_boxes = []
    merged_types = []
    merged_classes = []

    for (t, c), group in groups.items():
        while group:
            base = group.pop(0)
            i = 0
            while i < len(group):
                if calculate_iou(base, group[i]) > iou_threshold:
                    base = merge_boxes(base, group.pop(i))
                else:
                    i += 1
            merged_boxes.append(base)
            merged_types.append(t)
            merged_classes.append(c)

    return merged_boxes, merged_types, merged_classes

'''
Parameters:
    image_files(list): The path of input image.
    args: arguments.
    idx(int): index of the image.用于命名
    output_path_privative(str): output path.
    model: The loaded model.
Return:
    result_pair(dict): A dictionary of detection result info.
    position(list[float]): The coordinates of detected body part.
    mask_image(PIL.Image): The mask of detected area.
'''
def detection(image_files, args, idx, output_path_deprivative, ours_model):
    
    # 准备问题并推断
    input_ids = prepare_conv(args.question, ours_model, tokenizer, conv_mode)
    result = inference(image_files, input_ids, ours_model, image_processor, tokenizer, args)
    position, cur_image, mask_image = None, None, None
    
    print(f'Question: {args.question}')
    print(f'Result: {result}')
    
    # 查找结果中的关键词
    
    print('text_list: ',text_list)
    print('results: ',result)
    find_word = find_word_in_string(text_list, result)
    if find_word is not None:
        # 提取坐标信息
        matches = re.findall(r'\[(.*?)\]', result)
        list_str = matches[0] if matches else ''
        number_list = list(map(float, list_str.split(',')))
        print('Bounding Box Coordinates:', number_list)
        
        # 获取原始图像
        ori_image = Image.open(image_files[0])
        image_width, image_height = ori_image.size
        position = [
            int(number_list[0] * image_width),
            int(number_list[1] * image_height),
            int(number_list[2] * image_width),
            int(number_list[3] * image_height)
        ]

        # 在图像上绘制original检测框
        cur_image = ori_image.copy()
        draw_bbox = ImageDraw.Draw(cur_image)
        draw_bbox.rectangle(position, outline='blue', width=2)
        text_position = (position[0] + 2, position[1])
        draw_bbox.text(text_position, f'ori_{find_word}', fill="blue")

        # calculate and adjust the mask size
        position_new = adjust(args, position, image_width, image_height)

        # 在图像上绘制enlarged检测框
        draw_bbox = ImageDraw.Draw(cur_image)
        draw_bbox.rectangle(position_new, outline='red', width=2)
        text_position = (position_new[0] + 2, position_new[1])
        draw_bbox.text(text_position, find_word, fill="red")

        # mask
        mask_image = Image.new('L', (image_width, image_height), 0)
        draw = ImageDraw.Draw(mask_image)
        draw.rectangle(position_new, fill="white")
        # save detected image
        detected_path = os.path.join(output_path_deprivative,  f'detected_image_{idx}.png')
        cur_image.save(detected_path)
        print(f'Detection result {idx} image saved to {detected_path}')
        
    result_pair = {
        'image_file' : image_files[0],
        'question' : args.question,
        'position' :position,
        'answer' : result,
        'bbox_size': cal_size(position_new) if position is not None else None,
        'find_word': find_word
    }
    
    # 可以把find_word放进result_pair中，简化后续函数中没有用到的result_pair键值
    return result_pair, mask_image

'''
Parameters:
    masked_image(PIL.Image): The image to be inpainted.
    mask(PIL.Image): The diffusion model operate with the mask info of which.
    find_word(str): The label of the masked body part.
    task(str): A phrase describing the current stage of our pipeline.用于命名
    output_path(str): The directory to save the images.
Return:
    temp_path(str): The path to the inpainted image. 
'''
def inpainting(masked_image, mask, find_word, task, output_path, seed):
    prompt, negative_prompt = get_prompt(find_word)    
    image = masked_image.copy()
    image = image.convert("RGB")  
    mask_image = mask.convert("L")

    # WZQ 1026 Step1
    # 获取原始图片的width和height
    image_size = image.size
    width = image_size[0]
    height = image_size[1]

    # 这里需要检查下weight和height是不是都能被8整除，理论上aigc生成的都应该能满足这个的，毕竟生成的时候size基本也都是8的倍数
    if width % 8 != 0:
        width = math.ceil(width/8) * 8
    if height % 8 != 0:
        height = math.ceil(height/8) * 8

    # 0924 MQY
    # 增加了控制inpainting生成随机性的seed
    original_size = image.size
    generator = torch.Generator(device="cuda").manual_seed(seed)  
    # Perform inpainting using the pipeline
    result_image = pipe(
        prompt=prompt, 
        negative_prompt=negative_prompt, 
        generator=generator,
        image=image, 
        mask_image=mask_image,
        # WZQ 1026 Step2
        width = width,
        height = height
    ).images[0]

    # Resize the result back to the original size
    
    # WZQ 1026 加超分
    # img_np = np.array(result_image)
    # result_image = single_image_super_resolution(upsampler=upsampler, img=img_np, outscale=outscale)
    # result_image = Image.fromarray(result_image)
    
    # WZQ 1026 Step3 直接注释掉
    # result_image = result_image.resize(original_size, Image.LANCZOS)
    # Save the inpainted image
    temp_path = os.path.join(output_path, f'{task}_{find_word}.png')
    print(f'Saving inpainting image {temp_path}')
    result_image.save(temp_path)

    # return temp_path
    return result_image, temp_path


'''
    Removing the duplicates of grounding bboxes.

Parameters:
    pos_list(list[tuple]): grounding model得到的所有label以及对应bbox
    threshold(float): 允许两个mask最大重合占比
Return:
    pos(list[tuple]): 经过筛选的label与bbox
'''
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

        print(f'comparing {pos_list[idx1]}, {pos_list[idx2]}, ratio = {ratio}, ratio_1 = {ratio_1}')

        if ((ratio is not None and threshold < ratio < 1/threshold) or (ratio_1 is not None and threshold < ratio_1 < 1/threshold)) and pos_list[idx1][0] == pos_list[idx2][0]:
            remove_idx.add(idx1)
            print(f'removing {pos_list[idx1]}')

    for idx in range(len(pos_list)):
        if idx not in remove_idx:
            pos.append(pos_list[idx])

    return pos

def gen_mask(args, pos, size):
    width, height = size
    pos = adjust(args, pos, width, height)
    mask_image = Image.new('RGB', (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(mask_image)
    draw.rectangle(pos, fill="white")
    return mask_image, pos

def remove_incorrect_redundant(image, pos, label, pos_list, threshold, output_path):
    for idx, dic in enumerate(pos_list):
        p = dic['position']
        l = dic['label']
        enlarged_pos = dic['enlarged_position']

        if label != l:
            continue
        if dic['condition'] == 'deprivative':
            continue

        ratio_1 = cal_ratio(pos, enlarged_pos)
        ratio_2 = cal_ratio(enlarged_pos, pos)

        if (ratio_1 is not None and threshold < ratio_1 < 1/threshold) or (ratio_2 is not None and threshold < ratio_2 < 1/threshold):
            del pos_list[idx]
            print(f'A redundant {label} detection mistake occurred at {pos}.')
            mark_image = image.copy()
            draw_bbox = ImageDraw.Draw(mark_image)
            draw_bbox.rectangle(pos, outline='red', width=2)
            text_pos = (pos[0] + 2, pos[1])
            draw_bbox.text(text_pos, f'missing', fill="red")
            
            draw_bbox.rectangle(enlarged_pos, outline='blue', width=2)
            text_pos = (enlarged_pos[0] + 2, enlarged_pos[1])
            draw_bbox.text(text_pos, f'redundant', fill="blue")

            mark_path = os.path.join(output_path, f'red_missing_blue_redundant_{label}_{pos[0]}_{pos[1]}_{pos[2]}_{pos[3]}.png')
            mark_image.save(mark_path)
            return enlarged_pos
    return None

# 0921 MQY
# 在compare函数对比label(perdiction)的过程中加上split(' ')的步骤————in case of grounding给出形如'hand foot'的结果
def compare(image, pos, label, pos_list, output_path, threshold, task):
    for idx, dic in enumerate(pos_list):
        p = dic['position']
        l = dic['label'].split(' ')
        if label not in l:
            continue

        ratio_1 = cal_ratio(pos, p)
        ratio_2 = cal_ratio(p, pos)
        # ratio_size = cal_ratio_size(pos, p)
        if (ratio_1 is not None and threshold < ratio_1 < 1/threshold) or (ratio_2 is not None and threshold < ratio_2 < 1/threshold):
            if task == 'check_deprivative':
                del pos_list[idx]
                print(f'A redundant {label} detection mistake occurred at {pos}.')
            if task == 'redundant':
                mark_image = image.copy()
                draw_bbox = ImageDraw.Draw(mark_image)
                draw_bbox.rectangle(pos, outline='red', width=2)
                text_pos = (pos[0] + 2, pos[1])
                draw_bbox.text(text_pos, f'{task} {label}', fill="red")
                
                draw_bbox.rectangle(p, outline='blue', width=2)
                text_pos = (p[0] + 2, p[1])
                draw_bbox.text(text_pos, f'newly inpainting bbox', fill="red")

                mark_path = os.path.join(output_path, f'compared_marked_{task}_{label}_{pos[0]}_{pos[1]}_{pos[2]}_{pos[3]}.png')
                mark_image.save(mark_path)
            return True
    print(f'The original image appears to have a {task} {label} at {pos}.')

    if task == 'redundant':
        mark_image = image.copy()
        draw_bbox = ImageDraw.Draw(mark_image)
        draw_bbox.rectangle(pos, outline='red', width=2)
        text_pos = (pos[0] + 2, pos[1])
        draw_bbox.text(text_pos, f'{task} {label}', fill="red")

        mark_path = os.path.join(output_path, f'marked_{task}_{label}_{pos[0]}_{pos[1]}_{pos[2]}_{pos[3]}.png')
        mark_image.save(mark_path)

    return False
    
def get_mask_redundant(image_path, output_path, grounding_model, treshold=0.35,text_prompt = "head . hand . leg . foot . eye . ear . arm "):

    grounding_output_path = output_path / 'grounding_ori_res'
    grounding_output_path.mkdir(parents=True, exist_ok=True) 


    position_redundant_masks = grounding_new(str(image_path), str(grounding_output_path), grounding_model, treshold, text_prompt=text_prompt) # 用grounding模型获取原始图片上所有bounding box对角坐标数据
    position_redundant_masks = rmv_duplicates(position_redundant_masks, 0.8) 
    if len(position_redundant_masks) == 0:
        print(f'{image_path}:\nCould not ground anything.\n')
        return None
    return position_redundant_masks





def process_image(args, image_path, grounding_model, ours_model, rez_tools):
    
    results = []
    index = 0
    inpainting_seed = args.inpainting_seed

    '''Create output files paths'''
    # 创建输出路径 output_path
    # img_folder = image_path.parent  
    
    
    img_name = image_path.stem  
    # print(img_name)
    # assert False
    # output_path = img_folder / f'pipeline_res_0821_{img_name}'  
    output_path = Path(args.output_dir + '/' +img_name.split('_')[0]) / f'{img_name}'
    
    if os.path.exists(output_path):
        print('skip: ', output_path)
        return
    
    
    # print('output path::::', output_path)
    # assert False
    output_path.mkdir(parents=True, exist_ok=True)  
    # output_path_inpainting = output_path / 'inpainting_res'
    # output_path_inpainting.mkdir(parents=True, exist_ok=True)
    output_path_deprivative = output_path / 'ours_res'
    output_path_deprivative.mkdir(parents=True, exist_ok=True)
    output_path_redundant = output_path / 'redundant_res'
    output_path_redundant.mkdir(parents=True, exist_ok=True)
    final_image_path = image_path

    # 0921 MQY
    # Grounding: 提高阈值，用于缺失部分的假阳判断
    grounding_path = output_path_deprivative / 'grounding_with_higher_treshold'
    grounding_path.mkdir(parents=True, exist_ok=True)
    
    
    
    # WZQ 0922 Add 缺失判断的阈值，从args读取对应的阈值
    deprivative_treshold = args.deprivative_treshold
    
    # WZQ 0924 Add 添加 text_prompt，来控制多余
    redundant_class = args.text_list.replace(',',' . ')
    print('redundant_class prompt final:', redundant_class)
    # get the class list, concat to the prompt format
    
    
    
    # assert False
    
    grounding_ori_high_treshold = grounding_new(str(image_path), str(grounding_path), grounding_model, deprivative_treshold, text_prompt=redundant_class)

    # open image
    image = Image.open(str(image_path))
    size = image.size
    
    '''step 1: Reundancy solution'''
    print('\n_____________step 1: Redundancy Solution_______________')
    perception_redundant = []

    # get redundant grounding results
    
    
    # WZQ 0922 Add 多余判断mask的阈值 从args读取对应的值
    redundant_treshold = args.redundant_treshold
    position_redundant_masks = get_mask_redundant(image_path, output_path_redundant, grounding_model, redundant_treshold, text_prompt=redundant_class) # removed duplicates, original size
    repair_redundancy_image = image.copy()
    
    # inpainting and counting
    regen_image_redundant = [] # list of paths
    
    # WZQ 0924 添加异常处理，这里可能出现一个都没有grounding出来的情况
    if position_redundant_masks is not None:
        for idx, (label, pos) in enumerate(position_redundant_masks):
            output_path_temp = output_path_redundant / f'grounding_redundant_{label}_regenerate_{idx}'
            output_path_temp.mkdir(parents=True, exist_ok=True)
            # inpainting
            mask, new_pos = gen_mask(args, pos, size)
            inpainted_image, path = inpainting(image, mask, label, f'redundant_{label}_regenerate_{idx}', output_path_temp, inpainting_seed)
            

            image.save(str(output_path_temp / 'ori_image.jpg'))
            regen_image_redundant.append((path, [label, pos, new_pos]))
            
            # 计数
            # 0921 MQY
            # grounding加上treshold参数
            
            
            position_redundant_masks_new = grounding_new(path, str(output_path_temp), grounding_model, 0.25, text_prompt=redundant_class)
            redundant_dict = [{'position' : pos, 'label': label} for label, pos in position_redundant_masks_new]
            # 0924 MQY compare函数加上超参
            # 判断多余
            # 0928 MQY 把pos改成扩大过的new_pos
            if len(label.split(' ')) == 1 and not compare(image, new_pos, label, redundant_dict, output_path, args.comp_ratio_threshold_redundant, 'redundant'):

                
                # assert False
                temp_dict = {
                    'path' : path,
                    'position' : pos,
                    'enlarged_position' : new_pos,
                    'label' : label,
                    'condition' : 'redundant'
                }
                results.append(temp_dict)
                perception_redundant.append((label, pos))
        
                # Get a result image of redundancy repairement
                index += 1
                
                # WZQ 1026 Step4 这里不crop 直接存就可以 话说这里其实是不是可以不存。。但是不存的话下一步缺失要改路径，所以这里可以偷懒先存着
                inpainted_image = Image.open(path)
                final_image_path = os.path.join(output_path,  f'Final_image_{index}_redundant_{label}.png')
                inpainted_image.save(final_image_path)
                # 以下的全部注释掉
                # inpainted_image = Image.open(path)
                # cropped_regen_image = inpainted_image.crop(new_pos)
                # repair_redundancy_image.paste(cropped_regen_image, new_pos[:2])
                # final_image_path = os.path.join(output_path,  f'Final_image_{index}_redundant_{label}.png')
                # repair_redundancy_image.save(final_image_path)


    '''step 2: Deprivation solution'''
    print('\n_____________step 2: Deprivation Solution_______________')
    perception_deprivative = []
    find_word = 'before loop'
    
    
    image_files = [final_image_path]
    

    loop_idx = 0
    while find_word != None and loop_idx < args.max_loop:
        image_step2 = Image.open(final_image_path)
        size = image_step2.size
        loop_idx += 1

        result, mask_regen_image = detection(image_files, args, loop_idx, output_path_deprivative, ours_model)
        find_word = result['find_word']
        position = result['position']
        
        if find_word == None:
            break
        _, enlarged_pos = gen_mask(args, position, size)
            
        
        # 错误多余检测 remove incorrect redundant perception result 0911
        # 如果确定相应位置的redundant结果有误，应当删除对应的Final_image,获得新final_image 0913
        # 0928 MQY 改动remove_incorrect_redundant()函数内部的对比对象——result里的enlarged_pos
        to_remove_pos = remove_incorrect_redundant(image_step2, position, find_word, results, 0.4, output_path_deprivative) 
        
        # 0919 mqy：缺陷：一旦检测出错误多余，应当确保在本次循环解决此问题。
        # 然而错误多余源于inpainting模型的结果：把原本正常的部位抹除，使其缺失。
        # 这样检测出错误多于后，再次使用inpainting模型，希望能将那里补回去通常不太现实...inpainting模型依然会生成背景。
        if to_remove_pos is None:
            
            # 0921 MQY
            # 应该重新grounding的地方
            original_grounding_high_treshold = [{'position' : pos, 'label': label} for label, pos in grounding_ori_high_treshold]
            # 0924 MQY compare函数加上超参
            # if len(label.split(' ')) == 1 and compare(image_step2, position, find_word, original_grounding_high_treshold, output_path, args.comp_ratio_threshold_deprivative, 'check_FP'):
            if compare(image_step2, position, find_word, original_grounding_high_treshold, output_path, args.comp_ratio_threshold_deprivative, 'check_FP'):
                # 如果假阳，就跳过本次inpainting结果贴到Final以及加入感知results的过程
                continue

            # 添加results
            temp_dict = {
                    'path' : str(final_image_path), # !0919 暂时写为final_path
                    'position' : position,
                    'enlarged_position' : enlarged_pos,
                    'label' : find_word,
                    'condition' : 'deprivative'
            }
            results.append(temp_dict)    

        # inpainting
        # regenerated_image_path = inpainting(image_step2, mask_regen_image, find_word, f'regenerate_{loop_idx}', output_path_deprivative, inpainting_seed) 
        regenerated_image, regenerated_image_path = inpainting(image_step2, mask_regen_image, find_word, f'regenerate_{loop_idx}', output_path_deprivative, inpainting_seed) 
        # regenerated_image = Image.open(regenerated_image_path)
        

        # crop and paste
        index += 1
        # WZQ 1026 Step 5 这里和上面其实是一样的  直接存就OK
        final_image_path =os.path.join(output_path,  f'Final_image_{index}_deprivative_{find_word}.png') # save
        regenerated_image.save(final_image_path)
        print(f'Final image saved to {final_image_path}')
        image_files = [final_image_path]
        # 下面全部注释掉 
        # cropped_regen_image = regenerated_image.crop(enlarged_pos)
        # final_image = image_step2.copy()
        # final_image.paste(cropped_regen_image, enlarged_pos[:2])
        # final_image_path =os.path.join(output_path,  f'Final_image_{index}_deprivative_{find_word}.png') # save
        # final_image.save(final_image_path)
        # print(f'Final image saved to {final_image_path}')
        # image_files = [final_image_path]


    json_output_path = output_path / 'result_final.json'
    with open(json_output_path, "w") as f:
        json.dump(results, f)


    # WZQ 0922
    # 添加原图之前，对所有的bbox进行NMS变体
    nms_boxes = []
    nms_conditions = []
    nms_labels = []
    for r in results:
        nms_boxes.append(r['position'])
        nms_conditions.append(r['condition'])
        nms_labels.append(r['label'])
    
    merged_boxes, merged_conditions, merged_labels = merge_overlapping_boxes(nms_boxes, nms_conditions, nms_labels, iou_threshold=0.3)
    
    merged_len = len(merged_boxes)
    merged_box = []
    for i in range(merged_len):
        res_dict = {
            'position':merged_boxes[i],
            'label':merged_labels[i],
            'condition':merged_conditions[i]
        }
        merged_box.append(res_dict)
    
    results = merged_box

    # WZQ 0922 Add 这里可以根据result中的结果去添加bbox
    # 添加到原图上！
    # img_name = image_path.name  
    final_draw_bbox_image = Image.open(str(image_path))
    draw = ImageDraw.Draw(final_draw_bbox_image)
    
    output_image = output_path / 'final_res.png'
    for r in results:
        bbox_list = r['position']
        text = r['condition'] + '_' + r['label']
        if r['condition'] == 'redundant':
            draw.rectangle(bbox_list, outline='red', width=2)
            text_position = (bbox_list[0]+2, bbox_list[1])
            draw.text(text_position, text, font=None, fill="red")
        else:
            draw.rectangle(bbox_list, outline='blue', width=2)
            text_position = (bbox_list[0]+2, bbox_list[1])
            draw.text(text_position, text, font=None, fill="blue")
            pass
    
    final_draw_bbox_image.save(output_image)

if __name__ == '__main__':
    # args
    args = parse_args()
    image_path = args.image_dir
    print('image_dir:', image_path)
    text = args.text_list
    print('text: ', text)
    text_list = get_text_list(text)
    print('text_list: ',text_list)
    print(text_list)
    
    # load models
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stable-diffusion-2-inpainting",
        torch_dtype=torch.float16,
    )
    pipe.to("cuda")
    grounding_model = load_model("./GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py", "./GroundingDINO/weights/groundingdino_swinb_cogcoor.pth")
    ours_model, image_processor, tokenizer, conv_mode = load_model_llava(args)

    rez_model_name = args.reasoning_model
    rez_tools = []
    if rez_model_name == 'None':
        pass
    elif 'llava' in rez_model_name:
        rez_model, rez_image_processor, rez_tokenizer, rez_conv_mode = llava_load_model(rez_model_name)
        rez_tools = [rez_model, rez_image_processor, rez_tokenizer, rez_conv_mode]
    elif 'InternVL' in rez_model_name:
        rez_model, tokenizer = internVL_load_model(rez_model_name)
        rez_tools = [rez_model, tokenizer]
    

    image_dir = Path(image_path)
    # print('image dir:', image_dir)
    # assert False
    if image_dir.is_file():
        process_image(args, image_dir, grounding_model, ours_model, rez_tools)
    else:
        # WZQ 0922 Add, tackle the images' folder
        for folder in tqdm(image_dir.iterdir()):

            if folder.is_file():
                process_image(args, folder, grounding_model, ours_model, rez_tools)
            else:
                for img_file in folder.iterdir():
                    if img_file.is_file() and img_file.suffix == '.jpg':
                        process_image(args, img_file, grounding_model, ours_model, rez_tools)
