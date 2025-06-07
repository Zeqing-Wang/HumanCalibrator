import argparse
import torch
from llava.constants import (IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN,
                             DEFAULT_IM_END_TOKEN, IMAGE_PLACEHOLDER, )
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from tqdm import tqdm
from PIL import Image, ImageDraw
from io import BytesIO
import requests
import json
import re
import os

import logging
import traceback
import subprocess
from concurrent.futures import ThreadPoolExecutor
import concurrent
# os.environ['CUDA_VISIBLE_DEVICES'] = "5"
# inapinting
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--val_image_dir", type=str, required=False, help="Directory containing extracted keyframes.",
                        default="./MoldingHuman/data_coco/coco_val_replace_person_mask_filter")
    parser.add_argument("--output_dir", type=str, required=False, default='./MoldingHuman/baseline/llava-v1.5-7b-task-train_ver_with_neg_inpainting_1721922413_0392356_2_epoch_without_eye_knee_filter',
                        help="Directory to save output files.")
    parser.add_argument("--question", type=str, default="Describe the image in detail.",
                        help="Question to ask about the image.")

    parser.add_argument("--model-path", type=str, required=False, help="Path to the pretrained model.",
                        default="./LLaVA/checkpoints/llava-v1.5-7b-task-train_ver_with_neg_inpainting_1721922413_0392356_2_epoch_without_eye_knee_filter.json")
    parser.add_argument("--model-base", type=str, default=None, help="Base model to use.")
    parser.add_argument("--conv-mode", type=str, default=None, help="Conversation mode.")
    parser.add_argument("--sep", type=str, default=",", help="Separator.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature for sampling.")
    parser.add_argument("--top_p", type=float, default=None, help="Top-p sampling parameter.")
    parser.add_argument("--num_beams", type=int, default=1, help="Number of beams for beam search.")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum number of new tokens to generate.")
    parser.add_argument("--max_workers", type=int, default=1, help="Maximum number of new tokens to generate.")
    parser.add_argument("--output_freq", type=int, default=10, help="Save json frequency.")
    return parser.parse_args()

def find_word_in_string(word_list, input_string):
    for word in word_list:
        if word in input_string:
            return word  
    return None 

def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def load_model(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    return model, image_processor, tokenizer, conv_mode


def prepare_conv(qs, model, tokenizer, conv_mode):
    conv = conv_templates[conv_mode].copy()
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = (tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda())
    return input_ids


def inference(image_file, input_ids, model, image_processor, tokenizer, args):
    #images = load_images(image_files)
    image = load_image(image_file)
    images = [image]
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return outputs

def natural_sort_key(s):
    """
    Extract numbers from filenames for natural sorting.
    """
    import re
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def process_folder_singe(model, image_processor, tokenizer, conv_mode, image_folder, questions, args):
    # print('Into process ---------------------------------------------')
    # counter of folders
    idx_f = 0
    # counter of images
    idx_img = 0
    # loss dict
    loss_dict = {}
    # for folder in tqdm(os.listdir(val_image_dir)):
    #     idx_f += 1
        
    image_path = os.path.join(args.val_image_dir, image_folder)
    #     print(f'processing [{idx_f}/1588]: {folder}')
    
    for image in os.listdir(image_path):
        if 'eye' in image:
            continue
        if 'knee' in image:
            continue
        iou_loss_sum = 0
        idx_img += 1
        # print(f'processing [{idx_img}/15566]: {image}')
        flag_ori = False # check if the current image is the original image (unmasked)
        image_file = os.path.join(image_path, image)
        if image.startswith('ori_image'):
            flag_ori = True
        else:
            
            copied_image = load_image(image_file)
            width, height = copied_image.size

            # get the size of the mask
            str_img = image
            matches_ori = str_img.strip('.jpg').split('_')
            number_list_ori = [float(item) for item in matches_ori if '.' in item]
            # print(f'ori bbox: ', number_list_ori)

            x1_ori = int(number_list_ori[0] * width)
            y1_ori = int(number_list_ori[1] * height)
            x2_ori = int(number_list_ori[2] * width)
            y2_ori = int(number_list_ori[3] * height)

            ori_size = (x2_ori - x1_ori) * (y2_ori - y1_ori)
        question_ans = {}
        for idx, question in tqdm(enumerate(questions)):
            # cal the classification acc
            correct_num = 0
            
            # print(question)
            input_ids = prepare_conv(question, model, tokenizer, conv_mode)
            
            
            result = inference(image_file, input_ids, model, image_processor, tokenizer, args)
            print('here:', result)
            question_ans[question] = result
            try:
                matches = re.findall(r'\[(.*?)\]', result)
                list_str = matches[0] if matches else ''
                number_list = list(map(float, list_str.split(',')))
                print('RE Res:',number_list)

                if flag_ori:
                    iou_loss = 1
                else:
                    # 将归一化坐标转换为像素坐标
                    x1 = int(number_list[0] * width)
                    y1 = int(number_list[1] * height)
                    x2 = int(number_list[2] * width)
                    y2 = int(number_list[3] * height)
                    # 创建一个可以用来在图像上绘画的对象
                    copied_image = load_image(image_file)
                    draw = ImageDraw.Draw(copied_image)
                
                    # 画出矩形框
                    draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
                

                    # add text
                    # text_list = ['head', 'arm', 'leg', 'foot', 'hand', 'ear', 'eye', 'knee']
                    text_list = ['head', 'arm', 'leg', 'foot', 'hand', 'ear']
                
                    found_word = find_word_in_string(word_list=text_list, input_string=result)
                    text = found_word
                    
                    # consider two situation
                    # ori image
                    if flag_ori:
                        if text is None:
                            correct_num += 1
                    else:
                        if text == image.split('_')[1]:
                            correct_num += 1
                        pass
                    
                    text_position = (x1+2, y1)
                    draw.text(text_position, text, font=None, fill="red")
    
                

                    # 计算重合部分面积占比
                    x1_inter  = max(x1, x1_ori)
                    y1_inter  = max(y1, y1_ori)
                    x2_inter = min(x2, x2_ori)
                    y2_inter = min(y2, y2_ori)

                    #save
                    if not os.path.exists(os.path.join(args.output_dir,  'images_res')):
                        os.mkdir(os.path.join(args.output_dir,  'images_res'))
                    save_dir = os.path.join(args.output_dir,  'images_res' , image_folder)
                    print('save dir:', save_dir)
                    
                    if not os.path.exists(save_dir):
                        print('save dir created:', save_dir)
                        os.mkdir(save_dir)
                    copied_image.save(os.path.join(save_dir,f'{image}_bbox_q_{idx}.jpg'))
                    print(type(copied_image))

                    if x2_inter <= x1_inter or y2_inter <= y1_inter:
                        size_inter = 0
                    else:
                        size_inter = (x2_inter - x1_inter) * (y2_inter - y1_inter)
                    size_union = ori_size + (x2 - x1) * (y2 - y1) - size_inter # bbox并集大小
                    proportion = size_inter / size_union # IoU
                    size_min_closure = (max(x2, x2_ori) - min(x1, x1_ori))*(max(y2, y2_ori) - min(y1, y1_ori)) # bbox最小闭包大小
                    # print('IoU Loss detial:')
                    # print('size_min_closure:', size_min_closure)
                    # print('proportion:', proportion)
                    # print('size_union:', size_union)
                    iou_loss = 1 - (proportion - (size_min_closure - size_union)/size_min_closure)
                print(f'iou_loss = {iou_loss}')
                iou_loss_sum += iou_loss
            
            except Exception as e:
                # print(e)
                logging.exception(e)
                continue


        # this is image, and to the whole folder
        # 在字典中添加本图片的平均loss
        iou_loss_avg = iou_loss_sum / 5
        key = image_folder + '/' + image
        if key not in loss_dict:
            loss_dict[key] = {}
            # dump implemention: loss_dict[key]['acc'] = acc
            loss_dict[key]['acc'] = correct_num/len(questions)
            loss_dict[key]['iou_loss'] = iou_loss_avg
            loss_dict[key]['question_ans'] = question_ans
        print(f'{image} loss : {iou_loss_avg}')
    return image_folder, loss_dict
    # if idx > 0:
    #     break
    # print(f'Processed {idx} image folers.')
    
    pass


def main(args):
    val_image_dir = args.val_image_dir
    output_dir = args.output_dir
    print('output_path:', output_dir)
    os.makedirs(output_dir, exist_ok=True)
    model, image_processor, tokenizer, conv_mode = load_model(args)
    questions = [
        'Are there any missing parts on the person shown in the image?',
        # 'Is the person in the image missing any parts?',
        # 'Are any of the parts of the person in the picture missing?',
        # 'Is the person shown in the image incomplete?',
        # 'Is the person in the photograph missing any body parts?'
        ]
    count = 0
    # per folder process
    val_image_dirs = os.listdir(val_image_dir)
    
    # change multi process wzq -----------------------------------
    
    
    executor = ThreadPoolExecutor(max_workers=args.max_workers)
    futures = [
        executor.submit(process_folder_singe,
                        model,
                        image_processor,
                        tokenizer,
                        conv_mode,
                        image_folder, 
                        questions,
                        args
                        )
        # break point continue
        # expected output folder path
        for image_folder in val_image_dirs
    ]
    # for image_folder in val_image_dirs if not os.path.exists(os.path.join(output_dir, 'images_res' ,image_folder)
    # output path
    output_path = os.path.join(output_dir, 'new_ours_1104_without_knee_eye.json')
    
    # resout dict
    loss_dict = {}
    # multi process excute
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(val_image_dirs)):
        image_folder, loss_dict_res = future.result()
        loss_dict[image_folder] = loss_dict_res
        
        # loss_dict.append(res['question_id'])
        if count % args.output_freq == 0:
            json.dump(loss_dict, open(output_path, "w"))
        count = count + 1
    json.dump(loss_dict, open(output_path, "w"))
    

if __name__ == "__main__":
    args = parse_args()
    main(args)