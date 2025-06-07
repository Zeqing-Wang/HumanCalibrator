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

# os.environ['CUDA_VISIBLE_DEVICES'] = "5"
# inapinting
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--val_image_dir", type=str, required=False, help="Directory containing extracted keyframes.",
                        default="./MoldingHuman/data_coco/coco_val_replace_person_mask_filter")
    parser.add_argument("--output_dir", type=str, required=False, default='./MoldingHuman/baseline/llava-v1.5-7b',
                        help="Directory to save output files.")
    parser.add_argument("--question", type=str, default="Describe the image in detail.",
                        help="Question to ask about the image.")
    # llava-v1.5-7b
    # llava-v1.5-13b
    # llava-v1.6-34b
    parser.add_argument("--model-path", type=str, required=False, help="Path to the pretrained model.",
                        default="/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None, help="Base model to use.")
    parser.add_argument("--conv-mode", type=str, default=None, help="Conversation mode.")
    parser.add_argument("--sep", type=str, default=",", help="Separator.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature for sampling.")
    parser.add_argument("--top_p", type=float, default=None, help="Top-p sampling parameter.")
    parser.add_argument("--num_beams", type=int, default=1, help="Number of beams for beam search.")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum number of new tokens to generate.")
    parser.add_argument("--output_freq", type=int, default=20, help="Output frequence.")
    # parser.add_argument("--output_file", type=int, default=10, help="Output frequence.")
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

def main(args):
    val_image_dir = args.val_image_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    model, image_processor, tokenizer, conv_mode = load_model(args)
    questions = [
        'Are there any missing parts on the person shown in the image?',
        'Is the person in the image missing any parts?',
        'Are any of the parts of the person in the picture missing?',
        'Is the person shown in the image incomplete?',
        'Is the person in the photograph missing any body parts?'
        ]
    questions = [
        'Are there any missing parts on the person shown in the image?'
        ]
    questions = ['Are there any missing parts on the person shown in the image? If yes, please answer from \'head\', \'arm\', \'leg\', \'foot\', \'hand\', \'ear\', \'eye\', \'knee\'; otherwise, please answer \'no\'.']
    questions = ['Are there any absent body parts in the person shown in the image? If yes, please answer from \'head\', \'arm\', \'leg\', \'foot\', \'hand\', or \'ear\'; otherwise, please answer \'no\'. Answer the question using a single word:']
    # questions = ['Are there any absent body parts in the person shown in the image? Please answer from \'head\', \'arm\', \'leg\', \'foot\', \'hand\', \'ear\', or  \'no\'. Answer the question using a single word:']
    # counter of folders
    idx_f = 0
    # counter of images
    idx_img = 0
    # loss dict
    # loss_dict = {}
    res = {}
    
    # modify to the none
    for folder in tqdm(os.listdir(val_image_dir)):
        idx_f += 1
        
        image_path = os.path.join(val_image_dir, folder)
        # print(f'processing [{idx_f}/1588]: {folder}')
        res[folder] = {}
        for image in os.listdir(image_path):
            total_acc = 0
            res[folder][image] = {}
            res[folder][image]["acc"] = -1
            res[folder][image]["iou_loss"] = -1
            res[folder][image]["question_ans"] = {}
            image_file = os.path.join(image_path, image)
            # loss_sum = 0
            idx_img += 1
            flag_ori = False # check if the current image is the original image (unmasked)
            # text_list = ['head', 'arm', 'leg', 'foot', 'hand', 'ear', 'eye', 'knee']
            text_list = ['head', 'arm', 'leg', 'foot', 'hand', 'ear']
            # 已经去掉了所有的ori，所以这里就不需要了
            # if image.startswith('ori_image'):
            #     flag_ori = True
            # else:
                
            image_label = find_word_in_string(word_list=text_list, input_string=image)

            for idx, question in enumerate(questions):
                print(question)
                input_ids = prepare_conv(question, model, tokenizer, conv_mode)
                
                
                result = inference(image_file, input_ids, model, image_processor, tokenizer, args)
                # print('here:', result)
            
                found_word = find_word_in_string(word_list=text_list, input_string=result)
                # text = found_word
                res[folder][image]["question_ans"][question] = result
                

                if found_word == image_label:
                    total_acc +=1
                res[folder][image]["acc"] = total_acc/len(questions)
                #save
                # save_dir = os.path.join(output_dir, folder)
                # if not os.path.exists(save_dir):
                #     os.mkdir(save_dir)
                # copied_image.save(os.path.join(save_dir,f'{image}_bbox_q{idx}.jpg'))
                # print(type(copied_image))
            if idx_img % args.output_freq == 0:
                json.dump(res, open(os.path.join(output_dir, 'coco_val.json'), "w"))


if __name__ == "__main__":
    args = parse_args()
    main(args)