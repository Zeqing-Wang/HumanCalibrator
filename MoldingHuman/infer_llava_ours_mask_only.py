import argparse
import torch
from llava.constants import (IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN,
                             DEFAULT_IM_END_TOKEN, IMAGE_PLACEHOLDER, )
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from tqdm import tqdm
from PIL import Image
from io import BytesIO
import requests
import json
import re
import os
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image,ImageDraw
import torch


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--key_frame_dir", type=str, required=False, help="Directory containing extracted keyframes.",
                        default="data/pika-0a5d714a-3feb-54e0-88fa-60671605b61c/frame_ori")
    parser.add_argument("--output_dir", type=str, required=False, default='/home/keze/kz/code/VideoHumanPlausibility/MoldingHuman/mqy_test_image/pika-d5939810-6152-5e92-849f-55b3785568ec_6_lack_arm_sub_lack_hand_redundant_arm_sub_redundant_None.jpg/res',
                        help="Directory to save output files.")
    parser.add_argument("--question", type=str, default="Describe the image in detail.",
                        help="Question to ask about the image.")

    parser.add_argument("--model-path", type=str, required=False, help="Path to the pretrained model.",
                        default="/home/keze/kz/code/VideoHumanPlausibility/LLaVA/checkpoints/llava-v1.5-7b-task-train_only_mask_various_ver_with_round_3_same_qa_deepcopied_with_neg_0801_1721972267.4192767.json/checkpoint-10500")
    parser.add_argument("--model-base", type=str, default=None, help="Base model to use.")
    parser.add_argument("--conv-mode", type=str, default=None, help="Conversation mode.")
    parser.add_argument("--sep", type=str, default=",", help="Separator.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature for sampling.")
    parser.add_argument("--top_p", type=float, default=None, help="Top-p sampling parameter.")
    parser.add_argument("--num_beams", type=int, default=1, help="Number of beams for beam search.")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum number of new tokens to generate.")

    return parser.parse_args()


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
        # print('wzqwzq------image_file:',image_file,type(image_file))
        image_file = str(image_file)
        image = load_image(image_file)
        out.append(image)
    return out


def load_model_llava(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )

    if "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"

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


def inference(image_files, input_ids, model, image_processor, tokenizer, args):
    images = load_images(image_files)
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    with torch.inference_mode():
        # print("model_kwargs:", model_kwargs)
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

def find_word_in_string(word_list, input_string):
    for word in word_list:
        if word in input_string:
            return word  
    return None 

def main(args):
    model, image_processor, tokenizer, conv_mode = load_model_llava(args)
    question = 'Is the person in the photograph missing any body parts?'
    text_list = ['head', 'arm', 'leg', 'foot', 'hand', 'ear', 'eye', 'knee']
    count = 0

    image_path = '/home/keze/kz/code/VideoHumanPlausibility/MoldingHuman/mqy_test_image/pika-d5939810-6152-5e92-849f-55b3785568ec_6_lack_arm_sub_lack_hand_redundant_arm_sub_redundant_None.jpg/mask_hand_424.30133056640625_574.7677001953125_509.490234375_613.081298828125_5.png'
    file_name = image_path.split('/')[-1]
    output_dir = '/home/keze/kz/code/VideoHumanPlausibility/MoldingHuman/mqy_test_image/pika-d5939810-6152-5e92-849f-55b3785568ec_6_lack_arm_sub_lack_hand_redundant_arm_sub_redundant_None.jpg/res'
    output_dir = os.path.join(output_dir, file_name)
    os.makedirs(output_dir, exist_ok = True)

    # save original image
    image = Image.open(image_path).convert("RGB")
    word = find_word_in_string(text_list, file_name)
    img_path = os.path.join(output_dir, f'original_{word}.png')
    image.save(img_path)

    results = []
    json_output_path = os.path.join(output_dir, "results.json")
    if os.path.exists(json_output_path):
        with open(json_output_path, "r") as json_file:
            results = json.load(json_file)
    result_cur = []
    temp_path = image_path
    while True:
        input_ids = prepare_conv(question, model, tokenizer, conv_mode)
        image_files = [temp_path]
        result = inference(image_files, input_ids, model, image_processor, tokenizer, args)
        print(f'Question:{question}')
        print(result)
        count += 1
        find_word = find_word_in_string(text_list,result)
        prompt = None
        negative_prompt = None
        if find_word is not None:
            matches = re.findall(r'\[(.*?)\]', result)
            list_str = matches[0] if matches else ''
            number_list = list(map(float, list_str.split(',')))
            print('RE Res:',number_list)

            # get bbox coordinates
            ori_image = Image.open(temp_path)
            image_width, image_height = ori_image.size
            box = [
                number_list[0]*image_width,
                number_list[1]*image_height,
                number_list[2]*image_width,
                number_list[3]*image_height
                ]
            # enlarge the size of inpainting mask
            box_width = box[2]-box[0]
            box_height = box[3]-box[1]
            proportion = 0.5
            bigger_box = [
                max(box[0] - proportion * box_width, 0),
                max(box[1] - proportion * box_height, 0),
                min(box[2] + proportion * box_width, image_width),
                min(box[3] + proportion * box_height, image_height)

            ]
            # generate image with bbox
            bbox_image = ori_image.copy()
            draw_bbox = ImageDraw.Draw(bbox_image)
            draw_bbox.rectangle(box, outline='red', width=2)
            text_position = (box[0]+2, box[1])
            draw_bbox.text(text_position, find_word, font=None, fill="red")
            output_image_path = os.path.join(
                output_dir, 
                f'output_{find_word}_{box[0]}_{box[1]}_{box[2]}_{box[3]}_{count}.png'
            )
            bbox_image.save(output_image_path)

            # generate corresponding mask
            mask_image = Image.new('RGB', (image_width, image_height), (0, 0, 0))
            draw = ImageDraw.Draw(mask_image)
            draw.rectangle(bigger_box, fill="white")
            output_image_path = os.path.join(
                output_dir, 
                f'img_mask_{find_word}_{bigger_box[0]}_{bigger_box[1]}_{bigger_box[2]}_{bigger_box[3]}_{count}.png'
            )
            mask_image.save(output_image_path)
            print(f'Saving {output_image_path}')

            # regenerate the corresponding area via Stable Diffusion Model
            prompt = f"a human {find_word}, masterpiece, best quality, high quality ultrarealistic"
            mask_path = output_image_path
            negative_prompt = f"bad {find_word}"
            print(f'Prompt:{prompt}')
            print(f'Negative prompt:{negative_prompt}')

            # 打开图片和mask图片
            image = Image.open(temp_path).convert("RGB")  # 确保图片是RGB格式
            mask_image = Image.open(mask_path).convert("L")  # Mask应该是灰度格式
            original_size = image.size
            image = pipe(prompt=prompt, negative_prompt=negative_prompt, image=image, mask_image=mask_image).images[0]
            image = image.resize(original_size, Image.LANCZOS)
            temp_path = os.path.join(output_dir, f'inpainting_bigger_mask_{find_word}_{count}.png')
            # image.save(image_path)
            image.save(temp_path)

        else:
            break
        result_entry = {
            "question": question,
            "answer": result,
            "find_word": find_word,
            "count": count,
            "file_path" : temp_path,
            "prompt" : prompt,
            "negative_prompt" : negative_prompt
        }
        result_cur.append(result_entry)
        if count > 3:
            break
        # 将结果写入 JSON 文件
    results.append(result_cur)
    json_output_path = os.path.join(output_dir, "results.json")
    with open(json_output_path, "w") as json_file:
        json.dump(results, json_file, indent=4)
    print(f'Results saved to {json_output_path}')

        


if __name__ == "__main__":
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "/mnt/SSD4_7T/keze/kz/model/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16,
    )
    pipe.to("cuda")
    args = parse_args()
    main(args)