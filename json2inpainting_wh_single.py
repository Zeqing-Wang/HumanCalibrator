import json
import os
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image,ImageDraw
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
import cv2
# NMS变体
def merge_overlapping_boxes(boxes, types, classes, ori_boxes, iou_threshold=0.5):
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

    # groups = {}
    # for box, t, c in zip(boxes, types, classes):
    #     key = (t, c)
    #     if key not in groups:
    #         groups[key] = []
    #     groups[key].append(box)

    # merged_boxes = []
    # merged_types = []
    # merged_classes = []

    # for (t, c), group in groups.items():
    #     while group:
    #         base = group.pop(0)
    #         i = 0
    #         while i < len(group):
    #             if calculate_iou(base, group[i]) > iou_threshold:
    #                 base = merge_boxes(base, group.pop(i))
    #             else:
    #                 i += 1
    #         merged_boxes.append(base)
    #         merged_types.append(t)
    #         merged_classes.append(c)

    # return merged_boxes, merged_types, merged_classes

    # 将具有相同type和class的bbox分组
    groups = {}
    ori_box_mapping = {}
    for idx, (box, t, c) in enumerate(zip(boxes, types, classes)):
        key = (t, c)
        if key not in groups:
            groups[key] = []
            ori_box_mapping[key] = []
        groups[key].append(box)
        ori_box_mapping[key].append(ori_boxes[idx])

    merged_boxes = []
    merged_types = []
    merged_classes = []
    merged_ori_boxes = []  # 新增ori_boxes对应项

    for (t, c), group in groups.items():
        ori_group = ori_box_mapping[(t, c)]
        while group:
            base = group.pop(0)
            ori_base = ori_group.pop(0)  # 跟踪对应的ori_box
            i = 0
            while i < len(group):
                if calculate_iou(base, group[i]) > iou_threshold:
                    base = merge_boxes(base, group.pop(i))
                    ori_base = merge_boxes(ori_base, ori_group.pop(i))  # 更新ori_base
                else:
                    i += 1
            merged_boxes.append(base)
            merged_types.append(t)
            merged_classes.append(c)
            merged_ori_boxes.append(ori_base)

    return merged_boxes, merged_types, merged_classes, merged_ori_boxes  # 返回新增的ori_boxes



def process_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    
    # print('before nms len: ', len(data))
    nms_boxes = []
    nms_conditions = []
    nms_labels = []
    ori_boxes = []
    for r in data:
        # print(r)
        # assert False
        nms_boxes.append(r['enlarged_position'])
        nms_conditions.append(r['condition'])
        nms_labels.append(r['label'])
        ori_boxes.append(r['position'])
    
    merged_boxes, merged_conditions, merged_labels, merged_ori_boxes = merge_overlapping_boxes(nms_boxes, nms_conditions, nms_labels, ori_boxes, iou_threshold=0.3 )
    
    merged_len = len(merged_boxes)
    ori_merged_len = len(merged_ori_boxes)
    # print('merged_and_ori len:', merged_len, ori_merged_len)
    merged_box = []
    for i in range(merged_len):
        res_dict = {
            'position':merged_ori_boxes[i],
            'label':merged_labels[i],
            'condition':merged_conditions[i],
            'enlarged_position':merged_boxes[i]
        }
        merged_box.append(res_dict)
    
    data = merged_box
    # print('after nms len: ', len(data))
    # assert False
    
    # mask_list = []
    # for d in data:
    #     pred = d['label']
    #     mask_dict = {
    #         'condition':d['condition'],
    #         'label':d['label'],
    #         'position':d['position']
    #     }
    #     mask_list.append(mask_dict)
    
    # print(mask_list)
    # print(merged_box)
    # assert False
    return data
    pass

def gen_mask(pos, size):
    mask_image = Image.new('RGB', size, (0, 0, 0))
    draw = ImageDraw.Draw(mask_image)
    draw.rectangle(pos, fill="white")
    return mask_image
    pass

def get_prompt(phrase, condition):
    prompt = f"{phrase}, masterpiece, best quality, high quality ultrarealistic"
    negative_prompt = f"bad {phrase}, no {phrase}, monster, nonhuman, digital, machine, cartoon style, comic, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username"
    if condition == 'deprivative':
        return prompt, negative_prompt
    pass
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

def get_SD_prompt_redundant_humanbodypart(phrase):
    prompt = 'content that fits with the surrounding human bodies.'
    neg_dict = {
        "ear": "a human ear",
        "head": "a human head",
        "leg": "a human leg",
        "foot": "a human foot",
        "hand": "a human hand",
        "arm": "a human arm"
    }
    return prompt, neg_dict[phrase]

def inpainting(src_image, mask, pred, index, output_path, seed, condition, task, blur_factor=None,guidance_scale=7.5, pad_mask_crop=None, strength=1, width=1088, height=640, apply_overlay=True):
    if condition == 'redundant':
        prompt, negative_prompt = get_SD_prompt_redundant_humanbodypart(pred)
    elif condition == 'deprivative':
        prompt, negative_prompt = get_SD_prompt(pred)   
        # prompt, negative_prompt = get_prompt(pred, condition)
        

    image = src_image.copy()
    original_size = image.size
    
    image = image.convert("RGB").resize((512, 512))  
    # blurred_mask = pipe.mask_processor.blur(mask, blur_factor=10)
    # mask.save(os.path.join(output_path, f'mask_{index}_before_resizing.png'))

    mask_image = mask.copy().convert("L").resize((512, 512))
    mask_image.save(os.path.join(output_path, f'mask_{task}_{index}.png'))
    if blur_factor is not None:
        blurred_mask = pipe.mask_processor.blur(mask_image, blur_factor)
        blurred_mask.save(os.path.join(output_path, f'blurred_mask_{task}_{index}.png'))
        mask_image = blurred_mask

    
    generator = torch.Generator(device="cuda").manual_seed(seed)  
    # Perform inpainting using the pipeline
    result_image = pipe(
        width=width, 
        height=height,
        prompt=prompt, 
        negative_prompt=negative_prompt, 
        generator=generator,
        image=image, 
        mask_image=mask_image,
        guidance_scale=guidance_scale,
        pad_mask_crop=pad_mask_crop,
        strength=strength
    ).images[0]

    # Resize the result back to the original size
    # result_image = result_image.resize(original_size, Image.LANCZOS)
    '''
    overlaying the inpainting part to the source image
    '''
    # Save the inpainted image
    temp_path = os.path.join(output_path, f'{index}_{pred}_{task}.png')
    print(f'Saving inpainting image {temp_path}')
    result_image.save(temp_path)
    
    if apply_overlay:
        unmasked_unchanged_image = pipe.image_processor.apply_overlay(mask, image, result_image)
    else:
        unmasked_unchanged_image = result_image.copy()
    unmasked_unchanged_image.save(os.path.join(output_path, f'{index}_{pred}_{task}_unchanged.png'))
    # assert False

    return unmasked_unchanged_image

def load_4x_super_res():
    model_path = './Real-ESRGAN/weights/RealESRGAN_x4plus.pth'
    if 'x4plus' in model_path:
        outscale = 4
        netscale = 4
    dni_weight = [0.5, 0.5]
    tile = 0
    tile_pad = 10
    pre_pad = 0
    
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    
    upsampler = RealESRGANer(
    scale=netscale,
    model_path=model_path,
    dni_weight=dni_weight,
    model=model,
    tile=tile,
    tile_pad=tile_pad,
    pre_pad=pre_pad,
    half=False,
    gpu_id=0)
    
    return upsampler

def single_image_super_resolution(upsampler, img, outscale):
    output, _ = upsampler.enhance(img, outscale=4)
    return output
    pass

if __name__=='__main__':
        
    testing_res_path = './MoldingHuman/data_aigc/pipline_res/new_inpaint_without_SR'

    # ./MoldingHuman/data_aigc_redundant/images/pika-d5939810-6152-5e92-849f-55b3785568ec_6_lack_arm_sub_lack_hand_redundant_arm_sub_redundant_None.jpg
    # output_path = './pipeline/output_test_sel_1024'
    output_path = testing_res_path 
    
    # width = 1088
    # height = 640
    blur = None
    pad = None
    guidance = 7.5
    enlarged = True
    strength = 1
    seed = 92

    task = f'blur={blur}_pad={pad}_guidance={guidance}_enlarged={enlarged}_strength={strength}_seed={seed}_with_wh'
    
    # load model
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "/home/keze/kz/model/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16,
    )
    pipe.to("cuda")
    sel_list = None # 测全部
    # sel_list = ['0','22','29','32','44','73','76','84','85','96','97']
    # sel_list = ['58','92']
    
    
    
    # load model
    upsampler = load_4x_super_res()
    outscale = 4
    
    cnt = 0
    folders = os.listdir(testing_res_path)
    import random
    random.shuffle(folders)
    for folder in tqdm(folders):
        # 强烈谴责 pika-e29918cb-562d-559b-9098-c159dc8ece04_1
        
        if sel_list is not None and folder not in sel_list:
            pass
        else:
            # folder_path = os.path.join(testing_res_path, folder, 'pipeline_res_1106_maxLoop_20_redunTres_0.38_4', '1')
            folder_path = os.path.join(testing_res_path, '1')
            
            json_path = os.path.join(folder_path, 'result_final.json')
            # print(json_path)
            src_img_path = os.path.join(folder_path, 'redundant_res', 'grounding_ori_res', 'ori_image.jpg')
            check_inpainting_path = os.path.join(output_path, folder, 'inpainting_res')
            output_folder_path = os.path.join(output_path, folder, 'inpainting_res', task)
            final_path = os.path.join(output_folder_path, f'resized_final_img_{task}.jpg')
            final_sr_path = os.path.join(output_folder_path, f'resized_final_img_{task}_with_sr.jpg')
            # if os.path.exists(check_inpainting_path):
            if os.path.exists(final_sr_path):
                print('skip', folder)
                continue
            
            mask_list = process_json(json_path)

            os.makedirs(output_folder_path, exist_ok=True)
            cnt += 1
            src_img = Image.open(src_img_path)
            image_size = src_img.size
            width = image_size[0]
            height = image_size[1]
            src_img.save(os.path.join(output_folder_path, 'ori_image.jpg'))

            original_size = src_img.size
            inpainting_img = src_img.copy()
            res_img = src_img.copy()
            unmasked_unchanged_image = src_img.copy()

            index = 1       
            for pred_dict in mask_list:
                if enlarged:
                    pos = pred_dict['enlarged_position']
                else:
                    pos = pred_dict['position'] 
                
                pred = pred_dict['label']
                condition = pred_dict['condition']
                mask = gen_mask(pos, original_size)

                # Draw bbox
                output_image = os.path.join(output_folder_path, 'final_res.jpg')
                draw = ImageDraw.Draw(res_img)
                bbox_list = pos
                text = condition + '_' + pred

                if condition == 'redundant':
                    draw.rectangle(bbox_list, outline='red', width=2)
                    text_position = (bbox_list[0]+2, bbox_list[1])
                    draw.text(text_position, text, font=None, fill="red")
                else:
                    draw.rectangle(bbox_list, outline='blue', width=2)
                    text_position = (bbox_list[0]+2, bbox_list[1])
                    draw.text(text_position, text, font=None, fill="blue")
                
                res_img.save(output_image)

                unmasked_unchanged_image = inpainting(inpainting_img, mask, pred, index, output_folder_path, seed, condition, task, blur_factor=blur, guidance_scale=guidance, pad_mask_crop=pad, strength=strength, width=width, height=height, apply_overlay=False)
                index = index + 1
                inpainting_img = unmasked_unchanged_image
            
            final_image = inpainting_img
            final_path_without_sr = os.path.join(output_folder_path, f'resized_final_img_{task}_without_sr.jpg')
            # final_path = os.path.join(output_folder_path, f'resized_final_img_{task}.jpg')
            final_image.save(final_path_without_sr)
            print(f'Saving to {final_path}')
            
            img = cv2.imread(final_path_without_sr)
            
            
            output = single_image_super_resolution(upsampler=upsampler, img=img, outscale=outscale)
            final_path_with_sr = os.path.join(output_folder_path, f'resized_final_img_{task}_with_sr.jpg')
            cv2.imwrite(final_path_with_sr, output)
