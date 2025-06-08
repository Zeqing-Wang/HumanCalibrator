from PIL import Image
import requests
import torch
from transformers import CLIPProcessor, CLIPModel
import json
import os
import glob
import re
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLIPModel.from_pretrained("clip-vit-base-patch32")
model.to(device)
processor = CLIPProcessor.from_pretrained("clip-vit-base-patch32")


def cal_sim_text2image(image_path, text):
    image = Image.open(image_path)
    texts = [text]
    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True, truncation = True).to(device)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image # this is the image-text similarity score
    return logits_per_image[0].item()



# probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
# print('logits:', logits_per_image)
# print('probs:',probs)
def find_max_num_image(directory):
    max_num = 0
    max_file = ""
    for file in os.listdir(directory):
        if re.match(r'Final_image_\d+_.*', file):
            num = re.search(r'Final_image_(\d+)_', file).group(1)
            if int(num) > max_num:
                max_num = int(num)
                max_file = file
    return max_file


if __name__ =='__main__':
    # result folder
    test_data = './pipeline/output_test_all_1016'
    # prompt(text)
    ref_json = './MoldingHuman/cal_scores/human_clip_score_505.json'
    output_folder = './MoldingHuman/metrics/clip_scores'
    # 包含了要测试clip score的数据名称
    # test_data = './MoldingHuman/mqy_test_data_and_tools/test_acc_output_0926/test_2_epoch_without_eye_knee_filter_result_refine.json'
    
    ori_image = 'inpainting_res/ori_image.jpg'
    ours_image = 'inpainting_res/resized_final_img.jpg'
    ours_image = 'inpainting_res/resized_final_img_blur=None_pad=None_guidance=20_enlarged=False_strength=1_seed=92.jpg'
    
    
    # prompt_fiels = ['original_prompt','human_prompt'] # human_prompt & original_prompt
    # prompt_fiels = ['human_prompt']
    json_file = json.load(open(ref_json,"r"))
    
    original_image_original_sum_scores = 0
    processed_image_original_sum_scores = 0
    
    
    original_image_human_sum_scores = 0
    processed_image_human_sum_scores = 0
    
    folders = os.listdir(test_data)
    
    # 这里应该读区test data，并且根据数据目录
    # test_data_samples = json.load(open(test_data,"r"))
    output_res_list = []
    for folder in tqdm(folders):
        # sample_folder_path = test_sample['res_image'].replace('final_res.png','')
        processed_image_path = os.path.join(test_data, folder, ours_image)
        origin_image_path = os.path.join(test_data, folder, ori_image)
        test_sample = {
            'process_image_path' : processed_image_path,
            'origin_image_path' : origin_image_path
        }

        for prompt_sample_key in json_file.keys():
            if prompt_sample_key in folder:
                original_prompt = json_file[prompt_sample_key]['original_prompt']
                human_prompt = json_file[prompt_sample_key]['human_prompt']
    
        sim_score_original_with_original_prompt = cal_sim_text2image(image_path=origin_image_path, text=original_prompt)
        sim_score_original_with_human_prompt = cal_sim_text2image(image_path=origin_image_path, text=human_prompt)
        
        
        sim_score_processed_with_original_prompt = cal_sim_text2image(image_path=processed_image_path, text=original_prompt)
        sim_score_processed_with_human_prompt = cal_sim_text2image(image_path=processed_image_path, text=human_prompt)
        
        
        
        
        
        test_sample['sim_score_original_with_original_prompt'] = sim_score_original_with_original_prompt
        original_image_original_sum_scores = original_image_original_sum_scores + sim_score_original_with_original_prompt
        test_sample['sim_score_processed_with_original_prompt'] = sim_score_processed_with_original_prompt
        processed_image_original_sum_scores = processed_image_original_sum_scores + sim_score_processed_with_original_prompt
        
        
        test_sample['sim_score_original_with_human_prompt'] = sim_score_original_with_human_prompt
        original_image_human_sum_scores = original_image_human_sum_scores + sim_score_original_with_human_prompt
        test_sample['sim_score_processed_with_human_prompt'] = sim_score_processed_with_human_prompt 
        processed_image_human_sum_scores = processed_image_human_sum_scores + sim_score_processed_with_human_prompt
        output_res_list.append(test_sample)
    save_res = {
        'cal_res_list' : output_res_list
    }
    
    save_res['avg_original_image_orginal_prompt_score'] = original_image_original_sum_scores/len(folders)
    save_res['avg_processed_image_original_prompt_score'] = processed_image_original_sum_scores/len(folders)
    
    save_res['avg_original_image_human_prompt_score'] = original_image_human_sum_scores/len(folders)
    save_res['avg_processed_image_human_prompt_score'] = processed_image_human_sum_scores/len(folders)
    
    
    json.dump(save_res, open(os.path.join(output_folder, '1014_404_blur_None_seed_92_guidance=20_enlarged=False_clip_base_32_enlarged=False.json'), "w"))
        
        # pass
    
    
    pass
