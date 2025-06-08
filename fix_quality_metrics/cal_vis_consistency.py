import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLIPModel.from_pretrained("clip-vit-large-patch14-336")
model.to(device)
processor = CLIPProcessor.from_pretrained("clip-vit-large-patch14-336")

def load_and_preprocess_image(image_path):
    image = Image.open(image_path)
    return processor(images=image, return_tensors="pt")

def clip_img_score(img1_path, img2_path):
    image_a = load_and_preprocess_image(img1_path)["pixel_values"].to(device)
    image_b = load_and_preprocess_image(img2_path)["pixel_values"].to(device)
    with torch.no_grad():
        embedding_a = model.get_image_features(image_a)
        embedding_b = model.get_image_features(image_b)

    similarity_score = torch.nn.functional.cosine_similarity(embedding_a, embedding_b)
    return similarity_score.item()


if __name__ == '__main__':
    # 包含了要测试clip score的数据名称
    test_data = './results_aigc'
    pose_condition_image = 'original_prompt_sd14_mmpose.png'
    ori_image = 'inpainting_res/ori_image.jpg'
    ours_image = 'inpainting_res/resized_final_img.jpg'
    folders = os.listdir(test_data)
    
    ori_with_pose_sim_score_total = 0
    ori_with_ours_sim_score_total = 0
    for folder in tqdm(folders):
        origin_image_path = os.path.join(test_data, folder, ori_image)
        pose_condition_image_path = os.path.join(test_data, folder, pose_condition_image)
        ours_image_path = os.path.join(test_data, folder, ours_image)
        
        ori_with_pose_sim_score = clip_img_score(origin_image_path, pose_condition_image_path)
        ori_with_ours_sim_score = clip_img_score(origin_image_path, ours_image_path)
        
        ori_with_pose_sim_score_total = ori_with_pose_sim_score_total + ori_with_pose_sim_score
        ori_with_ours_sim_score_total =ori_with_ours_sim_score_total + ori_with_ours_sim_score
        
        
        pass
    
    print('pose condition:', ori_with_pose_sim_score_total/len(folders))
    print('ours:',ori_with_ours_sim_score_total/len(folders))

    # json_file = json.load(open(ref_json,"r"))
    
    # original_image_original_sum_scores = 0
    # processed_image_original_sum_scores = 0
    
    
    # original_image_human_sum_scores = 0
    # processed_image_human_sum_scores = 0