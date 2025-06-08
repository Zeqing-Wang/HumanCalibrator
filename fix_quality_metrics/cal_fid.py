import torch
import torchvision
import torchvision.transforms as transforms
from pytorch_fid import fid_score

# 准备真实数据分布和生成模型的图像数据
# 计算pose condition与原图的FID
generated_images_folder = './MoldingHuman/data_aigc/pose_condition'
generated_images_folder = './MoldingHuman/data_aigc/pure_repaired_images'
real_images_folder = './MoldingHuman/data_aigc/pure_ori_images'

# 加载预训练的Inception-v3模型
inception_model = torchvision.models.inception_v3(pretrained=True)

# 定义图像变换
transform = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 计算FID距离值
# fid_value = fid_score.calculate_fid_given_paths([real_images_folder, generated_images_folder],
#                                                  inception_model,
#                                                  # transform=transform
#                                                  )
fid_value = fid_score.calculate_fid_given_paths(
paths=[real_images_folder, generated_images_folder],
batch_size=1,
device="cuda:14",
dims=2048,
)
print('FID value:', fid_value)

