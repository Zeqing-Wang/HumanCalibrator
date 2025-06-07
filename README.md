# HumanCalibrator
Official implementation & Dataset for the paper:

**Is this Generated Person Existed in Real-world? Fine-grained Detecting and Calibrating Abnormal Human-body (CVPR 2025)**.

Zeqing Wang, Qingyang Ma, Wentao Wan, Haojie Li, Keze Wang, Yonghong Tian.

<a href='https://arxiv.org/abs/2411.14205'><img src='https://img.shields.io/badge/arXiv-2411.14205-red'></a>  

[🤗Datasets](https://huggingface.co/datasets/NNaptmn/HumanCalibrator)

[![Google Drive](https://img.shields.io/badge/Google%20Drive-4285F4?style=for-the-badge&logo=googledrive&logoColor=white)Training Dataset & Ckpt & Log & Baselines](https://huggingface.co/datasets/NNaptmn/HumanCalibrator)

## Dataset
⚠️ **Some of the images may make you uncomfortable,  especially in the AIGC Human-Aware 1k** (though we have try our best to construst a comfortable while suitable dataset via very hard manually filtering.)

### COCO Human-Aware Val
COCO Human-Aware Val is an automated generation split from the COCO 2017 Val split, with XX images, each image contains an  abnormality **with** the corresponding bounding box.

### AIGC Human-Aware 1K
AIGC Human-Aware 1K is a manually annotated split with 1000 iamges. Due to the uncertainty of anomaly location, we do not annotate the bounding box in the AIGC Human-Aware 1K.


### Training Data
Download from [🤗Hugging Face](https://huggingface.co/datasets/NNaptmn/HumanCalibrator), with the ''

## Environment Preparation
Following the [LLaVa](https://github.com/haotian-liu/LLaVA) , [GroundSamAnything](https://github.com/IDEA-Research/Grounded-Segment-Anything) and the [SD2](https://huggingface.co/stabilityai/stable-diffusion-2).


## Results
We provide all the results (with the middle results and the baselines) in XXX. In detail the results can be categoried in the following three types:

### HumanCalibrator on AIGC Human-Aware
### Other Models on AIGC Human-Aware
For the Closed-sorce VLMs (like GPT4o)


### AHD on COCO Human-aware
### Other Models on COCO Human-aware


## CKPT
The ckpt and the training log of our AHD, trained based on LLaVa1.5 7B, can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1E2l9fRQq9l5YjPJnVCAnrwNQJv9uFVZ0?usp=sharing) (**llava-v1.5-7b-task-train_ver_with_neg_inpainting_1721922413_0392356_2_epoch.tar.gz**). It can be inference with the LLaVa codebase.

## Run Script


## **Citation**

If you find the repo useful for your work, please star this repo and consider citing:

```
@article{wang2024generated,
  title={Is this Generated Person Existed in Real-world? Fine-grained Detecting and Calibrating Abnormal Human-body},
  author={Wang, Zeqing and Ma, Qingyang and Wan, Wentao and Li, Haojie and Wang, Keze and Tian, Yonghong},
  journal={arXiv preprint arXiv:2411.14205},
  year={2024}
}
```