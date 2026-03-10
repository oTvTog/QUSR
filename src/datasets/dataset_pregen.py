#!/usr/bin/env python3
"""
使用预生成LR图像的数据集类，节省训练时的内存
"""

import os
import random
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
from pathlib import Path

class PairedRandomTransform:
    """确保LR和HR图像使用相同随机参数的变换类"""
    def __init__(self, crop_size, target_size, flip_prob=0.5, use_consistent_crop=False, 
                 consistent_crop_images=None):
        self.crop_size = crop_size
        self.target_size = target_size
        self.flip_prob = flip_prob
        self.use_consistent_crop = use_consistent_crop
        self.consistent_crop_images = set(consistent_crop_images) if consistent_crop_images else set()
    
    def __call__(self, lr_img, hr_img, image_name=None):
        lr_w, lr_h = lr_img.size
        hr_w, hr_h = hr_img.size
        
        if (self.use_consistent_crop and image_name is not None and 
            (not self.consistent_crop_images or image_name in self.consistent_crop_images)):
            
            current_state = random.getstate()
            
            seed = hash(image_name) % (2**32)
            random.seed(seed)
            
            max_left = max(0, min(lr_w, hr_w) - self.crop_size)
            max_top = max(0, min(lr_h, hr_h) - self.crop_size)
            
            if max_left > 0 and max_top > 0:
                crop_left = random.randint(0, max_left)
                crop_top = random.randint(0, max_top)
            else:
                crop_left = max(0, (min(lr_w, hr_w) - self.crop_size) // 2)
                crop_top = max(0, (min(lr_h, hr_h) - self.crop_size) // 2)
            
            random.setstate(current_state)
        else:
            crop_left = random.randint(0, max(0, min(lr_w, hr_w) - self.crop_size))
            crop_top = random.randint(0, max(0, min(lr_h, hr_h) - self.crop_size))
        
        if lr_w >= self.crop_size and lr_h >= self.crop_size:
            lr_img = lr_img.crop((crop_left, crop_top, crop_left + self.crop_size, crop_top + self.crop_size))
        else:
            lr_img = lr_img.resize((self.crop_size, self.crop_size), Image.LANCZOS)
            
        if hr_w >= self.crop_size and hr_h >= self.crop_size:
            hr_img = hr_img.crop((crop_left, crop_top, crop_left + self.crop_size, crop_top + self.crop_size))
        else:
            hr_img = hr_img.resize((self.crop_size, self.crop_size), Image.LANCZOS)
        
        should_flip = random.random() < self.flip_prob
        if should_flip:
            lr_img = lr_img.transpose(Image.FLIP_LEFT_RIGHT)
            hr_img = hr_img.transpose(Image.FLIP_LEFT_RIGHT)
        
        lr_img = lr_img.resize((self.target_size, self.target_size), Image.LANCZOS)
        hr_img = hr_img.resize((self.target_size, self.target_size), Image.LANCZOS)
        
        return lr_img, hr_img

class PairedSRPreGenDataset(torch.utils.data.Dataset):
    def __init__(self, split=None, args=None):
        super().__init__()

        self.args = args
        self.split = split
        
        # 添加质量提示选择参数
        self.gt_quality_prompt_ratio = getattr(args, 'gt_quality_prompt_ratio', 0.0)
        self.use_gt_quality_prompt = getattr(args, 'use_gt_quality_prompt', False)
        
        # 质量提示路径
        self.quality_prompt_path = getattr(args, 'quality_prompt_path', 'preset/lowlevel_prompt_q')
        self.quality_prompt_gt_path = getattr(args, 'quality_prompt_gt_path', 'preset/lowlevel_prompt_q_GT')
        
        if split == 'train':
            self.lr_dir = getattr(args, 'pre_generated_lr_dir', 'preset/pre_generated_lr')
            
            with open(args.dataset_txt_paths, 'r') as f:
                self.gt_list = [line.strip() for line in f.readlines()]
            
            if args.highquality_dataset_txt_paths is not None:
                with open(args.highquality_dataset_txt_paths, 'r') as f:
                    self.hq_gt_list = [line.strip() for line in f.readlines()]
            
            if os.path.exists(self.lr_dir):
                print(f"训练集使用预生成的全尺寸LR图像: {self.lr_dir}")
                print(f"训练集使用原始GT图像作为HR: {len(self.gt_list)} 个")
            else:
                print(f"警告：全尺寸预生成LR目录不存在: {self.lr_dir}")
                print("请先运行 python pre_generate_lr.py 生成全尺寸LR图像")
            
            consistent_crop_images = getattr(args, 'consistent_crop_images_list', None)
            
            self.paired_transform = PairedRandomTransform(
                crop_size=args.resolution_ori,
                target_size=args.resolution_tgt,
                flip_prob=0.5,
                use_consistent_crop=getattr(args, 'use_consistent_crop', False),
                consistent_crop_images=consistent_crop_images
            )

        elif split == 'test':
            self.input_folder = os.path.join(args.dataset_test_folder, "test_SR_bicubic")
            self.output_folder = os.path.join(args.dataset_test_folder, "test_HR")
            self.lr_list = []
            self.gt_list = []
            
            lr_names = sorted(os.listdir(os.path.join(self.input_folder)))
            
            for lr_name in lr_names:
                if lr_name.endswith('.png') or lr_name.endswith('.jpg') or lr_name.endswith('.jpeg'):
                    base_name = os.path.splitext(lr_name)[0]
                    
                    possible_hr_names = [
                        f"{base_name}_gt.png",
                        f"{base_name}_gt.jpg",
                        f"{base_name}_gt.jpeg",
                        f"{base_name}.png", # 尝试与LR同名
                        f"{base_name}.jpg",
                        f"{base_name}.jpeg"
                    ]
                    
                    found_hr_path = None
                    for hr_name in possible_hr_names:
                        hr_path = os.path.join(self.output_folder, hr_name)
                        if os.path.exists(hr_path):
                            found_hr_path = hr_path
                            break
                    
                    if found_hr_path:
                        self.lr_list.append(os.path.join(self.input_folder, lr_name))
                        self.gt_list.append(found_hr_path)
                    else:
                        print(f"Warning: HR file not found for {lr_name}, expected one of {possible_hr_names}")
            
            print(f"Found {len(self.lr_list)} LR-HR pairs for validation")
            assert len(self.lr_list) == len(self.gt_list)
            
            val_transforms = []
            if hasattr(args, 'resolution_ori') and args.resolution_ori:
                val_transforms.append(transforms.CenterCrop((args.resolution_ori, args.resolution_ori)))
            val_transforms.append(transforms.Resize((args.resolution_tgt, args.resolution_tgt)))
            self.val_transform = transforms.Compose(val_transforms)

    def __len__(self):
        return len(self.gt_list)

    def get_quality_prompt(self, image_name_or_index, use_gt=False):
        """
        Get quality prompt text for a given image.
        
        Args:
            image_name_or_index: Image filename (without extension) or index
            use_gt: Whether to use GT quality prompts (more detailed)
            
        Returns:
            Quality prompt text string
        """
        try:
            # Determine base path
            base_path = self.quality_prompt_gt_path if use_gt else self.quality_prompt_path
            
            # Handle both filename and index inputs
            if isinstance(image_name_or_index, (int, str)) and str(image_name_or_index).isdigit():
                # Numeric index - format to match file naming
                prompt_file = f"{int(image_name_or_index):05d}.txt"
            else:
                # String filename - ensure .txt extension
                prompt_file = str(image_name_or_index)
                if not prompt_file.endswith('.txt'):
                    prompt_file += '.txt'
            
            prompt_path = os.path.join(base_path, prompt_file)
            
            # Read quality prompt
            if os.path.exists(prompt_path):
                with open(prompt_path, 'r', encoding='utf-8') as f:
                    quality_prompt = f.read().strip()
                return quality_prompt
            else:
                # Fallback to default quality prompt
                return "This image shows average quality with some details visible but could be improved."
                
        except Exception as e:
            return "This image shows average quality with some details visible but could be improved."

    def __getitem__(self, idx):
        if self.split == 'train':
            # 选择GT图像（用作HR）
            if hasattr(self, 'hq_gt_list') and self.hq_gt_list:
                if np.random.uniform() < self.args.prob:
                    gt_img_path = self.gt_list[idx]
                else:
                    idx = random.sample(range(0, len(self.hq_gt_list)), 1)
                    gt_img_path = self.hq_gt_list[idx[0]]
            else:
                gt_img_path = self.gt_list[idx]
            
            hr_img = Image.open(gt_img_path).convert('RGB')
            
            img_name = os.path.basename(gt_img_path)
            lr_path = os.path.join(self.lr_dir, img_name)
            
            if os.path.exists(lr_path):
                lr_img = Image.open(lr_path).convert('RGB')
            else:
                print(f"警告: 预生成的LR图像不存在 {lr_path}，使用原始图像")
                lr_img = hr_img  # 使用原始图像作为fallback
            
            img_name = os.path.basename(gt_img_path)
            lr_img, gt_img = self.paired_transform(lr_img, hr_img, img_name)

            img_t = F.to_tensor(lr_img)
            img_t = F.normalize(img_t, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            
            output_t = F.to_tensor(gt_img)
            output_t = F.normalize(output_t, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

            example = {}
            example["neg_prompt"] = self.args.neg_prompt_csd
            example["null_prompt"] = ""
            example["output_pixel_values"] = output_t
            example["conditioning_pixel_values"] = img_t

            use_gt_quality = False  
            
            # 获取LR质量提示
            quality_prompt = self.get_quality_prompt(idx, use_gt=use_gt_quality)

            example["quality_prompts"] = [quality_prompt]

            return example
            
        elif self.split == 'test':
            input_img = Image.open(self.lr_list[idx]).convert('RGB')
            output_img = Image.open(self.gt_list[idx]).convert('RGB')
            
            img_t = self.val_transform(input_img)
            output_t = self.val_transform(output_img)
            
            img_t = F.to_tensor(img_t)
            img_t = F.normalize(img_t, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            output_t = F.to_tensor(output_t)
            output_t = F.normalize(output_t, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

            example = {}
            example["neg_prompt"] = self.args.neg_prompt_csd
            example["null_prompt"] = ""
            example["output_pixel_values"] = output_t
            example["conditioning_pixel_values"] = img_t
            example["base_name"] = os.path.basename(self.lr_list[idx])

            return example 