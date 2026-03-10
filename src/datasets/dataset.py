import os
import random
import torch
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
from pathlib import Path

import numpy as np
from src.datasets.realesrgan import RealESRGAN_degradation

class PairedSROnlineTxtDataset(torch.utils.data.Dataset):
    def __init__(self, split=None, args=None):
        super().__init__()

        self.args = args
        self.split = split
        if split == 'train':
            self.degradation = RealESRGAN_degradation(args.deg_file_path, device='cpu')
            self.crop_preproc = transforms.Compose([
                transforms.RandomCrop((args.resolution_ori, args.resolution_ori)),
                transforms.Resize((args.resolution_tgt, args.resolution_tgt)),
                transforms.RandomHorizontalFlip(),
            ])
            with open(args.dataset_txt_paths, 'r') as f:
                self.gt_list = [line.strip() for line in f.readlines()]
            if args.highquality_dataset_txt_paths is not None:
                with open(args.highquality_dataset_txt_paths, 'r') as f:
                    self.hq_gt_list = [line.strip() for line in f.readlines()]

        elif split == 'test':
            self.input_folder = os.path.join(args.dataset_test_folder, "test_SR_bicubic")
            self.output_folder = os.path.join(args.dataset_test_folder, "test_HR")
            self.lr_list = []
            self.gt_list = []
            
            lr_names = sorted(os.listdir(os.path.join(self.input_folder)))
            
            for lr_name in lr_names:
                if lr_name.endswith('.png') or lr_name.endswith('.jpg') or lr_name.endswith('.jpeg'):
                    base_name = lr_name.replace('.png', '').replace('.jpg', '').replace('.jpeg', '')
                    
                    hr_name = base_name + '_gt.png'
                    if not os.path.exists(os.path.join(self.output_folder, hr_name)):
                        hr_name = base_name + '_gt.jpg'
                    if not os.path.exists(os.path.join(self.output_folder, hr_name)):
                        hr_name = base_name + '_gt.jpeg'
                    
                    hr_path = os.path.join(self.output_folder, hr_name)
                    if os.path.exists(hr_path):
                        self.lr_list.append(os.path.join(self.input_folder, lr_name))
                        self.gt_list.append(hr_path)
                    else:
                        print(f"Warning: HR file not found for {lr_name}, expected {hr_name}")
            
            self.crop_preproc = transforms.Compose([
                transforms.RandomCrop((args.resolution_ori, args.resolution_ori)),
                transforms.Resize((args.resolution_tgt, args.resolution_tgt)),
            ])
            
            print(f"Found {len(self.lr_list)} LR-HR pairs for validation")
            assert len(self.lr_list) == len(self.gt_list)

    def __len__(self):
        return len(self.gt_list)

    def __getitem__(self, idx):

        if self.split == 'train':
            if self.args.highquality_dataset_txt_paths is not None:
                if np.random.uniform() < self.args.prob:
                    gt_img = Image.open(self.gt_list[idx]).convert('RGB')
                else:
                    idx = random.sample(range(0, len(self.hq_gt_list)), 1)
                    gt_img = Image.open(self.hq_gt_list[idx[0]]).convert('RGB')
            else:
                gt_img = Image.open(self.gt_list[idx]).convert('RGB')
            gt_img = self.crop_preproc(gt_img)

            output_t, img_t = self.degradation.degrade_process(np.asarray(gt_img)/255., resize_bak=True)
            output_t, img_t = output_t.squeeze(0), img_t.squeeze(0)

            # input images scaled to -1,1
            img_t = F.normalize(img_t, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            # output images scaled to -1,1
            output_t = F.normalize(output_t, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

            example = {}
            # example["prompt"] = caption
            example["neg_prompt"] = self.args.neg_prompt_csd
            example["null_prompt"] = ""
            example["output_pixel_values"] = output_t
            example["conditioning_pixel_values"] = img_t

            return example
            
        elif self.split == 'test':
            input_img = Image.open(self.lr_list[idx]).convert('RGB')
            output_img = Image.open(self.gt_list[idx]).convert('RGB')
            
            min_w = min(input_img.size[0], output_img.size[0])
            min_h = min(input_img.size[1], output_img.size[1])
            
            if min_w >= self.args.resolution_ori and min_h >= self.args.resolution_ori:
                center_crop = transforms.CenterCrop((self.args.resolution_ori, self.args.resolution_ori))
                img_t = center_crop(input_img)
                output_t = center_crop(output_img)
            else:
                img_t = input_img
                output_t = output_img
            
            resize_transform = transforms.Resize((self.args.resolution_tgt, self.args.resolution_tgt))
            img_t = resize_transform(img_t)
            output_t = resize_transform(output_t)
            # input images scaled to -1, 1
            img_t = F.to_tensor(img_t)
            img_t = F.normalize(img_t, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            # output images scaled to -1,1
            output_t = F.to_tensor(output_t)
            output_t = F.normalize(output_t, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

            example = {}
            example["neg_prompt"] = self.args.neg_prompt_csd
            example["null_prompt"] = ""
            example["output_pixel_values"] = output_t
            example["conditioning_pixel_values"] = img_t
            example["base_name"] = os.path.basename(self.lr_list[idx])

            return example
