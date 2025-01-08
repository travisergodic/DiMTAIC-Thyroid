import os
import random

import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from src.utils import find_images_recursive, get_bbox_from_mask, get_lower_and_upper_bound


# def crop_thyroid(pil_image, pil_mask, crop_size=352):
#     img_w, img_h = pil_image.size
#     bboxes = get_bbox_from_mask(pil_mask)
#     x_c, y_c = (x1 + x2) // 2, (y1 + y2) // 2
#     start_x = np.clip(x_c - crop_size // 2, 0, img_w - crop_size)
#     start_y = np.clip(y_c - crop_size // 2, 0, img_h - crop_size)
#     end_x = start_x + crop_size
#     end_y = start_y + crop_size
#     return pil_image.crop((start_x, start_y, end_x, end_y)), pil_mask.crop((start_x, start_y, end_x, end_y))


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, df_label, crop_roi=True, crop_thyroid_prob=0.0, crop_thyroid_size=352, transform=None):
        super().__init__()
        self.df_label = df_label.reset_index(drop=True)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.crop_roi = crop_roi
        self.crop_thyroid_prob = crop_thyroid_prob
        self.crop_thyroid_size = crop_thyroid_size

    def __getitem__(self, index):
        name = self.df_label.loc[index, "case"]
        image_path = os.path.join(self.image_dir, f"{name}.png")
        mask_path = os.path.join(self.mask_dir, f"{name}.png")
    
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)

        # crop roi
        if self.crop_roi:
            y1, y2 = get_lower_and_upper_bound(image)
            W, H = image.size
            image = image.crop((0, y1, W, y2))
            if mask is not None:
                mask = mask.crop((0, y1, W, y2))

        # crop thyroid
        # if random.uniform(0, 1) < self.crop_thyroid_prob:
        #     image, mask = crop_thyroid(image, mask, self.crop_thyroid_size)

        if self.transform:
            image, mask = self.transform(image, mask)

        return {"data": image, "label": mask}

    def __len__(self):
        return len(self.df_label)
    

class ClassificationDataset(Dataset):
    def __init__(self, image_dir, df_label, crop_roi=True, crop_thyroid_prob=0.0, crop_thyroid_size=352, transform=None):
        super().__init__()
        self.df_label = df_label.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.crop_roi = crop_roi
        self.crop_thyroid_prob = crop_thyroid_prob
        self.crop_thyroid_size = crop_thyroid_size

    def __getitem__(self, index):
        name = self.df_label.loc[index, "case"]
        label = int(self.df_label.loc[index, "prob"])
        image_path = os.path.join(self.image_dir, f"{name}.png")
        mask_path = image_path.replace("train/img/", "train/label/")
    
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)

        # crop roi
        if self.crop_roi:
            y1, y2 = get_lower_and_upper_bound(image)
            W, H = image.size
            image = image.crop((0, y1, W, y2))

        bboxes = get_bbox_from_mask(mask)

        # crop thyroid
        # if random.uniform(0, 1) < self.crop_thyroid_prob:
        #     image, mask = crop_thyroid(image, mask, crop_size=self.crop_thyroid_size)

        if self.transform:
            image, _ = self.transform(image, mask, label, bboxes)

        return {"data": image, "label": label}

    def __len__(self):
        return len(self.df_label)
    

class InferenceDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None, crop_roi=True, transform=None):
        super().__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.crop_roi = crop_roi
        self.transform = transform
        self.image_paths = sorted(find_images_recursive(image_dir))
        if mask_dir:
            self.mask_paths = sorted(find_images_recursive(mask_dir))

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")

        # crop roi
        if self.crop_roi:
            y1, y2 = get_lower_and_upper_bound(image)
            W, H = image.size
            image = image.crop((0, y1, W, y2))

        if self.transform:
            image, _ = self.transform(image, mask=None, label=None, bboxes=None)
        
        res = {"data": image, "path": image_path}

        if self.mask_dir:
            mask_path = self.mask_paths[index]
            mask = np.array(Image.open(mask_path))
            res["area"] = int(mask.sum())
            if mask.sum() <= 300:
                mask = np.zeros_like(mask, dtype=np.uint8)
            else:
                kernel = np.ones((5, 5), np.uint8)
                mask = cv2.dilate(mask, kernel, iterations=6)
            
            res["mask"] = mask = torch.from_numpy(mask).float()
        return res
    
    def __len__(self):
        return len(self.image_paths)