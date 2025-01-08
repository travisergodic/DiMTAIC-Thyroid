import re
import os
import sys
import yaml
import random
import importlib
import logging
from pathlib import Path
from abc import abstractmethod

import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image



logger = logging.getLogger(__name__)


def get_cfg_by_file(cfg_file):
    try:
        sys.path.append(os.path.dirname(cfg_file))
        current_cfg = importlib.import_module(os.path.basename(cfg_file).split(".")[0])
        logger.info(f'Import {cfg_file} successfully!')
    except Exception:
        raise ImportError(f'Fail to import {cfg_file}')
    return current_cfg


def load_yaml(path):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def read_video(video_path):
    pil_images = []
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        pil_image = Image.fromarray(frame)
        pil_images.append(pil_image.convert("RGB"))
    cap.release()
    return pil_images


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    @abstractmethod
    def __call__(self):
        pass

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    def step(self, y_hat, y_true):
        val = self.__call__(y_hat, y_true)
        # print(val)
        n = y_true.size(0)
        self.update(val, n)
    
    @property
    def avg(self):
        return self.sum / self.count


def powerset(seq):
    """
    Returns all the subsets of this set. This is a generator.
    """
    if len(seq) <= 1:
        yield seq
        yield []
    else:
        for item in powerset(seq[1:]):
            yield [seq[0]]+item
            yield item


def get_lower_and_upper_bound(pil_or_cv2_img):
    if isinstance(pil_or_cv2_img, Image.Image):
        img = np.array(pil_or_cv2_img)
    else:
        img = pil_or_cv2_img

    if img.ndim == 3:
        img = img[:, :, 0] 

    temp = np.all(img == 0, axis=1)
    for i in range(temp.size):
        y1 = i
        if not temp[i]:
            break

    for i in range(temp.size-1, -1, -1):
        y2 = i 
        if not temp[i]:
            break
    
    if y2 - y1 >= 352:
        return y1, y2
    return y1, y1+352


def get_bounding_box(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 500  # Define the minimum area to consider as a valid region
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    bounding_boxes = [cv2.boundingRect(cnt) for cnt in valid_contours]
    # If you want the largest contour's bounding box:
    assert len(bounding_boxes) == 1 
    x, y, w, h = max(bounding_boxes, key=lambda box: box[2] * box[3])
    return x, y, x+w, y+h


def recover_mask(out_mask: np.ndarray, ori_image: np.ndarray):
    y1, y2 = get_lower_and_upper_bound(ori_image)
    H, W = ori_image.shape[:2]  # 512, 512
    mask = cv2.resize(out_mask, (W, y2 - y1), cv2.INTER_NEAREST)
    return np.concatenate(
        [np.zeros(shape=(y1, W)), mask, np.zeros(shape=(H-y2, W))], axis=0
    ).astype(np.uint8)


def find_largest_area(mask):
    if len(mask.shape) != 2:
        raise ValueError("Mask must be a grayscale image with shape (H, W).")
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    out_mask = np.zeros_like(mask)
    cv2.drawContours(out_mask, [largest_contour], -1, 1, thickness=cv2.FILLED)
    return out_mask


def filter_contours_by_area(mask, min_area=500):
    # Ensure the mask is binary (values 0 or 255)
    if len(mask.shape) != 2:
        raise ValueError("Mask must be a grayscale image with shape (H, W).")

    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty mask to draw the filtered contours
    filtered_mask = np.zeros_like(mask, dtype=np.uint8)

    # Loop through the contours and only draw those with area >= min_area
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            # Draw the valid contour on the mask
            cv2.drawContours(filtered_mask, [contour], -1, 1, thickness=cv2.FILLED)
    return filtered_mask


@torch.no_grad()
def find_best_threshold(model, test_loader, device, metric):
    model.eval()
    thresholds = np.linspace(0.2, 0.85, 27)
    model = model.to(device)
    threshold_to_score = dict()
    for threshold in thresholds:
        for batch in test_loader:
            data, mask = batch["data"].to(device), batch["label"].to(device)
            pred = model(data)
            pred = (torch.sigmoid(pred) > threshold).long()
            metric.step(pred, mask)
        threshold_to_score[threshold] = round(metric.avg, 4)
        metric.reset()
    logger.info(f"threshold_to_score: {threshold_to_score}")
    return max(threshold_to_score, key=threshold_to_score.get)


def find_images_recursive(path):
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    image_files = []
    for root, _, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[1].lower() in image_extensions:
                image_files.append(os.path.join(root, file))
    return image_files


def make_pseudo_label(src, dst):
    save_path_list = [re.sub(src, dst, path).replace("//", "/") for path in find_images_recursive(src)]
    cnt = 0
    for save_path in save_path_list:
        if not os.path.isfile(save_path):
            arr = np.random.choice([0, 1], size=(512, 512)).astype(np.uint8)
            Path(os.path.dirname(save_path)).mkdir(exist_ok=True, parents=True)
            Image.fromarray(arr).save(save_path)
            cnt += 1
    logger.info(f"Make {cnt} pseudo mask")


def get_bbox_from_mask(pil_mask):
    # Ensure mask is binary (0s and 255s)
    mask = np.array(pil_mask) * 255
    
    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Loop over all contours found (usually there's just one for a single mask)
    bboxes = []
    for contour in contours:
        # Get the bounding box for each contour
        x, y, w, h = cv2.boundingRect(contour)
        bboxes.append([x, y, x+w, y+h])
    return bboxes


def get_centered_square(image, point, square_size=384):
    img_h, img_w = image.shape
    half_square = square_size // 2
    x, y = point

    # Adjust the top-left corner of the square to keep it within bounds
    start_x = max(0, min(img_w - square_size, x - half_square))
    start_y = max(0, min(img_h - square_size, y - half_square))

    # Extract the square region
    return image[start_y:start_y + square_size, start_x:start_x + square_size]


def crop_all_image_roi(image_dir, mask_dir, df_label):
    records = []
    for _, row in df_label.iterrows():
        filename = f"{row['case']}.png"
        image_path = os.path.join(image_dir, filename)
        mask_path = os.path.join(mask_dir, filename)
        mask = np.array(Image.open(mask_path))
        x1, y1, x2, y2 = get_bbox_from_mask(mask)
        if all([ele==0 for ele in (x1, y1, x2, y2)]):
            continue

        center_pt = (x1+x2)//2, (y1+y2)//2
        # crop image
        image = np.array(Image.open(image_path))
        crop_image = get_centered_square(image, center_pt, square_size=352)
        filename = os.path.basename(image_path)
        dirname = os.path.dirname(image_path)
        Image.fromarray(crop_image).save(os.path.join(dirname, f"crop_{filename}"))
        # crop mask
        mask = np.array(Image.open(mask_path))
        crop_image = get_centered_square(mask, center_pt, square_size=352)
        filename = os.path.basename(mask_path)
        dirname = os.path.dirname(mask_path)
        Image.fromarray(crop_image).save(os.path.join(dirname, f"crop_{filename}"))

        records.append({"case": row['case'], "prob": row['prob']})
    return pd.DataFrame.from_records(records)


def get_largest_cam_box(batch_cam, ori_size, square_size):
    # batch_cam: (B, H, W)
    B, H, W = batch_cam.size()
    img_h, img_w = ori_size

    # Calculate scaling factors
    y_scale, x_scale = img_h / H, img_w / W

    # Flatten batch_cam to find max indices per batch
    flattened_cam = batch_cam.view(B, -1)  # Shape: (B, H*W)
    max_vals, max_indices = torch.max(flattened_cam, dim=1)

    # Convert 1D indices to 2D coordinates
    max_i = torch.div(max_indices, W, rounding_mode="floor")  # Row index
    max_j = max_indices % W   # Column index

    # Calculate center coordinates in original image
    y_coords = ((max_i + 0.5) * y_scale).long()
    x_coords = ((max_j + 0.5) * x_scale).long()

    # Define square box boundaries, adjusting for edges
    half_square = square_size // 2
    start_x = torch.clamp(x_coords - half_square, min=0, max=img_w - square_size)
    start_y = torch.clamp(y_coords - half_square, min=0, max=img_h - square_size)
    end_x = start_x + square_size
    end_y = start_y + square_size

    # Stack coordinates and return along with max cam values
    return torch.stack([start_x, start_y, end_x, end_y], dim=1), max_vals


def iou(bbox1, bbox2):
    # Calculate the Intersection over Union (IoU) of two bounding boxes
    x1, y1, x2, y2 = bbox1
    bx1, by1, bx2, by2 = bbox2
    
    inter_x1 = max(x1, bx1)
    inter_y1 = max(y1, by1)
    inter_x2 = min(x2, bx2)
    inter_y2 = min(y2, by2)
    
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    bbox1_area = (x2 - x1) * (y2 - y1)
    bbox2_area = (bx2 - bx1) * (by2 - by1)
    
    union_area = bbox1_area + bbox2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0


def generate_cutout_bbox(label, obj_bboxes, image_width, image_height):
    if label == 0:
        # Directly return obj_bbox if label is 0
        return obj_bboxes

    # label=1 => generate cutout bbox 
    for _ in range(20):
        cutout_size = random.randint(56, 224)
        
        # Generate random top-left corner for the cutout bbox
        x1 = random.randint(0, image_width - cutout_size)
        y1 = random.randint(0, image_height - cutout_size)
        cutout_bbox = [x1, y1, x1 + cutout_size, y1 + cutout_size]
        
        # Check IoU
        if any([iou(cutout_bbox, obj_bbox)<= 0.1 for obj_bbox in obj_bboxes]):
            return [cutout_bbox]
    return [[0, 0, 0, 0]]
