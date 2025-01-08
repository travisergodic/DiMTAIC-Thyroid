import os
import re
import sys
import random
import logging
import argparse
from pathlib import Path
sys.path.insert(0, os.getcwd())

import torch
import numpy as np
import pandas as pd
from PIL import Image

from src.logger_helper import setup_logger
from src.dataset import InferenceDataset
from src.model import MODEL
from src.tta import TTA
from src.transform import build_seg_transform
from src.utils import load_yaml, recover_mask


logger = setup_logger(level=logging.INFO)


def main():
    Path("./submit/label").mkdir(exist_ok=True, parents=True)
    seg_test_transform = build_seg_transform(is_train=False, **config["TRANSFORM"]["INFERENCE"])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # dataset
    test_dataset = InferenceDataset(
        image_dir=args.image_dir,
        crop_roi=config["TRANSFORM"]["CROP_ROI"], 
        transform=seg_test_transform
    )
    # dataloader
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.bs, shuffle=False, 
        num_workers=0, pin_memory=True
    )
    # model
    model = MODEL.build(**config["MODEL"])
    model = model.to(device)
    model.eval()

    # load checkpoint
    if args.weight is not None:
        model.load_state_dict(torch.load(args.weight))
        logger.info(f"Load model weights from {args.weight} successfully.")

    ## TTA
    # if "threshold" not in config["TTA"]:
    #     with open("./checkpoints/seg/best_threshold.txt", "r") as f:
    #         best_threshold = float(f.read())
    #     config["TTA"]["threshold"] = best_threshold
        
    model_wrapper = TTA.build(model=model, **config["TTA"])

    cnt = 0
    path_list = []
    for batch in test_loader:
        data = batch["data"].to(device=device)
        y_hat = model_wrapper(data).cpu()

        for i, path in enumerate(batch["path"]):
            path_list.append(path)
            # out mask
            out_mask = (y_hat[i] > args.threshold).squeeze().long()
            out_mask = out_mask.numpy().astype(np.uint8)
            # fm mask
            # fm_mask = (y_hat[i] > 0.25).squeeze().long()
            # fm_mask = fm_mask.numpy().astype(np.uint8)
            
            if config["TRANSFORM"]["CROP_ROI"]:
                ori_image = np.array(Image.open(path))
                out_mask = recover_mask(out_mask, ori_image)
                # fm_mask = recover_mask(fm_mask, ori_image)

            img = Image.fromarray(out_mask)
            save_path = re.sub(args.image_dir, "./submit/label/", path).replace("//", "/")
            dir_name = os.path.dirname(save_path)
            Path(dir_name).mkdir(exist_ok=True, parents=True)
            img.save(save_path)

            # img = Image.fromarray(fm_mask)
            # save_path = re.sub(args.image_dir, "./submit/soft_label/", path).replace("//", "/")
            # dir_name = os.path.dirname(save_path)
            # Path(dir_name).mkdir(exist_ok=True, parents=True)
            # img.save(save_path)
            cnt += 1

    vid_names = set([os.path.basename(path).rsplit("_", 1)[0] for path in path_list])
    vid_names = sorted(list(vid_names))

    pd.DataFrame(
        {"case": vid_names, "prob": [random.choice([0, 1]) for _ in range(len(vid_names))]}
    ).to_csv("./submit/label.csv", index=False)
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segmentation Model Inference.")
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--weight", type=str, default=None)
    args = parser.parse_args()
    config = load_yaml(args.config_file)
    main() 