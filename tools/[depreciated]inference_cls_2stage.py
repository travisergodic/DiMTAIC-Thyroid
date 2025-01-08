import os
import sys
import logging
import argparse
from pathlib import Path
sys.path.insert(0, os.getcwd())

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from src.logger_helper import setup_logger
from src.dataset import InferenceDataset
from src.model import MODEL
from src.transform import build_cls_transform
from src.utils import load_yaml, make_pseudo_label, get_largest_cam_box
from src.tta import TTA


logger = setup_logger(level=logging.INFO)


def main():
    Path("./submit/label").mkdir(exist_ok=True, parents=True)
    cls_test_transform = build_cls_transform(is_train=False, **config["TRANSFORM"]["INFERENCE"])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # dataset
    test_dataset = InferenceDataset(
        image_dir=args.image_dir,
        crop_roi=config["TRANSFORM"]["CROP_ROI"],
        transform=cls_test_transform
    )
    # dataloader
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.bs, shuffle=False, 
        num_workers=0, pin_memory=True
    )
    # model
    model = MODEL.build(**config["MODEL"])
    model.eval()
    model = model.to(device)

    # model wrapper
    # model_wrapper = TTA.build(model=model, **config["TTA"])

    # load checkpoint
    if args.weight is not None:
        model.load_state_dict(torch.load(args.weight))
        logger.info(f"Load model weights from {args.weight} successfully.")

    image_paths = []
    prob_list, prob_1stage_list, prob_2stage_list = [], [], []

    for batch in test_loader:
        data_512 = batch["data"].to(device=device)
        data_352 = F.interpolate(data_512, size=(352, 352), mode='bilinear', align_corners=False)
        with torch.no_grad():
            batch_cam = model.get_cam_score(data_352)
            batch_pred_1stage = batch_cam.mean(dim=(1, 2))
            batch_bboxes, max_vals = get_largest_cam_box(batch_cam, ori_size=(512, 512), square_size=352)
            crop_data_list = []
            for i in range(batch_bboxes.size(0)):
                x1, y1, x2, y2 = batch_bboxes[i]
                crop_data_list.append(data_512[i, :, y1:y2, x1:x2])
            
            crop_data_352 = torch.stack(crop_data_list, axis=0)
            batch_pred_2stage = model(crop_data_352).squeeze(1) 

            batch_pred_1stage = torch.sigmoid(batch_pred_1stage)
            batch_pred_2stage = torch.sigmoid(batch_pred_2stage)
            pred = torch.maximum(batch_pred_1stage, batch_pred_2stage)
        image_paths += batch["path"]
        prob_list += pred.cpu().numpy().tolist() 
        prob_1stage_list += batch_pred_1stage.cpu().numpy().tolist()
        prob_2stage_list += batch_pred_2stage.cpu().numpy().tolist()
  
    df_label = pd.DataFrame(
        {
            "image_path": image_paths, 
            "prob": prob_list, 
            "prob_1st": prob_1stage_list,
            "prob_2nd": prob_2stage_list
        }
    )
    df_label["vid_name"] = df_label["image_path"].apply(lambda s: os.path.basename(s).rsplit("_", 1)[0])

    result = (
        df_label.groupby("vid_name")["prob"].max()
        .to_frame()
        .reset_index(drop=False)
    )
    result_1st = (
        df_label.groupby("vid_name")["prob_1st"].max()
        .to_frame()
        .reset_index(drop=False)
    )
    result_2nd = (
        df_label.groupby("vid_name")["prob_2nd"].max()
        .to_frame()
        .reset_index(drop=False)
    )
    result.columns = ["case", "prob"]
    logger.info(f"prob: {list(result['prob'])}")
    logger.info(f"1st stage: {list(result_1st['prob_1st'])}")
    logger.info(f"2nd stage: {list(result_2nd['prob_2nd'])}")

    # result["prob"] = result["prob"].apply(lambda x: 1 if x>0.2 else 0)
    result.to_csv(os.path.join("./submit/label.csv"), index=False)

    make_pseudo_label(args.image_dir, "./submit/label/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train image classifier.")
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--weight", type=str, default=None)
    args = parser.parse_args()
    config = load_yaml(args.config_file)
    main() 