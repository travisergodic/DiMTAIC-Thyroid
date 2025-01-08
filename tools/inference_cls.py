import os
import sys
import logging
import argparse
from pathlib import Path
from functools import partial
sys.path.insert(0, os.getcwd())

import torch
import pandas as pd

from src.logger_helper import setup_logger
from src.dataset import InferenceDataset
from src.model import MODEL
from src.transform import build_cls_transform
from src.utils import load_yaml, make_pseudo_label
from src.tta import TTA


logger = setup_logger(level=logging.INFO)


def get_topk_area_max_prob(gp, topk, col):
    return gp.nlargest(topk, "area")[col].max()


def main():
    Path("./submit/label").mkdir(exist_ok=True, parents=True)
    cls_test_transform = build_cls_transform(is_train=False, **config["TRANSFORM"]["INFERENCE"])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # dataset
    test_dataset = InferenceDataset(
        image_dir=args.image_dir, 
        mask_dir="./submit/label/",
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
    model_wrapper = TTA.build(model=model, **config["TTA"])

    # load checkpoint
    if args.weight is not None:
        model.load_state_dict(torch.load(args.weight))
        logger.info(f"Load model weights from {args.weight} successfully.")

    image_paths, area_list = [], []
    prob_list, prob_352_list, prob_448_list = [], [], []
    for batch in test_loader:
        # mask = batch["mask"].to(device=device)
        # mask_352 = F.interpolate(mask.unsqueeze(1), size=(352, 352), mode='nearest').squeeze(1)

        # # has_roi = ~(mask == 0).all()
        # data_352 = batch["data"].to(device=device)
        # data_352 = F.interpolate(data_352, size=(352, 352), mode='bilinear', align_corners=False)
        data = batch["data"].to(device=device)

        with torch.no_grad():
            # pred_352_wm = model_wrapper(data_352, mask_352).squeeze(1)
            # pred_352 = model_wrapper(data_352).squeeze(1)
            
            # pred_352_wm = torch.sigmoid(pred_352_wm)
            # pred_352 = torch.sigmoid(pred_352)
            # pred = torch.maximum(pred_352, pred_352_wm)
            # pred = torch.where(has_roi, pred, 0)
            print(torch.sigmoid(torch.tensor(0)))
            pred = model_wrapper(data).squeeze(1)
            print(pred)
            pred = torch.sigmoid(pred)
        
        image_paths += batch["path"]
        prob_list += pred.cpu().numpy().tolist()
        # prob_352_list += pred_352.cpu().numpy().tolist()
        # prob_448_list += pred_352_wm.cpu().numpy().tolist()
        area_list += batch["area"].tolist()
    
    # print(prob_list)

    # df_label = pd.DataFrame(
    #     {"image_path": image_paths, "prob": prob_list, "prob_352": prob_352_list, "prob_448": prob_448_list, "area": area_list}
    # )
    df_label = pd.DataFrame({"image_path": image_paths, "prob": prob_list, "area": area_list})
    df_label["vid_name"] = df_label["image_path"].apply(lambda s: os.path.basename(s).rsplit("_", 1)[0])

    result = (
        df_label.groupby("vid_name")
        .apply(partial(get_topk_area_max_prob, topk=args.topk, col="prob"))
        .to_frame()
        .reset_index(drop=False)
    )
    # result_352 = (
    #     df_label.groupby("vid_name")
    #     .apply(partial(get_topk_area_max_prob, topk=args.topk, col="prob_352"))
    #     .to_frame()
    #     .reset_index(drop=False)
    # )
    # result_448 = (
    #     df_label.groupby("vid_name")
    #     .apply(partial(get_topk_area_max_prob, topk=args.topk, col="prob_448"))
    #     .to_frame()
    #     .reset_index(drop=False)
    # )
    result.columns = ["case", "prob"]
    # result_352.columns = ["case", "prob"]
    # result_448.columns = ["case", "prob"]
    logger.info(f'result: {list(result["prob"])}')
    # logger.info(f'result352: {list(result_352["prob"])}')
    # logger.info(f'result448: {list(result_448["prob"])}')
    # result["prob"] = result["prob"].apply(lambda x: 1 if x>=0.4 else 0)
    result.to_csv(os.path.join("./submit/label.csv"), index=False)
    make_pseudo_label(args.image_dir, "./submit/label/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classification Model Inference.")
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--weight", type=str, default=None)
    parser.add_argument("--use_mask", action="store_true")
    parser.add_argument("--topk", type=int, default=25)
    args = parser.parse_args()
    config = load_yaml(args.config_file)
    main() 