import os
import sys
import json
import logging
import argparse
sys.path.insert(0, os.getcwd())

import torch
import pandas as pd
from sklearn.model_selection import train_test_split

from src.logger_helper import setup_logger
from src.trainer import Trainer
from src.dataset import ClassificationDataset
from src.iter_hook import *
from src.model import MODEL
from src.loss import LOSS
from src.metric import METRIC
from src.hook import HOOK
from src.optimizer.build import build_optimizer
from src.iter_hook import build_iter_hook
from src.transform import build_cls_transform
from src.utils import load_yaml, find_best_threshold


logger = setup_logger(level=logging.INFO)

def main():
    df_label = pd.read_csv(args.csv_path)
    logger.info(f"value counts: {json.dumps(df_label['prob'].value_counts().to_dict())}")

    df_train, df_val = train_test_split(
        df_label, test_size=args.test_size, random_state=42, shuffle=True, stratify=df_label["prob"]
    )

    logger.info(f"Train: {len(df_train)} images.")
    logger.info(f"Test: {len(df_val)} images.")

    # transform
    train_transform = build_cls_transform(is_train=True, **config["TRANSFORM"]["TRAIN"])
    test_transform = build_cls_transform(is_train=False, **config["TRANSFORM"]["TEST"])

    # dataset
    train_dataset = ClassificationDataset(
        image_dir=args.image_dir, 
        df_label=df_train,
        crop_roi=config["TRANSFORM"]["CROP_ROI"], 
        crop_thyroid_prob=config["TRANSFORM"]["CROP_THYROID_PROB"], 
        crop_thyroid_size=config["TRANSFORM"]["CROP_THYROID_SIZE"],  
        transform=train_transform
    )

    test_dataset = ClassificationDataset(
        image_dir=args.image_dir, 
        df_label=df_val,
        crop_roi=config["TRANSFORM"]["CROP_ROI"], 
        transform=test_transform
    )

    # dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.bs, shuffle=True, 
        num_workers=0, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.bs, shuffle=False, 
        num_workers=0, pin_memory=True
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # model
    model = MODEL.build(**config["MODEL"]).to(device)

    # loss
    loss_fn = LOSS.build(**config["LOSS"])

    # hooks
    hooks = [HOOK.build(**cfg) for cfg in config["HOOKS"]]

    # optimizer
    if args.lr is not None:
        config["OPTIMIZER"]["lr"] = args.lr
        logger.info(f"Change learning rate to {args.lr}.")
    
    if args.optim is not None:
        config["OPTIMIZER"]["type"] = args.optim
        logger.info(f"Change optimizer to {args.optim}.")
        
    optimizer = build_optimizer(model, args.lr, **config["OPTIMIZER"])

    # load checkpoint
    if args.weight is not None:
        model.load_state_dict(torch.load(args.weight))
        logger.info(f"Load model weights from {args.weight} successfully.")

    # iter hook
    iter_hook = build_iter_hook(optimizer, max_norm=args.max_norm, iter_to_accumulate=args.accum_iter)
    
    # metric
    metric_list = [METRIC.build(**metric_cfg) for metric_cfg in config["METRICS"]]

    # build trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        iter_hook=iter_hook,
        loss_fn=loss_fn,
        metric_list=metric_list,
        device=device,
        n_epochs=args.n_epochs,
        save_freq=args.save_freq,
        checkpoint_dir=f"./checkpoints/cls/",
        monitor=args.monitor,
        hooks=hooks
    )

    # train model
    trainer.fit(train_loader, test_loader)

    # # best threshold
    # if args.test_size > 0:
    #     best_threshold = find_best_threshold(model, test_loader, device=device, metric=METRIC.build(type="F1"))
    # else:
    #     best_threshold = 0.5

    # with open("./checkpoints/cls/best_threshold.txt", 'w') as f:
    #     f.write(str(best_threshold))
        
    # logger.info(f"Best threshold: {best_threshold}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train image classifier.")
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--test_size", type=float, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--max_norm", type=int, required=True)
    parser.add_argument("--accum_iter", type=int, required=True)
    parser.add_argument("--n_epochs", type=int)
    parser.add_argument("--monitor", type=str, default="Dice")
    parser.add_argument("--save_freq", type=int)
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--optim", type=str, choices=["Adam", "SGD", "AdamW", "SAM_SGD", "SAM_Adam"])
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--weight", type=str, default=None)
    args = parser.parse_args()
    config = load_yaml(args.config_file)
    main() 