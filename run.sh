#!/bin/bash

data_root=/tcdata

# seg task
bs=10 # 10
lr=1e-4
n_epoch=70 # 70
test_size=0.1 # 0.1
optim=AdamW


python tools/train_seg.py --config_file configs/EMCAD_b2.yaml \
                          --csv_path ${data_root}/train/label.csv \
                          --test_size ${test_size} \
                          --image_dir ${data_root}/train/img \
                          --mask_dir ${data_root}/train/label \
                          --max_norm 5 \
                          --accum_iter 1 \
                          --n_epochs ${n_epoch} \
                          --save_freq 1 \
                          --bs ${bs} \
                          --lr ${lr} \
                          --optim ${optim} \
                          --monitor "Dice"


python tools/inference_seg.py --config_file  configs/EMCAD_b2.yaml \
                              --image_dir ${data_root}/test/ \
                              --bs 1 \
                              --weight ./checkpoints/seg/last.pt \
                              --threshold 0.5


# cls task
bs=20 # 20
lr=1e-4
n_epoch=70 # 70
test_size=0.1 # 0.1

python tools/train_cls.py --config_file configs/pvt_b2.yaml \
                          --csv_path ${data_root}/train/label.csv \
                          --test_size ${test_size} \
                          --image_dir ${data_root}/train/img \
                          --max_norm 5 \
                          --accum_iter 1 \
                          --n_epochs ${n_epoch} \
                          --save_freq 1 \
                          --bs ${bs} \
                          --lr ${lr} \
                          --optim AdamW \
                          --monitor "F1"

python tools/inference_cls.py --config_file configs/pvt_b2.yaml \
                              --image_dir ${data_root}/test/ \
                              --bs 1 \
                              --weight ./checkpoints/cls/best.pt

# zip
cd ./submit
zip -qr ../submit.zip .