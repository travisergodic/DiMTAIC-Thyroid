import os
import sys
sys.path.insert(0, os.getcwd())

from src.model import MODEL
from src.utils import load_yaml


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


pvt_b2_cfg = load_yaml("configs/pvt_b2.yaml")["MODEL"]
emcad_b2_cfg = load_yaml("configs/EMCAD_b2.yaml")["MODEL"]

pvt_b2_model = MODEL.build(**pvt_b2_cfg)
emcad_b2_model = MODEL.build(**emcad_b2_cfg)


pvt_b2_count = count_parameters(pvt_b2_model)
emcad_b2_count = count_parameters(pvt_b2_model)


print("total parameters:", pvt_b2_count + emcad_b2_count)