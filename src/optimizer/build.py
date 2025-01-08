import torch.optim as optim

from .sam import SAM
from src.registry import OPTIMIZER
from src.optimizer.params import PARAMS



@OPTIMIZER.register
def Adam(**kwargs):
    return optim.Adam(**kwargs)


@OPTIMIZER.register
def AdamW(**kwargs):
    return optim.AdamW(**kwargs)


@OPTIMIZER.register
def SGD(**kwargs):
    return optim.SGD(**kwargs)


@OPTIMIZER.register
def SAM_Adam(**kwargs):
    return SAM(base_optimizer=optim.Adam, **kwargs)


@OPTIMIZER.register
def SAM_AdamW(**kwargs):
    return SAM(base_optimizer=optim.AdamW, **kwargs)


@OPTIMIZER.register
def SAM_SGD(**kwargs):
    return SAM(base_optimizer=optim.SGD, **kwargs)


def build_optimizer_from_params(params, **kwargs):
    kwargs["params"] = params
    return OPTIMIZER.build(**kwargs)

def build_optimizer(model, base_lr, **kwargs):
    params_config = kwargs.pop("params")
    params_config["model"] = model
    params_config["base_lr"] = base_lr
    params = PARAMS.build(**params_config)
    return build_optimizer_from_params(params, **kwargs)