from src.registry import PARAMS


def get_emcad_layer_id(name):
    first, second = name.split(".")[:2]
    if first in ("decoder", "conv", "out_head1", "out_head2", "out_head3", "out_head4"):
        return 0
    return int(second[-1])


@PARAMS.register("same_lr")
def same_lr(model, base_lr, **kwargs):
    return [{"params": model.parameters(), "lr": base_lr}]


@PARAMS.register("emcad_layer_decay_lr") 
def emcad_layer_decay_lr(model, base_lr, layer_decay=.75, **kwargs):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}
    layer_scales = [layer_decay ** i for i in range(5)]

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if n.startswith("conv"):
            continue
            
        layer_id = get_emcad_layer_id(n)
        group_name = f"layer{layer_id}"

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr": this_scale * base_lr,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale * base_lr,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    # print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))
    return [param_groups[f"layer{layer_id}"] for layer_id in range(5)] 