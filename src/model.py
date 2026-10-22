"""GoogLeNet with transfer regimes A / B / C (Phase 2)."""
import torch.nn as nn
from torchvision.models import GoogLeNet_Weights, googlenet

BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


def build_model(num_classes=21):
    """GoogLeNet pre-trained on ImageNet with a new `num_classes`-way FC layer.

    - Auxiliary classifiers are discarded (aux_logits=False).
    - torchvision sets transform_input=True with these weights: it converts
      ImageNet-normalised inputs to the normalisation expected by the ported
      weights, so our standard ImageNet mean/std normalisation is correct.
    """
    model = googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1, aux_logits=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # 1024 -> 21
    return model


def set_regime(model, regime):
    """Freeze/unfreeze layers.

    A: FC only | B: last inception module (inception5b) + FC | C: everything.
    """
    for p in model.parameters():
        p.requires_grad = False

    if regime == "A":
        trainable = [model.fc]
    elif regime == "B":
        trainable = [model.inception5b, model.fc]
    elif regime == "C":
        trainable = [model]
    else:
        raise ValueError(f"Unknown regime: {regime!r} (expected A, B or C)")

    for module in trainable:
        for p in module.parameters():
            p.requires_grad = True
    return model


def set_train_mode(model):
    """model.train(), but keep fully frozen BatchNorm layers in eval mode.

    Otherwise the running statistics of 'frozen' layers would still be
    updated by the target data, which contradicts the notion of a frozen
    backbone.
    """
    model.train()
    for m in model.modules():
        if isinstance(m, BN_TYPES) and not any(p.requires_grad for p in m.parameters()):
            m.eval()


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def make_optimizer(model, train_cfg):
    """SGD with two parameter groups: new head (lr_head), pre-trained layers (lr_backbone)."""
    import torch

    head = [p for p in model.fc.parameters() if p.requires_grad]
    fc_ids = {id(p) for p in model.fc.parameters()}
    backbone = [p for p in model.parameters()
                if p.requires_grad and id(p) not in fc_ids]

    groups = [{"params": head, "lr": train_cfg["lr_head"]}]
    if backbone:
        groups.append({"params": backbone, "lr": train_cfg["lr_backbone"]})
    return torch.optim.SGD(
        groups,
        momentum=train_cfg["momentum"],
        weight_decay=train_cfg["weight_decay"],
    )
