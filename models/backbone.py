# ----------------------------------------------------------------------------------------------
# CoFormer Official Code
# Copyright (c) Junhyeong Cho. All Rights Reserved 
# Licensed under the Apache License 2.0 [see LICENSE for details]
# ----------------------------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved [see LICENSE for details]
# ----------------------------------------------------------------------------------------------

"""
Backbone modules.
"""
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List
from util.misc import NestedTensor, is_main_process
from .position_encoding import build_position_encoding

class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, name_backbone: str, num_channels: int, return_interm_layers: bool):
        super().__init__()
        if 'resnet' in name_backbone:
            if not train_backbone:
                for name, parameter in backbone.named_parameters():
                    parameter.requires_grad_(False)
            if return_interm_layers:
                return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
            else:
                return_layers = {'layer4': "0"}
        else:
            # TODO only resnet50 is supported
            assert False, f"backbone {name_backbone} is not supported now"
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        if 'resnet' in name:
            backbone = getattr(torchvision.models, name)(replace_stride_with_dilation=[False, False, dilation], 
                                                         pretrained=is_main_process())
            num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        else:
            # TODO only resnet is supported
            assert False, f"backbone {name} is not supported now"
        super().__init__(backbone, train_backbone, name, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    if not args.inference:
        train_backbone = args.lr_backbone > 0
    else:
        train_backbone = False
    return_interm_layers = False
    dilation = False
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model