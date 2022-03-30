# ----------------------------------------------------------------------------------------------
# CoFormer Official Code
# Copyright (c) Junhyeong Cho. All Rights Reserved 
# Licensed under the Apache License 2.0 [see LICENSE for details]
# ----------------------------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved [see LICENSE for details]
# ----------------------------------------------------------------------------------------------

"""
Utilities for bounding box manipulation and GIoU.
"""
import torch
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def swig_box_xyxy_to_cxcywh(x, mw, mh, device=None, gt=False):
    # if x0, y0, x1, y1 == -1, -1, -1, -1, then cx and cy are -1.
    # so we can determine the existence of groundings with b[:, 0] != -1 (for gt)
    x0, y0, x1, y1 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    cx = ((x0 + x1) / 2).unsqueeze(1)
    cy = ((y0 + y1) / 2).unsqueeze(1)
    w = ((x1 - x0)).unsqueeze(1)
    h = ((y1 - y0)).unsqueeze(1)
    b = torch.cat([cx, cy, w, h], dim=1)
    if device is None:
        if gt:
            b[b[:,0] != -1] /= torch.tensor([mw, mh, mw, mh], dtype=torch.float32)
            b[b[:,0] == -1] = -1
        else:
            b /= torch.tensor([mw, mh, mw, mh], dtype=torch.float32)
    else:
        if gt:
            b[b[:,0] != -1] /= torch.tensor([mw, mh, mw, mh], dtype=torch.float32, device=device)
            b[b[:,0] == -1] = -1
        else:
            b /= torch.tensor([mw, mh, mw, mh], dtype=torch.float32, device=device)
    return b

def swig_box_cxcywh_to_xyxy(x, mw, mh, device=None, gt=False):
    # if x_c, y_c, w, h == -1, -1, -1, -1, then x0 < 0.
    # so we can determine the existence of groundings with b[:, 0] < 0 (for gt)
    if device is None:
        if gt:
            x[x[:,0] != -1] *= torch.tensor([mw, mh, mw, mh], dtype=torch.float32)
        else:
            x *= torch.tensor([mw, mh, mw, mh], dtype=torch.float32)
    else:
        if gt:
            x[x[:,0] != -1] *= torch.tensor([mw, mh, mw, mh], dtype=torch.float32, device=device)
        else:
            x *= torch.tensor([mw, mh, mw, mh], dtype=torch.float32, device=device)
    x_c, y_c, w, h = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    x0 = (x_c - 0.5 * w).unsqueeze(1)
    y0 = (y_c - 0.5 * h).unsqueeze(1)
    x1 = (x_c + 0.5 * w).unsqueeze(1)
    y1 = (y_c + 0.5 * h).unsqueeze(1)
    b = torch.cat([x0, y0, x1, y1], dim=1)
    if gt:
        b[b[:,0] < 0] = -1
    return b

# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area
