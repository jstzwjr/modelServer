#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/8/9 16:15
# @Author  : wangjianrong
# @File    : run.py

import torch
import numpy as np
import cv2
from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.bbox import _center_scale_to_box,_box_to_center_scale
from alphapose.utils.transforms import get_affine_transform,im_to_torch,heatmap_to_coord_simple
from pathlib import Path
import os
# print(Path(__file__).absolute().parents[0].as_posix())

cur_dir = Path(__file__).absolute().parents[0].as_posix()

torch.set_num_threads(1)


cfg = update_config(os.path.join(cur_dir, "configs/coco/resnet/256x192_res50_lr1e-3_1x-duc.yaml"))
pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
pose_model.load_state_dict(torch.load(os.path.join(cur_dir, "pretrained_models/fast_421_res50-shuffle_256x192.pth"), map_location="cuda:0"))
pose_model = pose_model.to("cuda:0")
pose_model.eval()

def test_transform(src, bbox):
    input_size = (256,192)
    _aspect_ratio = float(input_size[1]) / input_size[0]

    xmin, ymin, xmax, ymax = bbox
    center, scale = _box_to_center_scale(
        xmin, ymin, xmax - xmin, ymax - ymin, _aspect_ratio)
    # print(center,scale)
    scale = scale * 1.0

    inp_h, inp_w = input_size

    trans = get_affine_transform(center, scale, 0, [inp_w, inp_h])
    img = cv2.warpAffine(src, trans, (int(inp_w), int(inp_h)), flags=cv2.INTER_LINEAR)
    bbox = _center_scale_to_box(center, scale)
    img = im_to_torch(img)
    img[0].add_(-0.406)
    img[1].add_(-0.457)
    img[2].add_(-0.480)

    return img, bbox

def inference(frame_rgb, boxes):
    inps = [None for _ in range(len(boxes))]
    cropped_boxes = [None for _ in range(len(boxes))]

    for i, box in enumerate(boxes):
        inps[i], cropped_box = test_transform(frame_rgb, box)
        cropped_boxes[i] = torch.FloatTensor(cropped_box)

    # Pose Estimation
    inps = torch.stack(inps)
    inps = inps.to("cuda:0")
    hm = pose_model(inps)
    # return [],[]
    hm_data = hm.cpu()

    pose_coords = []
    pose_scores = []
    for i in range(hm_data.shape[0]):
        bbox = cropped_boxes[i].tolist()
        pose_coord, pose_score = heatmap_to_coord_simple(hm_data[i][np.arange(17)], bbox, hm_shape=(64, 48), norm_type=None)
        pose_coords.append(torch.from_numpy(pose_coord).unsqueeze(0))
        pose_scores.append(torch.from_numpy(pose_score).unsqueeze(0))
    preds_img = torch.cat(pose_coords)
    preds_scores = torch.cat(pose_scores)
    preds_img = preds_img.numpy()
    preds_scores = preds_scores.numpy()
    return preds_img, preds_scores


