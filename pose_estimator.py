#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/8/9 16:38
# @Author  : wangjianrong
# @File    : pose_estimator.py

import sys
sys.path.append("./Alphapose")
from AlphaPose.run import inference as infer
sys.path.pop(-1)

import os
import numpy as np
import cv2

ratio_thres = float(os.environ.get("ratio_thres", 0.3))
conf_kps_thres = float(os.environ.get("conf_kps_thres", 0.5))


def action_recog(hm_pts, hm_scores):
    results = []
    for i in range(len(hm_pts)):
        hm_pt = hm_pts[i]
        key_scores = hm_scores[i, [5, 6, 11, 12]]
        if key_scores.min() < conf_kps_thres:
            print("not accurate pts!")
            results.append(-1)
            continue
        # process hand pts
        hands = []
        for j in [9, 10]:
            if hm_scores[i, j] >= 0.5:
                hands.append(hm_pt[j])
        ears = []
        for j in [3, 4]:
            if hm_scores[i, j] >= 0.5:
                ears.append(hm_pt[j])

        if len(hands) == 0 or len(ears) == 0:
            print("hand or ear pts miss!")
            results.append(-1)
            continue

        dis_func = cal_euclidean_distance

        left_hip_pt = hm_pt[12]
        right_hip_pt = hm_pt[11]
        left_shoulder_pt = hm_pt[6]
        right_sholder_pt = hm_pt[5]
        mean_shoulder_pt = (
            (left_shoulder_pt[0] + right_sholder_pt[0]) / 2, (left_shoulder_pt[1] + right_sholder_pt[1]) / 2)
        mean_hip_pt = ((left_hip_pt[0] + right_hip_pt[0]) / 2, (left_hip_pt[1] + right_hip_pt[1]) / 2)
        # print(mean_shoulder_pt)
        # print(mean_hip_pt)
        body_length = dis_func(mean_shoulder_pt, mean_hip_pt)

        dis_hand_ear = np.inf
        for hand in hands:
            for ear in ears:
                dis_hand_ear = min(dis_hand_ear, dis_func(hand, ear))

        # print(dis_hand_ear)
        # print(body_length)
        # print(dis_hand_ear/body_length)

        dis_ratio = dis_hand_ear / body_length
        if dis_ratio <= ratio_thres:
            results.append(1)
        else:
            results.append(0)
    return results


def cal_euclidean_distance(pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2
    return np.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))


def cal_ver_distance(pt1, pt2):
    return abs(pt1[1] - pt2[1])


def inference(frame_bgr, dets):
    '''
    dets:[[xmin,ymin,xmax,ymax,label,polygon_id],[],[],...]
    '''
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    boxes = dets[:, :4]
    # 估计pose点坐标和置信度
    hm_pts, hm_scores = infer(frame_rgb, boxes)

    # 动作识别
    action_results = action_recog(hm_pts, hm_scores)
    action_results = np.array(action_results)

    # 根据识别结果过滤检测目标
    valid_inds = action_results == 1
    filter_results = dets[valid_inds]
    if len(filter_results) == 0:
        return []

    return filter_results

# class PoseEstimator:
#     def __init__(self):
#         pass

#     def inference(self, frame_rgb, dets):
#         '''
#         dets:[[xmin,ymin,xmax,ymax,label,polygon_id],[],[],...]
#         '''
#         boxes = dets[:, :4]
#         # 估计pose点坐标和置信度
#         hm_pts, hm_scores = inference(frame_rgb, boxes)

#         # 动作识别
#         action_results = action_recog(hm_pts, hm_scores)

#         # 根据识别结果过滤检测目标
#         valid_inds = action_results == 1
#         filter_results = dets[valid_inds]
#         if len(filter_results) == 0:
#             return []

#         return filter_results
