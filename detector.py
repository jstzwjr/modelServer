#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/8/9 10:07
# @Author  : wangjianrong
# @File    : detector.py

from abc import ABCMeta, abstractmethod
import sys
sys.path.append("./yolov5")
from yolov5.run import inference
sys.path.pop(-1)

import json
import cv2
import base64
from time import time
import numpy as np


def cv2_base64(image):
    base64_str = cv2.imencode('.jpg', image)[1].tostring()
    base64_str = base64.b64encode(base64_str)
    return base64_str


# 根据多边形过滤roi外的目标
def filter_polygons(polygons, results, frame):
    try:
        if len(results) == 0:
            return []
        if len(polygons) == 0:
            print("rois empty!!!")
            return []

        src_h, src_w, *_ = frame.shape
        polygon_masks = {polygon['id']: np.zeros(frame.shape[:2], dtype=np.uint8) for polygon in polygons}
        for polygon in polygons:
            idx = polygon['id']
            polygon = polygon['polygon']
            if len(polygon) % 2 != 0:
                print("polygon '{}' len error!".format(polygon))
                return []
            polygon = np.array(polygon).reshape(-1, 1, 2) * (src_w, src_h)
            polygon = polygon.astype(np.int32)
            cv2.fillConvexPoly(polygon_masks[idx], polygon, 1)

        final_results = []
        for idx, polygon_mask in polygon_masks.items():
            total_num_pixel = np.sum(polygon_mask)
            for xmin, ymin, xmax, ymax, cls_id in results:
                num_pixel = np.sum(polygon_mask[ymin:ymax, xmin:xmax])
                print(f"num_pixel / total_num_pixel={num_pixel}/{total_num_pixel}={num_pixel / total_num_pixel}")
                if num_pixel >= 10:
                    final_results.append([xmin, ymin, xmax, ymax, cls_id, idx])
        return final_results
    except Exception as e:
        print(e)
        print("filter target by polygons error!")
        return []

# 根据目标id过滤目标
def filter_target_ids(results, target_ids=(), conf_thres=0.7):
    try:
        if len(results) == 0:
            return []
        if len(target_ids) == 0:
            filter_results = results
        else:
            filter_results = list(filter(lambda x: int(x[-1]) in target_ids, results))
        if len(filter_results) == 0:
            return []
        filter_results = np.array(filter_results)
        # ▒| ▒▒~M▒置信度▒~G滤▒~[▒▒| ~G▒~H▒~O▒▒~@~I▒~I
        keep_inds = filter_results[:, -2] >= conf_thres
        filter_results = filter_results[keep_inds]

        # ▒~H| ▒~Y▒置信度▒~L▒~O▒▒~]▒~U~Yxmin,ymin,xmax,ymax,cls_id
        filter_results = filter_results[:, [0, 1, 2, 3, 5]].astype(np.int32)
        if len(filter_results) == 0:
            return []
        return filter_results

    except Exception as e:
        print(e)
        print("filter target id error!")
        return []

# 模型推理+roi过滤+目标id过滤
def processTest(polygons, frames, target_ids=()):
    '''
    return [[xmin,ymin,xmax,ymax,label,polygon_id]]
    '''
    # TODO
    # batch inference
    if not isinstance(frames, (tuple, list)):
        frames = [frames]
        polygons = [polygons]
        target_ids = [target_ids]

    try:
        # inference的返回值为list
        # list的每个成员为np.ndarray
        # 每个np.ndarry每行格式为xmin,ymin,xmax,ymax,conf,cls_id
        results = inference(frames)
    except Exception as e:
        print(e)
        print("model inference error!")
        return []

    for i, result in enumerate(results):
        if result is None or len(result) == 0:
            results[i] = []

    # 根据target id 过滤检测到的目标
    for i in range(len(results)):
        results[i] = filter_target_ids(results[i], target_ids[i])

    # 根据roi过滤目标
    for i, result in enumerate(results):
        results[i] = filter_polygons(polygons[i], results[i], frames[i])

    return results


class Yolov5DetABC(metaclass=ABCMeta):
    def __init__(self):
        self.modelName = "yolov5"
        self.target_ids = ()
        self.label = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                      'traffic light',
                      'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
                      'cow',
                      'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
                      'frisbee',
                      'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                      'surfboard',
                      'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
                      'apple',
                      'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                      'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                      'cell phone',
                      'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                      'teddy bear',
                      'hair drier', 'toothbrush']

    def inference(self, batch_messages):
        '''
        batch inference
        '''
        polygons = []
        frames = []
        msgDics = []
        s = time()
        for message in batch_messages:
            messageDic = json.loads(message['message'])
            msgDics.append(messageDic)
            img = message['img']
            rois = messageDic["rois"]
            frames.append(img)
            polygons.append(rois)

        target_ids = [self.target_ids] * len(frames)
        print("prepare:", time() - s)

        s = time()
        results = processTest(polygons, frames, target_ids=target_ids)
        print("inference:", time() - s)

        s = time()
        outputs = self.wrap_output(msgDics, results, frames)
        print("post:", time() - s)
        return outputs

    def wrap_output(self, msgDics, results, frames):
        outputs = [self.wrap_single_output(msgDic, result, frame)
                   for msgDic, result, frame in zip(msgDics, results, frames)]
        return outputs

    def wrap_single_output(self, messageDic, result, img):
        modelName = messageDic["modelName"]
        batchNum = messageDic["time"]
        cameraId = messageDic["cameraId"]
        modelId = messageDic["modelId"]
        s = time()
        base64string = cv2_base64(img)
        base64string = str(base64string, encoding='utf-8')
        print("to base64:", time() - s)
        final_results = {
            "cameraId": cameraId,
            "modelId": modelId,
            "modelName": modelName,
            "frameBase64": base64string,
            "batchNum": batchNum,
            "results": [
            ]
        }
        rois = messageDic["rois"]
        for roi in rois:
            emptyResult = {
                "roiId": roi["id"],
                "value": [],
                "bbox": []
            }
            final_results["results"].append(emptyResult)

        s = time()
        # results = {}
        for *box, label, polygon_id in result:
            xmin, ymin, xmax, ymax = list(map(int, box))
            label = int(label)
            polygon_id = int(polygon_id)
            for polygon_result in final_results["results"]:
                if polygon_result["roiId"] == polygon_id:
                    polygon_result["value"].append(self.label[self.target_ids.index(label)])
                    polygon_result["bbox"].append([xmin, ymin, xmax, ymax])

            # if polygon_id not in results:
            #     results[polygon_id] = {
            #         "value": [],
            #         "bbox": [],
            #     }
            # results[polygon_id]["value"].append(self.label[self.target_ids.index(label)])
            # results[polygon_id]["bbox"].append([xmin, ymin, xmax, ymax])
        # for polygon_id, result in results.items():
        #     singleRoiResult = {
        #         "polygonId": polygon_id,
        #         "value": result["value"],
        #         "bbox": result["bbox"]
        #     }
        #     final_results["results"].append(singleRoiResult)
        # print("final_results:", final_results)
        print("format:", time() - s)
        return final_results


class PersonDepart(Yolov5DetABC):
    def __init__(self):
        super(PersonDepart, self).__init__()
        self.modelName = "person_depart"
        self.target_ids = (0,)
        self.label = ["person"]


class PlayPhone(Yolov5DetABC):
    def __init__(self):
        super(PlayPhone, self).__init__()
        self.modelName = "play_phone"
        self.target_ids = (67,)
        self.label = ["phone"]


class PersonCount(Yolov5DetABC):
    def __init__(self):
        super(PersonCount, self).__init__()
        self.modelName = "person_count"
        self.target_ids = (0,)
        self.label = ["person"]


class PersonAndPhone(Yolov5DetABC):
    def __init__(self):
        super(PersonAndPhone, self).__init__()
        self.modelName = "play_phone"
        self.target_ids = (0, 67)
        self.label = ["person", "phone"]

# class CallPhone(Yolov5DetABC):
#     def __init__(self):
#         super(CallPhone, self).__init__()
#         self.modelName = "play_phone"


# class PersonSleep(Yolov5DetABC):
#     def __init__(self):
#         super(PersonSleep, self).__init__()
#         self.modelName = "person_sleep"
