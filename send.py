#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/8/9 9:38
# @Author  : wangjianrong
# @File    : send.py

import cv2
# import socket
# from imutils.video import VideoStream
import imagezmq
from time import time
import json

'''
{
    "cameraId": 106, 
    "time": 1628225808.4163904, 
    "modelId": 1, 
    "modelName": "person", 
    "rois": [
        {
            "id": 24, 
            "polygon": [
                668.1328025440455, 405.25390021124633,
                1131.1327956448079, 552.2538980207756, 
                871.1327995191098, 911.2538926712589, 
                354.13280722301, 860.2538934312181
            ]
        }, 
        {
            "id": 25, 
            "polygon": [
                1128.1327956895113, 277.5039021148696, 
                1519.1327898631575, 402.5039002522245, 
                1462.1327907125237, 598.5038973315969, 
                972.1327980140925, 419.5038999989048
            ]
        }
    ]
}
'''
sender = imagezmq.ImageSender(connect_to="tcp://127.0.0.1:5000")

# rpi_name = socket.gethostname()
# print(rpi_name)

cap = cv2.VideoCapture("rtsp://admin:dhzl123456@192.168.4.45:554/h264/ch1/main/av_stream")

flag = False

msg1 = {
        "cameraId": 106,
        "time": None,
        "modelId": 1,
        "modelName": "person_depart",
        "rois": [
            {
                "id": 24,
                "polygon": [
                    0, 0, 1.0, 0, 1., 1., 0, 1.
                ]
            },
            {
                "id": 25,
                "polygon": [
                    0, 0, 1.0, 0, 1., 1., 0, 1.
                ]
            }
        ]
    }

msg2 = {
        "cameraId": 106,
        "time": None,
        "modelId": 1,
        "modelName": "person_count",
        "rois": [
            {
                "id": 24,
                "polygon": [
                    0, 0, 1.0, 0, 1., 1., 0, 1.
                ]
            },
            {
                "id": 25,
                "polygon": [
                    0, 0, 1.0, 0, 1., 1., 0, 1.
                ]
            }
        ]
    }

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    if flag:
        msg = msg1
    else:
        msg = msg2

    flag = not flag

    msg["time"] = str(int(time()))
    response = sender.send_image(json.dumps(msg), frame)
    print(response)
