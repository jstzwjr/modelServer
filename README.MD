# 模型调度脚本

## 1、detector.py

基于yolov5的inference函数封装行人检测、电话检测模型；

inference函数返回一个batch输入图片的结果，格式为：

[result1,result2,result3,.....]

每个result为一个np.ndarray，每一行为xmin,ymin,xmax,ymax,conf,cls_id

#### 20210821更新

新增每个类别的置信度阈值

## 2、modelServer.py

modelServer.py为模型调度脚本，包括以下几个线程：

（1）zmq数据读取线程，将读取到的图片和msg保存到一个双向队列；

（2）主线程，通过读取双向队列，构造batch数据，提交模型进行推理，并将结果推送任务丢到线程池，实现异步发送

#### 20210821更新

更新modelServer3.py脚本，主要新增功能：

（1）集成alphapose，适配打电话模型；

（2）新增数据批解析、批推理功能；

（3）通过线程池实现模型异步推理；

（4）图片转base64移植结果发送函数；

（5）每个label增加不同置信度阈值功能。

## 3、send.py

模型imagezmq图片发送任务

## 4、数据格式

### 接收数据格式：

{
    "cameraId": 106, 
    "time": "32234234433433", 
    "modelId": 1, 
    "modelName": "person", 
    "rois": [
        {
            "id": 24, 
            "polygon": [
                0., 0.,
                1., 0., 
                1., 1., 
                0., 1.
            ]
        }, 
        {
            "id": 25, 
            "polygon":[
                0., 0.,
                1., 0., 
                1., 1., 
                0., 1.
            ]
        }
    ]
}

### 发送数据格式

{
    "cameraId":1,
    "modelId":100,
    "modelName":"smoke",
    "frameBase64":"base64string",
    "batchNum":"32234234433433",
    "results": [
        {
            "roiId": 100,
            "value": [
                person",
                "phone"
            ],
            "bbox": [
                [
                    0,
                    0,
                    1,
                    1
                ],
                [
                    0,
                    0,
                    1,
                    1
                ]
            ]
        }
    ]
}

