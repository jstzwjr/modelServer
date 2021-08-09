# 模型调度脚本

## 1、detector.py

基于yolov5的inference函数封装行人检测、电话检测模型；

inference函数返回一个batch输入图片的结果，格式为：

[result1,result2,result3,.....]

每个result为一个np.ndarray，每一行为xmin,ymin,xmax,ymax,conf,cls_id

## 2、modelServer.py

modelServer.py为模型调度脚本，包括以下几个线程：

（1）zmq数据读取线程，将读取到的图片和msg保存到一个双向队列；

（2）主线程，通过读取双向队列，构造batch数据，提交模型进行推理，并将结果推送任务丢到线程池，实现异步发送

## 3、send.py

模型imagezmq图片发送任务