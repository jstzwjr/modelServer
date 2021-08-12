import sys
import imagezmq
import json
import collections
import threading
import urllib.request

from detector import PersonDepart, PersonCount, PlayPhone, CallPhone
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
from collections import defaultdict
from time import sleep, time
import base64
import cv2


pool = ThreadPoolExecutor(max_workers=8)
pool_infer = ThreadPoolExecutor(max_workers=4)

batch_size = 2

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

def cv2_base64(image):
    base64_str = cv2.imencode('.jpg', image)[1].tostring()
    base64_str = base64.b64encode(base64_str)
    return base64_str


image_hub = imagezmq.ImageHub(open_port='tcp://*:5556')
mydeque = collections.deque(maxlen=10)

modelDics = {
    "person_depart": PersonDepart(),
    "play_phone": PlayPhone(),
    "person_count": PersonCount(),
    "call_phone": CallPhone()    # 如果不需要检测打电话，建议注释该行，减少模型显存占用
}


def recv():
    global mydeque
    try:
        while True:
            # print(len(mydeque),"...................")
            messageJson, image = image_hub.recv_image()
            # print(messageJson)
            message = {'message': messageJson, 'img': image}
            # messageDic = json.loads(messageJson)
            mydeque.append(message)
            image_hub.send_reply(b'OK')
    except Exception as e:
        recv()

# 消息接收线程
pool.submit(recv)


# t = threading.Thread(target=recv)
# t.setDaemon(True)
# t.start()

# 发送识别结果到router节点
# 将图片转base64移到该函数内
def sendResult(resultDic, img):
    # resultDic = {"cameraId": messageDic['cameraId'], "time": messageDic['time'], "result": result}
    # return None
    url = 'http://127.0.0.1:9081/result'
    headers = {'Content-Type': 'application/json'}
    base64string = cv2_base64(img)
    base64string = str(base64string, encoding='utf-8')
    resultDic["frameBase64"] = base64string
    request = urllib.request.Request(url=url, headers=headers, method='POST',
                                     data=json.dumps(resultDic).encode(encoding='UTF8'))

    response = urllib.request.urlopen(request)
    # print(response)


# 发送识别结果到业务主服务
def sendEventLog(resultDic, img):
    # return None
    # messageDic = {"cameraId":self.id,"time":timestamp,"modelId":modelId,"modelName":modelName,"roi":roi['polygon']}
    #resultDic = {"cameraId": messageDic['cameraId'], "modelId": messageDic['modelId'], "batchNum": messageDic['time'],
    #             "frameBase64": cv2_base64(img), "result": result}

    url = 'http://127.0.0.1:9001/hz/result/save'
    headers = {'Content-Type': 'application/json'}
    base64string = cv2_base64(img)
    base64string = str(base64string, encoding='utf-8')
    resultDic["frameBase64"] = base64string

    request = urllib.request.Request(url=url, headers=headers, method='POST',
                                     data=json.dumps(resultDic).encode(encoding='UTF8'))

    response = urllib.request.urlopen(request)
    # print(response)


def main():
    while True:
        try:
            if len(mydeque) == 0:
                continue
            model_messages = defaultdict(list)
            while True:
                if len(mydeque) == 0:
                    if len(model_messages):
                        break
                    continue
                # while len(batch_messages) != batch_size:
                message = mydeque.popleft()  # {'message':messageJson,'img':image}
                if message is None:
                    break
                # batch_messages.append(message)
                msg = json.loads(message['message'])
                modelName = msg["modelName"]
                model_messages[modelName].append(message)
                if len(model_messages[modelName]) == batch_size:
                    break
            
            # 模型异步推理
            s = time()
            tasks = []
            for modelName, batch_messages in model_messages.items():
                if len(batch_messages) == 0:
                    continue
                print(len(batch_messages))

                task = pool_infer.submit(modelDics[modelName].detect, batch_messages)
                tasks.append(task)
            # submit的返回值是乱序的
            all_results = wait(tasks, return_when=ALL_COMPLETED)
            batch_results = []
            for single_restlts in all_results.done:
                # batch_results.append(restlt.result())
                batch_results = single_restlts.result()
                modelName = batch_results[0]["modelName"]
                batch_messages = model_messages[modelName]
                print(modelName)
                print(batch_results)
                # 将图片转base64移到sendResult函数内，避免模型后处理事件过长引起阻塞
                # 图片转base64 耗时约45ms
                for i, result in enumerate(batch_results):
                    message = batch_messages[i]
                    img = message['img']
                    #base64string = cv2_base64(img)
                    #base64string = str(base64string, encoding='utf-8')
                    #result["frameBase64"] = base64string
                    pool.submit(sendEventLog, result, img)

            e = time()
            print("total:",e-s)


            # for modelName, batch_messages in model_messages.items():
            #     if len(batch_messages) == 0:
            #         continue
            #     results = batch_results.pop(0)
            #     print(modelName)
            #     print(results)
            #     # 将图片转base64移到sendResult函数内，避免模型后处理事件过长引起阻塞
            #     # 图片转base64 耗时约45ms
            #     for i, result in enumerate(results):
            #         message = batch_messages[i]
            #         img = message['img']
            #         pool.submit(sendResult, result, img)


            # # 模型同步推理
            # for modelName, batch_messages in model_messages.items():
            #     if len(batch_messages) == 0:
            #         continue
            #     print(len(batch_messages))
                
            #     results = modelDics[modelName].detect(batch_messages)
            #     print(results)
            
            #     # 将图片转base64移到sendResult函数内，避免模型后处理事件过长引起阻塞
            #     # 图片转base64 耗时约45ms
            #     for i, result in enumerate(results):
            #         message = batch_messages[i]
            #         img = message['img']
            #         pool.submit(sendResult, result, img)
            #     # for result in results:
            #         # pool.submit(sendResult, result)

            # sendResult(final_results)
            # print(resultDic)
            # sendResult(messageDic)
            # sendEventLog(messageDic,img)
        except Exception as e:
            print(e)


if __name__ == '__main__':
    main()
