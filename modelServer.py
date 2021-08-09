import sys
import imagezmq
import json
import collections
import threading
import urllib.request

from detector import PersonDepart, PersonCount, PlayPhone
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from time import sleep, time

pool = ThreadPoolExecutor(max_workers=8)
pool_infer = ThreadPoolExecutor(max_workers=2)

batch_size = 1

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

image_hub = imagezmq.ImageHub(open_port='tcp://*:5000')
mydeque = collections.deque(maxlen=10)

modelDics = {
    "person_depart": PersonDepart(),
    "play_phone": PlayPhone(),
    "person_count": PersonCount()
}


def recv():
    global mydeque
    try:
        while True:
            messageJson, image = image_hub.recv_image()
            # print(messageJson)
            message = {'message': messageJson, 'img': image}
            # messageDic = json.loads(messageJson)
            mydeque.append(message)
            image_hub.send_reply(b'OK')
    except Exception as e:
        recv()


pool.submit(recv)


# t = threading.Thread(target=recv)
# t.setDaemon(True)
# t.start()

# 发送识别结果到router节点
def sendResult(resultDic):
    # resultDic = {"cameraId": messageDic['cameraId'], "time": messageDic['time'], "result": result}
    # return None
    url = 'http://192.168.4.69:9081/result'
    headers = {'Content-Type': 'application/json'}
    request = urllib.request.Request(url=url, headers=headers, method='POST',
                                     data=json.dumps(resultDic).encode(encoding='UTF8'))

    response = urllib.request.urlopen(request)
    # print(response)


# 发送识别结果到业务主服务
def sendEventLog(messageDic, img):
    # return None
    # messageDic = {"cameraId":self.id,"time":timestamp,"modelId":modelId,"modelName":modelName,"roi":roi['polygon']}
    resultDic = {"cameraId": messageDic['cameraId'], "modelId": messageDic['modelId'], "batchNum": messageDic['time'],
                 "frameBase64": cv2_base64(img), "result": result}

    url = 'http://192.168.40.19:9083/hz/result/save'
    headers = {'Content-Type': 'application/json'}
    request = urllib.request.Request(url=url, headers=headers, method='POST',
                                     data=json.dumps(resultDic).encode(encoding='UTF8'))

    response = urllib.request.urlopen(request)
    # print(response)


def main():
    sleep(5)
    while True:
        try:
            if len(mydeque) == 0:
                continue
            batch_messages = []
            model_messages = defaultdict(list)
            while True:
                if len(mydeque) == 0:
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

            for modelName, batch_messages in model_messages.items():
                if len(batch_messages) == 0:
                    continue
                # pool_infer.submit(modelDics[modelName].inference, batch_messages)
                results = modelDics[modelName].inference(batch_messages)
                for result in results:
                    pool.submit(sendResult, result)

            # sendResult(final_results)
            # print(resultDic)
            # sendResult(messageDic)
            # sendEventLog(messageDic,img)
        except Exception as e:
            print(e)


if __name__ == '__main__':
    main()
