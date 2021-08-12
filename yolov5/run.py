#!/usr/bin/env python
# encoding: utf-8
'''
@author: wangjianrong
@software: pycharm
@file: tmp.py.py
@time: 2020/7/17 10:48
@desc:
'''


from numpy.lib.arraysetops import isin
from models.experimental import *
from utils.datasets import *
from utils.utils import *

'''
global variables
'''
#os.environ['conf_threshold'] = str(0.4)
#os.environ['iou_threshold'] = str(0.1)
#os.environ['require_color_filter'] = str(0)
augment = False

iou_thres = 0.5
conf_thres = 0.25
# thres = float(os.environ.get('conf_threshold',0.25))
# conf_thres = thres

weights, imgsz = 'yolov5/weights/yolov5x.pt', 640
device = torch.device('cuda:0')
half = device.type != 'cpu'  # half precision only supported on CUDA
# Load modelc
model = attempt_load(weights, map_location=device)  # load FP32 model
if half:
    model.half()  # to FP16
imgsz = check_img_size(imgsz, s=model.stride.max())


def create_blank(width, height):
  image = np.zeros((height, width), np.uint8)
  return image

# process.............
def processTest2(polygon,sensor_id=None,batch_num=None,frame=None):
    '''
    目前只支持一个多边形输入,polygon为归一化的坐标
    '''
    try:
        img_h,img_w,*_ = frame.shape
        mask = np.zeros((img_h, img_w), np.uint8)
        if not isinstance(polygon,np.ndarray):
            polygon = np.array(polygon)
        polygon = polygon * [img_w, img_h]
        polygon = polygon.astype(np.int32)
        cv2.fillConvexPoly(mask,polygon,(1,))

        results = inference(frame)

        final_results = []
        # polygon总像素个数
        total_num_pixel = np.sum(mask)
        for xmin, ymin, xmax, ymax, conf in results:
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmax)
            ymax = int(ymax)

            num_pixel = np.sum(mask[ymin:ymax, xmin:xmax])
            #print(f"num_pixel / total_num_pixel={num_pixel}/{total_num_pixel}={num_pixel / total_num_pixel}")
            if num_pixel >= 10:
                # final_results.append({"label":0,"score":[conf],"bbox":[xmin, ymin, xmax, ymax]})
                final_results.append([xmin,ymin,xmax,ymax,conf,0])
        return final_results
    except Exception as e:
        print(e)
        return []


def filter_polygons(polygons, results, frame):
    try:
        if len(results)==0:
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
                # print(f"num_pixel / total_num_pixel={num_pixel}/{total_num_pixel}={num_pixel / total_num_pixel}")
                if num_pixel >= 10:
                    final_results.append([xmin, ymin, xmax, ymax, cls_id, idx])
        return final_results
    except Exception as e:
        print(e)
        print("filter target by polygons error!")
        return []


# 取消该函数内的置信度阈值过滤
def filter_target_ids(results, target_ids=(), conf_thres=0.5):
    try:
        if len(results) == 0:
            return []
        # 根据目标id过滤结果，如果target_ids是一个空元组，则返回全部结果
        # if len(target_ids) == 0:
        #     filter_results = results
        # else:
        #     filter_results = list(filter(lambda x: int(x[-1]) in target_ids, results))
        # if len(filter_results) == 0:
        #     return []
        # filter_results = np.array(filter_results)
        filter_results = np.array(results)

        # 对每个类根据置信度阈值过滤目标
        # 如果target_ids为空，则使用统一阈值
        if len(target_ids) == 0:
            keep_inds = filter_results[:,-2] >= conf_thres
            filter_results = filter_results[keep_inds]
        else:
            # 如果target_ids长度不为0，且conf_thres不为tuple或者list
            if not isinstance(conf_thres, (tuple,list)):
                conf_thres = [conf_thres] * len(target_ids)
            
            results_conf_filtered = []
            for i in range(len(target_ids)):
                keep_inds = (filter_results[:,-2] >= conf_thres[i]) * (filter_results[:,-1] == target_ids[i])
                res_single_class = filter_results[keep_inds]
                results_conf_filtered.append(res_single_class)
            filter_results = np.concatenate(results_conf_filtered,axis=0)

        
        # keep_inds = filter_results[:,-2] >= conf_thres
        # filter_results = filter_results[keep_inds]

        # 删除置信度，只返回xmin,ymin,xmax,ymax,cls_id
        filter_results = filter_results[:,[0,1,2,3,5]].astype(np.int32)
        if len(filter_results) == 0:
            return []
        return filter_results

    except Exception as e:
        print(e)
        print("filter target id error!")
        return []



def processTest(polygons, frames, target_ids=(), conf_thres=()):
    '''
    return [[xmin,ymin,xmax,ymax,label,polygon_id]]
    '''
    # TODO
    # batch inference
    if not isinstance(frames,(tuple,list)):
        frames = [frames]
        polygons = [polygons]
        target_ids = [target_ids]
    #if len(polygons) == 0:
        #print("rois empty!!!")
        #return []

    # src_h, src_w, *_ = frame.shape

    # [[xmin,ymin,xmax,ymax,conf,cls_id],...]
    try:
        results = inference(frames)
    except Exception as e:
        print(e)
        print("model inference error!")
        return []
    
    for i, result in enumerate(results):
        if result is None or len(result)==0:
            results[i] = []

    # if results is None or len(results) == 0:
        # return []

    # 根据target id 过滤检测到的目标
    for i in range(len(results)):
        results[i] = filter_target_ids(results[i], target_ids[i], conf_thres)

    

    # 根据roi过滤目标
    for i, result in enumerate(results):
        results[i] = filter_polygons(polygons[i], results[i], frames[i])

    return results




def inference(im0s):
    #if isinstance(im0s,str):
        #im0s = cv2.imread(im0s)
    batch_imgs = []
    if not isinstance(im0s,(tuple,list)):
        im0s = [im0s]
    for img in im0s:
        img = preprocess(img)
        batch_imgs.append(img)
    batch_x = torch.cat(batch_imgs,dim=0)

    # t1 = torch_utils.time_synchronized()
    with torch.no_grad():
      pred = model(batch_x, augment=False)[0]

      # Apply NMS
      pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False)
    # t2 = torch_utils.time_synchronized()
    # print('inference:{}s'.format(t2-t1))

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        # im0 = im0s
        im0 = im0s[i]
        img = batch_imgs[i]
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
    results = []
    for result in pred:
        if result is None or len(result) == 0:
            results.append(np.empty((0,6)))
        else:
            results.append(result.cpu().numpy())
    return results
    # return pred[0].cpu().numpy()
    #res = pred[0]
    #if res is None:
    #    res = []
    #else:
    #    ind = res[:, -2] >= thres
    #    res = res[ind]
    #    ind = res[:,-1] == 0
    #    res = res[ind]
    #    res = res[:, :5].cpu().numpy().tolist()
    #return res


def preprocess(im0s):
    img = letterbox(im0s, (imgsz, imgsz))[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img


if __name__ == '__main__':

    cap = cv2.VideoCapture("rtsp://119.45.35.191:9554/rtp/gb_play_34020001001320000011_34020000001320000001")
    # cap = cv2.VideoCapture(1)
    # cap = cv2.VideoCapture("rtsp://admin:dhzl123456@192.168.4.45:554/h264/ch1/main/av_stream")
    while True:
        ret,frame = cap.read()
        if not ret:
            continue

        results = processTest([[0.,0.],[1.,0.],[1.,1.],[0.,1.]],None,None,frame)
        #print(results)
        for result in results:
            cls_id = int(result["label"])
            conf = float(result["score"][cls_id])
            bbox = result["bbox"]
            xmin, ymin, xmax, ymax = list(map(int,bbox))
            cv2.rectangle(frame,(xmin,ymin),(xmax,ymax),(0,0,255))
            cv2.putText(frame,"{:.2f}".format(conf),(xmin,ymin),1,2,(0,0,255))
        cv2.imshow("frame",frame)
        if 27 == cv2.waitKey(1):
            break
    cap.release()

