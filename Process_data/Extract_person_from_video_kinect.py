import os
import csv
import cv2
import torch
import numpy as np
import argparse
from yolo.common import DetectMultiBackend
from yolo.augmentations import letterbox
from yolo.general import non_max_suppression, scale_coords

def get_parser():
    parser = argparse.ArgumentParser(description = 'Parameters of Extract Person Frame') 
    parser.add_argument(
        '--root_video_path', 
        type = str,
        default = '/data/liujinfu/dataset/kinetics_400/raw-part/compress/train_256')
    parser.add_argument(
        '--root_output_path', 
        type = str,
        default = '/data/liujinfu/dataset/kinetics_400/Process_Code_ljf/output/person')
    parser.add_argument(
        '--raw_csv_path', 
        type = str,
        default = '/data/liujinfu/dataset/kinetics_400/label/train_256.csv')
    parser.add_argument(
        '--device', 
        type = int,
        default = 0)
    parser.add_argument(
        '--model_path', 
        type = str,
        default = './pretrained/yolov5m.pt')
    parser.add_argument(
        '--data_yaml', 
        type = str,
        default = './pretrained/coco128.yaml')
    parser.add_argument(
        '--debug', 
        type = bool,
        default = False)
    return parser

class Detect_Person():
    def __init__(self, model_path, data_yaml, device):
        self.model_path = model_path
        self.data_yaml = data_yaml # './pretrained/coco128.yaml'
        self.device = torch.device(device)
        self.yolo_model = self.init_model()
    def _xywh2cs(self, x, y, w, h, image_size):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        image_width = image_size[0]
        image_height = image_size[1]
        aspect_ratio = image_width * 1.0 / image_height
        pixel_std = 200

        if w > aspect_ratio * h:
            h = w * 1.0 / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio
        scale = np.array(
            [w * 1.0 / pixel_std, h * 1.0 / pixel_std],
            dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale
    
    def init_model(self):
        model = DetectMultiBackend(self.model_path, device=self.device, dnn=False, data=self.data_yaml, fp16=False)
        return model
    
    def detect_human(self, frame_img, model, time):
        img = letterbox(frame_img, (640, 640), stride=model.stride, auto=model.pt)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img) 

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0 
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim 1 C H W

        pred = model(img, augment=False, visualize=False)

        conf_thres = 0.25  # confidence threshold
        iou_thres = 0.45  # NMS IOU threshold
        max_det = 1000  # maximum detections per image
        classes = None
        agnostic_nms = False
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        det = pred[0]
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame_img.shape).round()
        det = det.cpu().numpy()

        det = det[np.argwhere(det[:, -1] == 0), :] # 0 class: Person

        loc = np.zeros((det.shape[0], 6))
        for idx in range(det.shape[0]):
            loc[idx, :] = det[idx, :]
            loc[idx, -1] = time
        return loc
    
    def extract_person(self, img_frame):
        H, W, C = img_frame.shape
        persons_locs = self.detect_human(img_frame, self.yolo_model, 0)

        if persons_locs.shape[0] == 0: # no Person
            Num = 0
            return img_frame, Num
        
        elif persons_locs.shape[0] == 1: # one Person
            x1 = int(persons_locs[0][0])
            y1 = int(persons_locs[0][1])
            x2 = int(persons_locs[0][2])
            y2 = int(persons_locs[0][3])
            img = img_frame[y1:y2, x1:x2, :]
            Num = 1
            return img, Num

        elif persons_locs.shape[0] > 1: # More Person
            x1 = W
            x2 = 0
            y1 = H
            y2 = 0
            for _, loc in enumerate(persons_locs):
                x1 = min(x1, int(loc[0]))
                y1 = min(y1, int(loc[1]))
                x2 = max(x2, int(loc[2]))
                y2 = max(y2, int(loc[3]))
            img = img_frame[y1:y2, x1:x2, :]
            Num = 2
            return img, Num
        
def Extract_Person_frame(root_video_path, root_output_path, raw_csv_path, model_path, data_yaml, device, debug):
    model = Detect_Person(model_path, data_yaml, device)
    csv_reader = csv.reader(open(raw_csv_path))
    for idx, row in enumerate(csv_reader): 
        if (idx == 0):
            continue # ['label', 'youtube_id', 'time_start', 'time_end', 'split', 'is_cc'] 
        label, youtube_id, time_start, time_end, split, is_cc = row
        video_name = youtube_id + "_" + time_start.zfill(6) + "_" + time_end.zfill(6) + ".mp4"
        print("Process ", idx, " ", video_name)
        
        video_path = os.path.join(root_video_path, label, video_name)
        save_path = os.path.join(root_output_path, label, video_name.split(".")[0])
        if not os.path.exists(save_path):
                os.makedirs(save_path)
                
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        ret = True
        while ret:
            ret, rgb_img = cap.read()  # read each frame
            if (not ret):
                break
            img, _ = model.extract_person(rgb_img)
            img = cv2.resize(img, (224, 224))
            cv2.imwrite(save_path + '/' + str(frame_idx) + '.jpg', img)
            frame_idx = frame_idx + 1
        
        if debug: # just process one video
            break
    
if __name__ == "__main__":
    # nohup python Extract_person_from_video_kinect.py \
    # --device 0 \
    # --root_video_path /data/liujinfu/dataset/kinetics_400/raw-part/compress/train_256 \
    # --raw_csv_path /data/liujinfu/dataset/kinetics_400/label/train_256.csv \
    # > /data/liujinfu/process_kinectis_train_person.txt 2>&1 &
    parser = get_parser()
    args = parser.parse_args()
    root_video_path = args.root_video_path
    root_output_path = args.root_output_path
    raw_csv_path = args.raw_csv_path
    model_path = args.model_path
    data_yaml = args.data_yaml
    device = args.device
    debug = args.debug
    Extract_Person_frame(root_video_path, root_output_path, raw_csv_path, model_path, data_yaml, device, debug)
    print("All done!")