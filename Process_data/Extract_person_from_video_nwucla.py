import os
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
        '--sample_name_path', 
        type = str,
        default = '/data/liujinfu/TPAMI-24/Process_data/sample_txt/NW_UCLA_Video.txt')
    parser.add_argument(
        '--video_path', 
        type = str,
        default = '/data/liujinfu/dataset/multiview_action_videos')
    parser.add_argument(
        '--output_path', 
        type = str,
        default = '/data/liujinfu/dataset/NW-UCLA/person_224')
    parser.add_argument(
        '--device', 
        type = int,
        default = 1)
    parser.add_argument(
        '--model_path', 
        type = str,
        default = './pretrained/yolov5m.pt')
    parser.add_argument(
    '--data_yaml', 
    type = str,
    default = './pretrained/coco128.yaml')
    return parser

class Detect_Person():
    def __init__(self, model_path, data_yaml, device):
        self.model_path = model_path
        self.data_yaml = data_yaml
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
        persons_locs = self.detect_human(img_frame, self.yolo_model, 0)
        if persons_locs.shape[0] == 0: # no Person
            return None
        else: 
            x1 = int(persons_locs[0][0])
            y1 = int(persons_locs[0][1])
            x2 = int(persons_locs[0][2])
            y2 = int(persons_locs[0][3])
            img = img_frame[y1:y2, x1:x2, :]
            return img

def Extract_Person_frame(samples, video_path, output_path, model_path, data_yaml, device):
    model = Detect_Person(model_path, data_yaml, device)
    for _, name in enumerate(samples):
        print("Processing " + name)
        a_idx = name[:3]
        video_file_path = os.path.join(video_path, a_idx) + '/' + name[4:] + '.avi'
        save_path = os.path.join(output_path, name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        cap = cv2.VideoCapture(video_file_path)
        frame_idx = 0
        ret = True
        while ret:
            ret, rgb_img = cap.read()  # read each frame
            if (not ret):
                break
            img = model.extract_person(rgb_img) # Here we just need img3
            if img is None: # no person
                continue
            else:
                save_name = save_path + '/' + str(frame_idx) + '.jpg'
                img = cv2.resize(img, (224, 224))
                cv2.imwrite(save_name, img)
                frame_idx += 1

if __name__ == "__main__":    
    # nohup python Extract_person_from_video_nwucla.py --sample_name_path ./sample_txt/NW_UCLA_Video.txt --device 1 > ./outlog/NW-UCLA.log 2>&1 &
    parser = get_parser()
    args = parser.parse_args()
    sample_name_path = args.sample_name_path
    samples = np.loadtxt(sample_name_path, dtype=str)
    video_path = args.video_path
    output_path = args.output_path
    model_path = args.model_path
    data_yaml = args.data_yaml
    device = args.device
    Extract_Person_frame(samples, video_path, output_path, model_path, data_yaml, device)
    print("All done!")