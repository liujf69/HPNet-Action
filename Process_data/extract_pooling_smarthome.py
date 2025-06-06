import os
import pose_hrnet

from yolo.augmentations import letterbox
from yolo.common import DetectMultiBackend
from yolo.transforms import get_affine_transform
from yolo.general import non_max_suppression, scale_coords

import torch
import torch.optim
import torch.utils.data
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torch.nn.functional as F
from config import cfg
from config import update_config

import cv2
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        default = './pretrained/w48_256x192_adam_lr1e-3_split2_sigma4.yaml',
                        help = 'experiment configure file name',
                        type = str)
    parser.add_argument('opts',
                    help = "Modify config options using the command-line",
                    default = None,
                    nargs = argparse.REMAINDER)
    parser.add_argument('--videos_path',
                    help = "the path of rgb videos",
                    default = "/data/liujinfu/dataset/Toyota-Smarthome/mp4")
    parser.add_argument('--save_path',
                    help = "the save path of pooling features",
                    default = "/data/liujinfu/dataset/Toyota-Smarthome/pooling")
    parser.add_argument('--sample_txt',
                        type = str,
                        # default = '/data/liujinfu/dataset/Toyota-Smarthome/train_CS.txt'),
                        default = '/data/liujinfu/TPAMI-24/Process_data/test.txt')
    parser.add_argument('--device',
                        type = int,
                        default = 0)

    args = parser.parse_args()
    return args

class HP_estimation():
    def __init__(self, device_ids = 4):
        self.model_detect, self.model_pose = self.init_model(device_ids)

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

    def init_model(self, device_ids):
        args = parse_args()
        update_config(cfg, args)
        model_pose = eval('pose_hrnet.get_pose_net')(cfg, is_train = False)
        model_pose.load_state_dict(torch.load('./pretrained/pose_hrnet_w48_256x192_split2_sigma4.pth', map_location = 'cpu'))
        model_pose = torch.nn.DataParallel(model_pose, device_ids = device_ids)
        self.device = torch.device(device_ids[0])
        model_detect = DetectMultiBackend('./pretrained/yolov5m.pt', device=self.device, dnn=False, data='./pretrained/coco128.yaml', fp16=False)

        return model_detect, model_pose # YoloV5, SimCC

    # use yolov5 to detect human
    def detect_human(self, frame_img, model, time):
        img = letterbox(frame_img, (640, 640), stride=model.stride, auto=model.pt)[0] # letterbox缩放
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img) # 将一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0 # 归一化
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

    # 姿态估计，针对每个人估计姿态
    def estimate_pose(self, data_numpy, model_pose, loc):
        data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

        x = loc[0]  # x1
        y = loc[1]  # y1
        w = loc[2] - loc[0]  # x2-x1
        h = loc[3] - loc[1]  # y2-y1
        image_size = (data_numpy.shape[0], data_numpy.shape[1])
        c, s = self._xywh2cs(x, y, w, h, image_size)
        r = 0
        image_size = (192, 256)

        trans = get_affine_transform(c, s, r, image_size)
        input = cv2.warpAffine(
            data_numpy,
            trans,
            (int(image_size[0]), int(image_size[1])),
            flags=cv2.INTER_LINEAR)

        # Data loading code
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        input = transform(input)
        input = input[np.newaxis, :, :, :]
        input = input.to(self.device)
        pred_x, pred_y, feature_list = model_pose(input) # pred pose

        idx_x = pred_x.argmax(2)
        idx_y = pred_y.argmax(2)

        idx_x = idx_x.cpu().float().numpy().squeeze(0)
        idx_y = idx_y.cpu().float().numpy().squeeze(0)

        idx_x /= cfg.MODEL.SIMDR_SPLIT_RATIO
        idx_y /= cfg.MODEL.SIMDR_SPLIT_RATIO

        '''
        "keypoints": {
            0: "nose",
            1: "left_eye",
            2: "right_eye",
            3: "left_ear",
            4: "right_ear",
            5: "left_shoulder",
            6: "right_shoulder",
            7: "left_elbow",
            8: "right_elbow",
            9: "left_wrist",
            10: "right_wrist",
            11: "left_hip",
            12: "right_hip",
            13: "left_knee",
            14: "right_knee",
            15: "left_ankle",
            16: "right_ankle"
        },
        '''
        return idx_x, idx_y, feature_list

    # 结合目标检测和姿态估计的总体实现
    def extract_feature_from_one_person(self, model, model_pose, img_frame):
        persons_locs = self.detect_human(img_frame, model, 0) # detect Person

        skeletons = np.zeros((2, 1, 17, 2), dtype = np.float32) # M 1 17 2
        origin_skeletons = np.zeros((2, 1, 17, 2), dtype = np.float32) # M 1 17 2
        y_list_0 = np.zeros((2, 1, 48, 64, 48), dtype = np.float32) # M 1 48 64 48
        x_list_1 = np.zeros((2, 1, 96, 32, 24), dtype = np.float32) # M 1 96 32 24
        x_list_2 = np.zeros((2, 1, 192, 16, 12), dtype = np.float32) # M 1 192 16 12
        x_list_3 = np.zeros((2, 1, 384, 8, 6), dtype = np.float32) # M 1 384 8 6
        x_ = np.zeros((2, 1, 17, 64, 48), dtype = np.float32) 

        
        # no detect Person
        if persons_locs.shape[0] == 0: 
            data_dict = {"skeleton": skeletons, "origin_ske": origin_skeletons, "location": persons_locs} 
            return data_dict, y_list_0, x_list_1, x_list_2, x_list_3, x_

        else:
            loc1 = persons_locs[0]
            skeleton = np.zeros((1, 17, 2))
            origin_skeleton = np.zeros((1, 17, 2))
            idx_x1, idx_y1, feature_list = self.estimate_pose(img_frame, model_pose, loc1) 

            skeleton[0, :, 0] = idx_x1
            skeleton[0, :, 1] = idx_y1
            skeleton = skeleton.astype(np.float32) #[1, 17, 2]
            skeletons[0] = skeleton
            
            origin_idx_x1, origin_idx_y1 = self.get_origin(loc1, idx_x1, idx_y1)
            origin_skeleton[0, :, 0] = origin_idx_x1
            origin_skeleton[0, :, 1] = origin_idx_y1
            origin_skeletons[0] = origin_skeleton
            
            y_list_0[0] = feature_list[0].cpu().detach().numpy()
            x_list_1[0] = feature_list[1].cpu().detach().numpy()
            x_list_2[0] = feature_list[2].cpu().detach().numpy()
            x_list_3[0] = feature_list[3].cpu().detach().numpy()
            x_[0] = feature_list[4].cpu().detach().numpy()

            y_list_0[1] = y_list_0[0]
            x_list_1[1] = x_list_1[0]
            x_list_2[1] = x_list_2[0]
            x_list_3[1] = x_list_3[0]
            x_[1] = x_[0]
            skeletons[1] = skeletons[0]
            origin_skeletons[1] = origin_skeletons[0]

            data_dict = {"skeleton": skeletons, "origin_ske": origin_skeletons, "location": persons_locs} 
            return data_dict, y_list_0, x_list_1, x_list_2, x_list_3, x_

    def HPose_estimation(self, img_frame):
        data_dict, y_list_0, x_list_1, x_list_2, x_list_3, x_ = self.extract_feature_from_one_person(self.model_detect, self.model_pose, img_frame)
        return data_dict, y_list_0, x_list_1, x_list_2, x_list_3, x_
    
    def get_origin(self, loc, idx_x, idx_y):
        x = loc[0]  # x1
        y = loc[1]  # y1
        w = loc[2] - loc[0]  # x2-x1
        h = loc[3] - loc[1]  # y2-y1
        image_size = (480, 640)
        c, s = self._xywh2cs(x, y, w, h, image_size)
        r = 0
        image_size = (192, 256)
        trans = get_affine_transform(c, s, r, image_size)
        inv_trans = self.inv_align(trans)

        origin_idx_x = np.zeros((17)) # 17
        origin_idx_y = np.zeros((17)) # 17
        for idx in range(17):
            origin_idx_x[idx] = idx_x[idx]*inv_trans[0][0] + idx_y[idx]*inv_trans[0][1] + inv_trans[0][2]
            origin_idx_y[idx] = idx_x[idx]*inv_trans[1][0] + idx_y[idx]*inv_trans[1][1] + inv_trans[1][2]
        
        return origin_idx_x, origin_idx_y
    
    def inv_align(self, M):
        # M的逆变换
        k  = M[0, 0]
        b1 = M[0, 2]
        b2 = M[1, 2]
        return np.array([[1/k, 0, -b1/k], [0, 1/k, -b2/k]])

    def view_pose(self, img, origin_idx_x, origin_idx_y):
        skeleton = [
            [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8],
            [7, 9], [8, 10], [9, 11], [1, 2], [1, 3], [2, 4], [3, 5], [1, 7], [1, 6]
        ]
        for idx in skeleton:
            st_x = int(origin_idx_x[idx[0] - 1])
            st_y = int(origin_idx_y[idx[0] - 1])
            ed_x = int(origin_idx_x[idx[1] - 1])
            ed_y = int(origin_idx_y[idx[1] - 1])
            cv2.line(img, (st_x, st_y), (ed_x, ed_y), (0, 255, 0), 2)

        # for i in range(17):
        #     cv2.circle(img, (int(origin_idx_x[i]), int(origin_idx_y[i])), 5, (0,255,0), -1)
        return img


def main(args):
    model = HP_estimation(device_ids = [args.device])
    samples_txt = args.sample_txt
    videos_path = args.videos_path
    save_path = args.save_path
    samples_name = np.loadtxt(samples_txt, dtype=str)

    for _, name in enumerate(samples_name): # for each sample
        print("Processing " + name)
        
        rgb_video_path = os.path.join(videos_path, name)
        cap = cv2.VideoCapture(rgb_video_path)
        
        F_x_list_1 = []
        Pose_data = []
        origin_Pose_data = []
        
        while True:
            ret, img = cap.read()  # read each video frame
            if (not ret): # read done or read error
                break
            # get pose
            data_dict, y_list_0, x_list_1, x_list_2, x_list_3, x_ = model.HPose_estimation(img) # Here we just need x_list_1
            keypoints_2d = data_dict['skeleton'] # M 1 17 2
            Pose_data.append(keypoints_2d) # T M 1 17 2
            F_x_list_1.append(x_list_1) # T M 1 96 32 24
            origin_Pose_data.append(data_dict['origin_ske'])

        Pose_data = torch.from_numpy(np.array(Pose_data)).squeeze(2) # T M 17 2
        origin_Pose_data = torch.from_numpy(np.array(origin_Pose_data)).squeeze(2) # T M 17 2
        F_x_list_1 = torch.from_numpy(np.array(F_x_list_1)).squeeze(2) # T M 96 32 24
        Pose_data[..., :2] /= torch.tensor([192//2, 256//2])
        Pose_data[..., :2] -= torch.tensor([1, 1]) # norm [-1, 1]
        T, M, V, C = Pose_data.shape # T M V C
        Pose_data = Pose_data.reshape(T*M, V, C) # TM V C        
        F_x_list_1 = F_x_list_1.reshape(T*M, F_x_list_1.shape[-3], F_x_list_1.shape[-2], F_x_list_1.shape[-1]) # TM C H W # Here we just process x_list_1
        
        feature = F.grid_sample(F_x_list_1, Pose_data.unsqueeze(-2), align_corners=True).squeeze(-1).permute(0, 2, 1).contiguous() \
    
        # resize to the same frames -> 64 frames
        save_features = feature.reshape(T, M, 17, 96) # T M V C
        save_features = save_features.permute(1, 2, 3, 0).reshape(2*17*96, T) # MVC T
        save_features = save_features[None, None, :, :]
        save_features = F.interpolate(save_features, size = (2*17*96, 64), mode = 'bilinear', align_corners = False)
        save_features = save_features.squeeze(0).squeeze(0).reshape(2, 17, 96, 64) # M V C T
        save_features = save_features.permute(-1, 0, 1, 2) # T M V C
        save_features = np.array(save_features, dtype = np.float32) # T M V C
        
        # np.save(save_path + "/" + name.split(".mp4")[0] + ".npy", save_features)    
        pose_save_path = "/data/liujinfu/dataset/Toyota-Smarthome/HPE_Pose" + "/" + name.split(".mp4")[0] + ".npy"  
        np.save(pose_save_path, origin_Pose_data)
    print("All done!")
    
if __name__ == "__main__":
    # nohup python extract_pooling_smarthome.py --sample_txt /data/liujinfu/dataset/Toyota-Smarthome/train_CS.txt --device 0 > ./outlog/Pooling_Smarthome_train_CS_0126.log 2>&1 &
    # nohup python extract_pooling_smarthome.py --sample_txt /data/liujinfu/dataset/Toyota-Smarthome/validation_CS.txt --device 0 > ./outlog/Pooling_Smarthome_validation_CS_0126.log 2>&1 &
    # nohup python extract_pooling_smarthome.py --sample_txt /data/liujinfu/dataset/Toyota-Smarthome/test_CS.txt --device 0 > ./outlog/Pooling_Smarthome_test_CS_0126.log 2>&1 &
    args = parse_args()
    main(args)