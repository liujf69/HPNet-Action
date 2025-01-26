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
import matplotlib.pyplot as plt
import os

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
                    default = "/data/liujinfu/dataset/UAV/all_rgb/all_rgb")
    parser.add_argument('--save_path',
                    help = "the save path of pooling features",
                    default = "./visual_save")
    parser.add_argument('--sample_txt',
                        type = str,
                        default = './sample_txt/test.txt'),
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
        image_size = (1080,1920)
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
    samples_name = np.loadtxt(samples_txt, dtype=str)

    for _, name in enumerate(samples_name): # for each sample
        print("Processing " + name)
        
        save_path = os.path.join(args.save_path, name) 
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_frames_path = os.path.join(args.save_path, name) + "/" + "frames"
        if not os.path.exists(save_frames_path):
            os.makedirs(save_frames_path)
         
        rgb_video_path = os.path.join(videos_path, name)
        cap = cv2.VideoCapture(rgb_video_path)
        
        F_y_list_0 = []
        F_x_list_1 = []
        F_x_list_2 = []
        Pose_data = []
        F_x_ = []
        Origin_Pose_data = []
        frame_idx = 0
        while True:
            ret, img = cap.read()  # read each video frame
            if (not ret): # read done or read error
                break
            # get pose
            data_dict, y_list_0, x_list_1, x_list_2, x_list_3, x_ = model.HPose_estimation(img) # Here we just need x_list_1
            keypoints_2d = data_dict['skeleton'] # M 1 17 2
            origin_keypoints_2d = data_dict['origin_ske'] # M 1 17 2
            Pose_data.append(keypoints_2d) # T M 1 17 2
            Origin_Pose_data.append(origin_keypoints_2d) # T M 1 17 2
            F_y_list_0.append(y_list_0) # T M 1 48 64 48
            F_x_list_1.append(x_list_1) # T M 1 96 32 24
            F_x_list_2.append(x_list_2) # T M 1 192 16 12
            F_x_.append(x_) # T M 1 17 32 24
            cv2.imwrite(save_frames_path + "/" + str(frame_idx) + ".jpg", img)
            frame_idx += 1

        Pose_data = torch.from_numpy(np.array(Pose_data)).squeeze(2) # T M 17 2
        Save_pose_data = Pose_data # T M 17 2 # save
        Origin_pose_data = torch.from_numpy(np.array(Origin_Pose_data)).squeeze(2) # T M 17 2
        F_y_list_0 = torch.from_numpy(np.array(F_y_list_0)).squeeze(2) # T M 48 64 48 # save
        F_x_list_1 = torch.from_numpy(np.array(F_x_list_1)).squeeze(2) # T M 96 32 24
        F_x_list_2 = torch.from_numpy(np.array(F_x_list_2)).squeeze(2) # T M 192 16 12
        F_x_ = torch.from_numpy(np.array(F_x_)).squeeze(2) # T M 96 32 24
        
        Save_F_x_list_1 = F_x_list_1 # T M 96 32 24 # save to visual
        
        Pose_data[..., :2] /= torch.tensor([192//2, 256//2])
        Pose_data[..., :2] -= torch.tensor([1, 1]) # norm [-1, 1]
        T, M, V, C = Pose_data.shape # T M V C
        Pose_data = Pose_data.reshape(T*M, V, C) # TM V C        
        F_x_list_1 = F_x_list_1.reshape(T*M, F_x_list_1.shape[-3], F_x_list_1.shape[-2], F_x_list_1.shape[-1]) # TM C H W # Here we just process x_list_1
        pooled_features = F.grid_sample(F_x_list_1, Pose_data.unsqueeze(-2), align_corners=True).squeeze(-1).permute(0, 2, 1).contiguous() # save
        save_pooled_features = pooled_features.reshape(T, M, 17, 96) # T M V C # save
                
        np.save(save_path + "/" + "_pose.npy", Save_pose_data)
        np.save(save_path + "/" + "_origin_pose.npy", Origin_pose_data)
        np.save(save_path + "/" + "_y_list_0.npy", F_y_list_0)
        np.save(save_path + "/" + "_x_list_1.npy", Save_F_x_list_1)
        np.save(save_path + "/" + "_x_list_2.npy", F_x_list_2)
        np.save(save_path + "/" + "_pooled.npy", save_pooled_features)
        np.save(save_path + "/" + "_x_.npy", F_x_)
                
    print("All done!")
    
def visualize_confidence_map(tensor, heatmap_size=None, output_path='confidence_heatmap.png', type = "y_list_0"):
    # 确保 tensor 的大小为 [C, H, W]
    assert tensor.ndim == 3, "Input tensor must have 3 dimensions [C, H, W]"
    C, H, W = tensor.shape
    # 将 tensor 展平，并进行归一化
    tensor_min = tensor.min()
    tensor_max = tensor.max()
    normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min)  # 归一化到 [0, 1]
    
    if type == "y_list_0":
        # 合并通道为单一的置信度图
        confidence_map = np.sum(normalized_tensor, axis=0)  # shape: [H, W] # sum is better
    elif type == "x_list_1":
        confidence_map = np.max(normalized_tensor, axis=0)  # shape: [H, W] # max is better
    elif type == "x_list_2":
        confidence_map = np.max(normalized_tensor, axis=0)  # shape: [H, W]
    elif type == "x_":
        confidence_map = np.max(normalized_tensor, axis=0)  # shape: [H, W]
    
    # 将置信度图调整为目标大小
    confidence_map_resized = cv2.resize(confidence_map, (heatmap_size[0], heatmap_size[1]))
    
    # if type == "x_list_1":
    #     confidence_map_blurred = cv2.GaussianBlur(confidence_map_resized, (15, 15), sigmaX=5, sigmaY=5) # 高斯模糊平滑
    confidence_map_blurred = np.clip(confidence_map_resized, 0, 1) # 归一化热图到 [0, 1] 范围

    # 将热图转换为伪彩色图像
    heatmap_colored = cv2.applyColorMap((confidence_map_blurred * 255).astype(np.uint8), cv2.COLORMAP_JET)
    # 保存热图
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Heatmap saved to {output_path}")

def visualize_vector_matrix(matrix, output_path="vector_heatmap.png"):
    # 确保矩阵是 2D 的
    assert matrix.ndim == 2, "Input matrix must have 2 dimensions [V, C]"

    # 获取 V, C
    V, C = matrix.shape

    # 对矩阵进行归一化到 [0, 1]
    matrix_min = matrix.min()
    matrix_max = matrix.max()
    normalized_matrix = (matrix - matrix_min) / (matrix_max - matrix_min)

    # 转换为伪彩色图像
    heatmap_colored = cv2.applyColorMap((normalized_matrix * 255).astype(np.uint8), cv2.COLORMAP_JET)

    # 保存热图
    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Heatmap saved to {output_path}")

def view_rgb_pose(img, origin_idx_x, origin_idx_y):
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

if __name__ == "__main__":    
    args = parse_args()
    main(args)
    
    # visual test
    test_txt = np.loadtxt(args.sample_txt, dtype = str)
    for idx, name in enumerate(test_txt):
        print("Process ", name)
        output_path = os.path.join(args.save_path, name) 
        pose_data_path = os.path.join(output_path, "_pose.npy")
        origin_pose_data_path = os.path.join(output_path, "_origin_pose.npy")
        y_list_0_path = os.path.join(output_path, "_y_list_0.npy")
        x_list_1_path = os.path.join(output_path, "_x_list_1.npy")
        x_list_2_path = os.path.join(output_path, "_x_list_2.npy")
        pooled_features_path = os.path.join(output_path, "_pooled.npy")
        x_path = os.path.join(output_path, "_x_.npy")
        pose_data = np.load(pose_data_path, allow_pickle = True) # T M V 2
        origin_pose_data = np.load(origin_pose_data_path, allow_pickle = True) # T M V 2
        y_list_0 = np.load(y_list_0_path, allow_pickle = True) # T M 48 64 48
        x_list_1 = np.load(x_list_1_path, allow_pickle = True) # T M 96 32 24
        x_list_2 = np.load(x_list_2_path, allow_pickle = True) # T M 192 16 12
        pooled_features = np.load(pooled_features_path, allow_pickle = True) # T M V 96
        x_ = np.load(x_path, allow_pickle = True) # T M 16 64 48

        Num_frames = y_list_0.shape[0]
        Num_bodies = y_list_0.shape[1]
        for T in range(Num_frames):
            T = 200
            M = 0 # body 1    
            y_list_0_visual_data = y_list_0[T, M]
            save_y_list_0_path = output_path + '/' + str(T) + "_y_list_0.png"
            visualize_confidence_map(y_list_0_visual_data, 
                                    heatmap_size = (y_list_0_visual_data.shape[1], y_list_0_visual_data.shape[2]), 
                                    output_path = save_y_list_0_path,
                                    type = "y_list_0")  
            
            x_list_1_visual_data = x_list_1[T, M]
            save_x_list_1_path = output_path + '/' + str(T) + "_x_list_1.png"
            visualize_confidence_map(x_list_1_visual_data, 
                                    heatmap_size = (x_list_1_visual_data.shape[1], x_list_1_visual_data.shape[2]), 
                                    output_path = save_x_list_1_path,
                                    type = "x_list_1") 
            
            x_list_2_visual_data = x_list_2[T, M]
            save_x_list_2_path = output_path + '/' + str(T) + "_x_list_2.png"
            visualize_confidence_map(x_list_2_visual_data, 
                                    heatmap_size = (x_list_2_visual_data.shape[1], x_list_2_visual_data.shape[2]), 
                                    output_path = save_x_list_2_path,
                                    type = "x_list_2")
            
            x_visual_data = x_[T, M]
            save_x_path = output_path + '/' + str(T) + "_x.png"
            visualize_confidence_map(x_visual_data, 
                        heatmap_size = (x_visual_data.shape[1], x_visual_data.shape[2]), 
                        output_path = save_x_path,
                        type = "x_") 
    
            pooled_features = pooled_features[T, M]
            save_pooled_features_path = output_path + '/' + str(T) + "_pooled_features.png"
            visualize_vector_matrix(pooled_features, save_pooled_features_path) 
            
            origin_pose = origin_pose_data[T, M]
            img = cv2.imread(output_path + "/" + "frames" + "/" + str(T) + ".jpg")
            origin_idx_x = origin_pose[:, 0]
            origin_idx_y = origin_pose[:, 1]
            rgb_pose = view_rgb_pose(img, origin_idx_x, origin_idx_y)
            cv2.imwrite(output_path + "/" + str(T) + "_rgb_pose.jpg", rgb_pose)
            
            print("debug pause")
            break # Test one frame
            
    print("All done!")