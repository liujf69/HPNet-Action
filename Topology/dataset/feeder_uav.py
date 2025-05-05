import os
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset

class Feeder(Dataset):
    def __init__(self, feature_data_path, sample_txt_path, bone = False, vel = False):
        self.feature_data_path = feature_data_path
        self.sample_txt_path = sample_txt_path
        self.load_data()
        self.bone_pairs = [(1, 6), (2, 1), (3, 1), (4, 2), (5, 3), (6, 7),
                    (7, 1), (8, 6), (9, 7), (10, 8), (11, 9),
                    (12, 6), (13, 7), (14, 12), (15, 13), (16, 14), (17, 15)]
        self.bone = bone
        self.vel = vel

    def load_data(self):
        self.sample_txt = np.loadtxt(self.sample_txt_path, dtype = str)
        self.label = []
        for idx, name in enumerate(self.sample_txt):
            action = name.split("_")[0]
            label = int(name.split("A")[1][:3])
            self.label.append(label)
        self.sample_name = self.sample_txt.tolist()

    def __len__(self):
        return len(self.sample_txt)

    def __getitem__(self, index):
        sample_name = self.sample_txt[index]
        feature_path = self.feature_data_path + '/' + sample_name.split('.txt')[0] + '.npy'
        label = self.label[index]
        feature_data = np.load(feature_path, allow_pickle=True) # T(64) M(2) V(17) C(96)
        data = torch.from_numpy(feature_data).permute(3, 0, 2, 1) # C, T, V, M
        if self.bone:
            bone_data = torch.zeros(data.shape)
            for v1, v2 in self.bone_pairs:
                bone_data[:, :, v1 - 1] = data[:, :, v1 - 1] - data[:, :, v2 - 1]
            data = bone_data
        if self.vel:
            data[:, :-1] = data[:, 1:] - data[:, :-1]
            data[:, -1] = 0
        return data, label, index
    
    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)

if __name__ == "__main__":
    train_loader = torch.utils.data.DataLoader(
        dataset = Feeder(feature_data_path = '/data/liujinfu/dataset/UAV/pooling', 
                            sample_txt_path = '/data/liujinfu/dataset/UAV/Process_Pose/new_CS_train_V1.txt',
                            bone = True, 
                            vel = False),
        batch_size = 4,
        shuffle = False,
        num_workers = 2,
        drop_last = False)
    
    for batch_size, (data, label, idx) in enumerate(train_loader):
        data = data.float() # B C T V M
        label = label.long() # B 1
        print("pasue")