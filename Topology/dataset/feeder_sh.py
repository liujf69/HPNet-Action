import os
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset

Smarthome_action_label_dict = {'Cook.Cleandishes': 0, 'Cook.Cleanup': 1, 'Cook.Cut': 2, 'Cook.Stir': 3, 'Cook.Usestove': 4, 'Cutbread': 5, 'Drink.Frombottle': 6,
                               'Drink.Fromcan': 7, 'Drink.Fromcup': 8, 'Drink.Fromglass': 9, 'Eat.Attable': 10, 'Eat.Snack': 11, 'Enter': 12, 'Getup': 13, 
                               'Laydown': 14, 'Leave': 15, 'Makecoffee.Pourgrains': 16, 'Makecoffee.Pourwater': 17, 'Maketea.Boilwater': 18, 'Maketea.Insertteabag': 19,
                               'Pour.Frombottle': 20, 'Pour.Fromcan': 21, 'Pour.Fromkettle': 22, 'Readbook': 23, 'Sitdown': 24, 'Takepills': 25, 'Uselaptop': 26,
                               'Usetablet': 27, 'Usetelephone': 28, 'Walk': 29, 'WatchTV': 30}

Smarthome_action_label_dict_CV = {'Cutbread': 0, 'Drink.Frombottle': 1, 'Drink.Fromcan': 2, 'Drink.Fromcup': 3, 'Drink.Fromglass': 4, 'Eat.Attable': 5, 'Eat.Snack': 6,
                               'Enter': 7, 'Getup': 8, 'Leave': 9, 'Pour.Frombottle': 10, 'Pour.Fromcan': 11, 'Readbook': 12, 'Sitdown': 13, 
                               'Takepills': 14, 'Uselaptop': 15, 'Usetablet': 16, 'Usetelephone': 17, 'Walk': 18}

class Feeder(Dataset):
    def __init__(self, feature_data_path, sample_txt_path, bone = False, vel = False, benchmark = 'CS'):
        self.feature_data_path = feature_data_path
        self.sample_txt_path = sample_txt_path
        self.benchmark = benchmark
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
            if self.benchmark == 'CS':
                label = int(Smarthome_action_label_dict[action])
            else:
                label = int(Smarthome_action_label_dict_CV[action])
            self.label.append(label)
        self.sample_name = self.sample_txt.tolist()

    def __len__(self):
        return len(self.sample_txt)

    def __getitem__(self, index):
        sample_name = self.sample_txt[index]
        feature_path = self.feature_data_path + '/' + sample_name.split('.mp4')[0] + '.npy'
        label = self.label[index]
        feature_data = np.load(feature_path, allow_pickle=True) # T(64) M(2) V(17) C(780)
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


