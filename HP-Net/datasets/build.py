from torch.utils.data import DataLoader
import torch.distributed as dist
import torch
import numpy as np
from functools import partial

import os.path as osp
from collections.abc import Mapping, Sequence
from mmcv.utils import Registry
from torch.utils.data import Dataset
import copy
import os.path as osp
from abc import ABCMeta, abstractmethod
from collections import defaultdict
import os.path as osp
import mmcv
import numpy as np
import torch
from .pipeline import *
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from mmcv.parallel import collate
import pandas as pd

img_norm_cfg = dict(
    mean = [123.675, 116.28, 103.53], std = [58.395, 57.12, 57.375], to_bgr = False)

# NTU
NTU120_CSub_Train = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 
                        56, 57, 58, 59, 70, 74, 78, 80, 81, 82, 83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103]
NTU60_CSub_Train = [1,  2,  4,  5,  8,  9,  13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]

NTU120_CSet_Train = [i for i in range(1, 33) if i % 2 == 0]
NTU60_CView_Train = [2, 3]

# Smarthome
Smarthome_action_label_dict = {'Cook.Cleandishes': 0, 'Cook.Cleanup': 1, 'Cook.Cut': 2, 'Cook.Stir': 3, 'Cook.Usestove': 4, 'Cutbread': 5, 'Drink.Frombottle': 6,
                               'Drink.Fromcan': 7, 'Drink.Fromcup': 8, 'Drink.Fromglass': 9, 'Eat.Attable': 10, 'Eat.Snack': 11, 'Enter': 12, 'Getup': 13, 
                               'Laydown': 14, 'Leave': 15, 'Makecoffee.Pourgrains': 16, 'Makecoffee.Pourwater': 17, 'Maketea.Boilwater': 18, 'Maketea.Insertteabag': 19,
                               'Pour.Frombottle': 20, 'Pour.Fromcan': 21, 'Pour.Fromkettle': 22, 'Readbook': 23, 'Sitdown': 24, 'Takepills': 25, 'Uselaptop': 26,
                               'Usetablet': 27, 'Usetelephone': 28, 'Walk': 29, 'WatchTV': 30}

Smarthome_action_label_dict_CV = {'Cutbread': 0, 'Drink.Frombottle': 1, 'Drink.Fromcan': 2, 'Drink.Fromcup': 3, 'Drink.Fromglass': 4, 'Eat.Attable': 5, 'Eat.Snack': 6,
                               'Enter': 7, 'Getup': 8, 'Leave': 9, 'Pour.Frombottle': 10, 'Pour.Fromcan': 11, 'Readbook': 12, 'Sitdown': 13, 
                               'Takepills': 14, 'Uselaptop': 15, 'Usetablet': 16, 'Usetelephone': 17, 'Walk': 18}

class NTUDataset(Dataset, metaclass=ABCMeta):
    def __init__(self, ann_file, pipeline, repeat = 1, data_prefix=None, train = True, test_mode=False, multi_class=False,
                 num_classes=None, start_index=1, modality='RGB', power=0, dynamic_length=False, BenchMark = "NTU120-XSub",
                 feature_data_path = None, sample_txt_path = None, bone = None, vel = None):
        super().__init__()
        self.ann_file = ann_file
        self.repeat = repeat
        self.data_prefix = data_prefix
        self.train = train
        self.test_mode = test_mode
        self.multi_class = multi_class
        self.num_classes = num_classes
        self.start_index = start_index
        self.modality = modality
        self.power = power
        self.dynamic_length = dynamic_length
        self.pipeline = Compose(pipeline)
        self.BenchMark = BenchMark
        
        self.labels = []
        self.video_infos = []
        self.index_map = []
        self.nframes = []
        
        self.feature_data_path = feature_data_path
        self.sample_txt_path = sample_txt_path
        self.sample_txt = np.loadtxt(self.sample_txt_path, dtype = str)
        self.sample_name = self.sample_txt.tolist()
        self.bone_pairs = [(1, 6), (2, 1), (3, 1), (4, 2), (5, 3), (6, 7),
                    (7, 1), (8, 6), (9, 7), (10, 8), (11, 9),
                    (12, 6), (13, 7), (14, 12), (15, 13), (16, 14), (17, 15)]
        self.bone = bone
        self.vel = vel

        if self.BenchMark == "UCF-101": # UCF
            with open(self.ann_file, 'r') as fin:
                for line in fin:
                    line_split = line.strip().split()
                    label = int(line_split[1]) - 1
                    name = line_split[0].split(".")[0]
                    filename = osp.join(self.data_prefix, name)
                    
                    self.labels.append(label)
                    self.video_infos.append(dict(filename=filename, label=label, name=name))
        
        elif self.BenchMark == "Smarthome-CS": # Smarthome
            sample_txt = np.loadtxt(self.ann_file, dtype = str)
            for _, name in enumerate(sample_txt):
                action = name.split("_")[0]
                label = int(Smarthome_action_label_dict[action])
                filename = osp.join(self.data_prefix, name.split(".mp4")[0])
                self.labels.append(label)
                self.video_infos.append(dict(filename=filename, label=label, name=name))
        
        elif self.BenchMark == "Smarthome-CV": # Smarthome
            sample_txt = np.loadtxt(self.ann_file, dtype = str)
            for _, name in enumerate(sample_txt):
                action = name.split("_")[0]
                label = int(Smarthome_action_label_dict_CV[action])
                filename = osp.join(self.data_prefix, name.split(".mp4")[0])
                self.labels.append(label)
                self.video_infos.append(dict(filename=filename, label=label, name=name))
            
        else: # NTU
            sample_txt = np.loadtxt(self.ann_file, dtype = str)
            for _, name in enumerate(sample_txt):
                subject = int(name[9:12])
                setup = int(name[1:4])
                camera = int(name[5:8])
                label = int(name[-3:]) - 1
                filename = osp.join(self.data_prefix, name)
                
                # NTU120 XSub
                if self.BenchMark == "NTU120-XSub":
                    if self.train:
                        if subject in NTU120_CSub_Train: 
                            self.labels.append(label)
                            self.video_infos.append(dict(filename=filename, label=label, name=name))
                    else:
                        if subject not in NTU120_CSub_Train: # cross-subject
                            self.labels.append(label)
                            self.video_infos.append(dict(filename=filename, label=label, name=name))
                
                # NTU120 XSet
                elif self.BenchMark == "NTU120-XSet":
                    if self.train:
                        if setup in NTU120_CSet_Train: 
                            self.labels.append(label)
                            self.video_infos.append(dict(filename=filename, label=label, name=name))
                    else:
                        if setup not in NTU120_CSet_Train: # cross-setup
                            self.labels.append(label)
                            self.video_infos.append(dict(filename=filename, label=label, name=name))
                            
                # NTU60 XSub
                elif self.BenchMark == "NTU60-XSub":
                    if self.train:
                        if subject in NTU60_CSub_Train: 
                            self.labels.append(label)
                            self.video_infos.append(dict(filename=filename, label=label, name=name))
                    else:
                        if subject not in NTU60_CSub_Train: # cross-subject
                            self.labels.append(label)
                            self.video_infos.append(dict(filename=filename, label=label, name=name))
            
                # NTU60 XView
                else:
                    assert self.BenchMark == "NTU60-XView"
                    if self.train:
                        if camera in NTU60_CView_Train: 
                            self.labels.append(label)
                            self.video_infos.append(dict(filename=filename, label=label, name=name))
                    else:
                        if camera not in NTU60_CView_Train: # cross-view
                            self.labels.append(label)
                            self.video_infos.append(dict(filename=filename, label=label, name=name))
                    
        self.frames_per_clip = 8
        self.num_classes = max(self.labels) + 1

    def load_ske_data(self, index):
        sample_name = self.sample_txt[index]
        if self.BenchMark == "Smarthome-CS" or self.BenchMark == "Smarthome-CV":
            feature_path = self.feature_data_path + '/' + sample_name.split('.mp4')[0] + '.npy'
        else: # NTU
            feature_path = self.feature_data_path + '/' + sample_name.split('.avi')[0] + '.npy'
        feature_data = np.load(feature_path, allow_pickle=True) # T(64) M(2) V(17) C(780)
        data = torch.from_numpy(feature_data).permute(3, 0, 2, 1) # C, T, V, M
        j_data = data # j
        if self.bone: # True
            bone_data = torch.zeros(data.shape)
            for v1, v2 in self.bone_pairs:
                bone_data[:, :, v1 - 1] = data[:, :, v1 - 1] - data[:, :, v2 - 1]
        if self.vel: # True
            data[:, :-1] = data[:, 1:] - data[:, :-1]
            data[:, -1] = 0
            jm_data = data
        return j_data, jm_data, bone_data, sample_name # j, jm, b, name

    @abstractmethod
    def load_annotations(self):
        """Load the annotation according to ann_file into video_infos."""

    # json annotations already looks like video_infos, so for each dataset,
    # this func should be the same
    def load_json_annotations(self):
        """Load json annotation file to get video information."""
        video_infos = mmcv.load(self.ann_file)
        num_videos = len(video_infos)
        path_key = 'frame_dir' if 'frame_dir' in video_infos[0] else 'filename'
        for i in range(num_videos):
            path_value = video_infos[i][path_key]
            if self.data_prefix is not None:
                path_value = osp.join(self.data_prefix, path_value)
            video_infos[i][path_key] = path_value
            if self.multi_class:
                assert self.num_classes is not None
            else:
                assert len(video_infos[i]['label']) == 1
                video_infos[i]['label'] = video_infos[i]['label'][0]
        return video_infos

    def parse_by_class(self):
        video_infos_by_class = defaultdict(list)
        for item in self.video_infos:
            label = item['label']
            video_infos_by_class[label].append(item)
        return video_infos_by_class

    @staticmethod
    def label2array(num, label):
        arr = np.zeros(num, dtype=np.float32)
        arr[label] = 1.
        return arr

    @staticmethod
    def dump_results(results, out):
        """Dump data to json/yaml/pickle strings or files."""
        return mmcv.dump(results, out)

    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        # print(self.video_infos[idx])
        results['modality'] = self.modality
        results['start_index'] = self.start_index

        if self.multi_class and isinstance(results['label'], list):
            onehot = torch.zeros(self.num_classes)
            onehot[results['label']] = 1.
            results['label'] = onehot

        aug1 = self.pipeline(results)
        if self.repeat > 1:
            aug2 = self.pipeline(results)
            ret = {"imgs": torch.cat((aug1['imgs'], aug2['imgs']), 0),
                    "label": aug1['label'].repeat(2),
            }
            return ret
        else:
            return aug1 # imgs and label

    def prepare_test_frames(self, idx):
        """Prepare the frames for testing given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        results['modality'] = self.modality
        results['start_index'] = self.start_index
        if self.multi_class and isinstance(results['label'], list):
            onehot = torch.zeros(self.num_classes)
            onehot[results['label']] = 1.
            results['label'] = onehot

        return self.pipeline(results)

    def __len__(self):
        """Get the size of the dataset."""
        return len(self.video_infos)

    def __getitem__(self, idx):
        """Get the sample for either training or testing given index."""
        j_data, jm_data, b_data, name = self.load_ske_data(idx)
        if self.test_mode:
            return self.prepare_test_frames(idx), j_data, jm_data, b_data, name
        return self.prepare_train_frames(idx), j_data, jm_data, b_data, name

class VideoDataset(NTUDataset):
    def __init__(self, ann_file, pipeline, labels_file, start_index=0, **kwargs):
        super().__init__(ann_file, pipeline, start_index=start_index, **kwargs)
        self.labels_file = labels_file

    @property
    def classes(self):
        with open(self.labels_file) as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]
        classes_all = lines
        return classes_all

    def load_annotations(self):
        """Load annotation file to get video information."""
        if self.ann_file.endswith('.json'):
            return self.load_json_annotations()

        video_infos = []
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                line_split = line.strip().split()
                if self.multi_class:
                    assert self.num_classes is not None
                    filename, label = line_split[0], line_split[1:]
                    label = list(map(int, label))
                else:
                    filename, label = line_split
                    label = int(label)
                if self.data_prefix is not None:
                    filename = osp.join(self.data_prefix, filename)
                video_infos.append(dict(filename=filename, label=label, tar=self.use_tar_format))
        return video_infos

class SubsetRandomSampler(torch.utils.data.Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.epoch = 0
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)

    def set_epoch(self, epoch):
        self.epoch = epoch

def mmcv_collate(batch, samples_per_gpu=1): 
    if not isinstance(batch, Sequence):
        raise TypeError(f'{batch.dtype} is not supported.')
    if isinstance(batch[0], Sequence):
        transposed = zip(*batch)
        return [collate(samples, samples_per_gpu) for samples in transposed]
    elif isinstance(batch[0], Mapping):
        return {
            key: mmcv_collate([d[key] for d in batch], samples_per_gpu)
            for key in batch[0]
        }
    else:
        return default_collate(batch)

def build_dataloader(logger, config):
    train_pipeline_ntu = [
        dict(type='Getdata'),
        dict(type='LoadRGB', num_clips=config.DATA.NUM_FRAMES,  train=True),
        dict(type='Flip', flip_ratio=0.5),
        dict(type='ColorJitter', p=config.AUG.COLOR_JITTER),
        dict(type='GrayScale', p=config.AUG.GRAY_SCALE),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='FormatShape', input_format='NCHW'),
        dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
        dict(type='ToTensor', keys=['imgs', 'label']),
    ]
        
    train_data = VideoDataset(ann_file = config.DATA.TRAIN_FILE, 
                              data_prefix = config.DATA.Data_path,
                              labels_file = config.DATA.LABEL_LIST, 
                              pipeline = train_pipeline_ntu, 
                              train = True,
                              BenchMark = config.DATA.BENCHMARK,
                              feature_data_path = config.DATA.ske_feature_data_path, 
                              sample_txt_path = config.DATA.ske_train_sample_txt_path, 
                              bone = config.DATA.bone, 
                              vel = config.DATA.vel)

    if config.DISTRIBUTED == True:
        num_tasks = dist.get_world_size()
        global_rank = dist.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
        train_data, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    else :
        sampler_train = None
    train_loader = DataLoader(
        train_data, sampler = sampler_train,
        batch_size = config.TRAIN.BATCH_SIZE,
        num_workers = config.TRAIN.NUM_WORKERS,
        pin_memory = False, # True debug_xclip
        drop_last = True,
        # collate_fn = partial(mmcv_collate, samples_per_gpu = config.TRAIN.BATCH_SIZE),
    )
    
    val_pipeline = [
        dict(type='Getdata'),
        dict(type='LoadRGB', num_clips=config.DATA.NUM_FRAMES, train=False),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='FormatShape', input_format='NCHW'),
        dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
        dict(type='ToTensor', keys=['imgs'])
    ]
    if config.TEST.NUM_CROP == 3:
        val_pipeline[3] = dict(type='Resize', scale=(-1, config.DATA.INPUT_SIZE))
        val_pipeline[4] = dict(type='ThreeCrop', crop_size=config.DATA.INPUT_SIZE)
    if config.TEST.NUM_CLIP > 1:
        val_pipeline[1] = dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=config.DATA.NUM_FRAMES, multiview=config.TEST.NUM_CLIP)
    
    val_data = VideoDataset(ann_file = config.DATA.VAL_FILE, 
                            data_prefix = config.DATA.Data_path, 
                            labels_file = config.DATA.LABEL_LIST, 
                            pipeline = val_pipeline, 
                            train = False,
                            BenchMark = config.DATA.BENCHMARK,
                            feature_data_path = config.DATA.ske_feature_data_path, 
                            sample_txt_path = config.DATA.ske_val_sample_txt_path, 
                            bone = config.DATA.bone, 
                            vel = config.DATA.vel)

    if config.DISTRIBUTED == True:
        indices = np.arange(dist.get_rank(), len(val_data), dist.get_world_size())
        sampler_val = SubsetRandomSampler(indices)
    else :
        sampler_val = None
    
    if config.TEST.ONLY_TEST == True:
        val_loader = DataLoader(
            val_data, sampler = sampler_val,
            batch_size = config.TEST.BATCH_SIZE,
            num_workers = config.TRAIN.NUM_WORKERS,
            pin_memory = True,
            drop_last = False,
            shuffle = False
        )
    else: # Train
        val_loader = DataLoader(
            val_data, sampler = sampler_val,
            batch_size = config.TEST.BATCH_SIZE,
            num_workers = config.TRAIN.NUM_WORKERS,
            pin_memory = True,
            drop_last = False,
            # collate_fn = partial(mmcv_collate, samples_per_gpu=2),
            shuffle = False
        )

    return train_data, val_data, train_loader, val_loader
