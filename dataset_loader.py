import torch
import torch.utils.data as data
import os
import numpy as np
import utils 


class UCF_crime(data.DataLoader):
    def __init__(self, root_dir, modal, mode, num_segments, len_feature, seed=-1, is_normal=None):
        if seed >= 0:
            utils.set_seed(seed)
        self.mode = mode
        self.modal = modal
        self.num_segments = num_segments
        self.len_feature = len_feature
        split_path = os.path.join('list','KD/label_25/ucf-label-i3d_{}.list'.format(self.mode))
        split_file = open(split_path, 'r')
        self.vid_list = []
        for line in split_file:
            self.vid_list.append(line.split())
        split_file.close()
        if self.mode == "Train":
            if is_normal is True:
                self.vid_list = self.vid_list[104:]
            elif is_normal is False:
                self.vid_list = self.vid_list[:104]
            else:
                assert (is_normal == None)
                print("Please sure is_normal=[True/False]")
                self.vid_list=[]
    def __len__(self):
        return len(self.vid_list)

    def __getitem__(self, index):
        
        if self.mode == "Test":
            data,label,name = self.get_data(index)
            return data,label,name
        else:
            data,label = self.get_data(index)
            return data,label

    def get_data(self, index):
        vid_info = self.vid_list[index][0]  
        name = vid_info.split("/")[-1].split("_x264")[0]
        video_feature = np.load(vid_info).astype(np.float32)   

        if "Normal" in vid_info.split("/")[-1]:
            label = 0
        else:
            label = 1
        if self.mode == "Train":
            new_feat = np.zeros((self.num_segments, video_feature.shape[1])).astype(np.float32)
            r = np.linspace(0, len(video_feature), self.num_segments + 1, dtype = int)
            for i in range(self.num_segments):
                if r[i] != r[i+1]:
                    new_feat[i,:] = np.mean(video_feature[r[i]:r[i+1],:], 0)
                else:
                    new_feat[i:i+1,:] = video_feature[r[i]:r[i]+1,:]
            video_feature = new_feat
        if self.mode == "Test":
            return video_feature, label, name      
        else:
            return video_feature, label      

class Unlabeled_UCF_crime(data.DataLoader):
    def __init__(self, root_dir, modal, num_segments, len_feature, seed=-1):
        if seed >= 0:
            utils.set_seed(seed)
        self.modal = modal
        self.num_segments = num_segments
        self.len_feature = len_feature
<<<<<<< HEAD
        origin_split_path = os.path.join('list','KD/unlabel_50/ucf-unlabel-i3d.list')
        weak_aug_split_path = os.path.join('list','KD/unlabel_50/ucf-unlabel-i3d_5.list')
=======
        origin_split_path = os.path.join('list','KD/unlabel_50/ucf-unlabel-i3d_svr3.list')
        weak_aug_split_path = os.path.join('list','KD/unlabel_50/ucf-unlabel-i3d_5_svr3.list')
>>>>>>> 958ac30 (50 dark)
        origin_split_file = open(origin_split_path, 'r')
        weak_aug_split_file = open(weak_aug_split_path, 'r')
        self.origin_vid_list = []
        self.weak_aug_vid_list = []
        for line in origin_split_file:
            self.origin_vid_list.append(line.split())
        for line in weak_aug_split_file:
            self.weak_aug_vid_list.append(line.split())
        origin_split_file.close()
        weak_aug_split_file.close()
    def __len__(self):
        return len(self.origin_vid_list)

    def __getitem__(self, index):
        # aug_features = np.empty(0)
        data,aug_data = self.get_data(index)
        return data,aug_data

    def get_data(self, index):
        origin_vid_info = self.origin_vid_list[index][0]
        weak_aug_vid_info = self.weak_aug_vid_list[index][0]
        name = origin_vid_info.split("/")[-1].split("_x264")[0]
        origin_video_feature = np.load(origin_vid_info).astype(np.float32)
        weak_aug_video_feature = np.load(weak_aug_vid_info).astype(np.float32) 

        # if "Normal" in origin_video_feature.split("/")[-1]:
        #     label = 0
        # else:
        #     label = 1
        origin_new_feat = np.zeros((self.num_segments, origin_video_feature.shape[1])).astype(np.float32)
        weak_aug_new_feat = np.zeros((self.num_segments, weak_aug_video_feature.shape[1])).astype(np.float32)
        origin_r = np.linspace(0, len(origin_video_feature), self.num_segments + 1, dtype = int)
        weak_aug_r = np.linspace(0, len(weak_aug_video_feature), self.num_segments + 1, dtype = int)
        for i in range(self.num_segments):
            if origin_r[i] != origin_r[i+1]:
                origin_new_feat[i,:] = np.mean(origin_video_feature[origin_r[i]:origin_r[i+1],:], 0)
            else:
                origin_new_feat[i:i+1,:] = origin_video_feature[origin_r[i]:origin_r[i]+1,:]
                
            if weak_aug_r[i] != weak_aug_r[i+1]:
                weak_aug_new_feat[i,:] = np.mean(weak_aug_video_feature[weak_aug_r[i]:weak_aug_r[i+1],:], 0)
            else:
                weak_aug_new_feat[i:i+1,:] = weak_aug_video_feature[weak_aug_r[i]:weak_aug_r[i]+1,:]
        origin_video_feature = origin_new_feat
        weak_aug_video_feature = weak_aug_new_feat
        
        return origin_video_feature, weak_aug_video_feature     
        
class XDVideo(data.DataLoader):
    def __init__(self, root_dir, mode, modal, num_segments, len_feature, seed=-1, is_normal=None):
        if seed >= 0:
            utils.set_seed(seed)
        self.data_path=root_dir
        self.mode=mode
        self.modal=modal
        self.num_segments = num_segments
        self.len_feature = len_feature
        if self.modal == 'all':
            self.feature_path = []
            if self.mode == "Train":
                for _modal in ['RGB', 'Flow']:
                    self.feature_path.append(os.path.join(self.data_path, "i3d-features",_modal))
            else:
                for _modal in ['RGBTest', 'FlowTest']:
                    self.feature_path.append(os.path.join(self.data_path, "i3d-features",_modal))
        else:
            self.feature_path = os.path.join(self.data_path, modal)
        split_path = os.path.join("list",'XD_{}.list'.format(self.mode))
        split_file = open(split_path, 'r',encoding="utf-8")
        self.vid_list = []
        for line in split_file:
            self.vid_list.append(line.split())
        split_file.close()
        if self.mode == "Train":
            if is_normal is True:
                self.vid_list = self.vid_list[9525:]
            elif is_normal is False:
                self.vid_list = self.vid_list[:9525]
            else:
                assert (is_normal == None)
                print("Please sure is_normal = [True/False]")
                self.vid_list=[]
        
    def __len__(self):
        return len(self.vid_list)

    def __getitem__(self, index):
        data,label = self.get_data(index)
        return data, label

    def get_data(self, index):
        vid_name = self.vid_list[index][0]
        label=0
        if "_label_A" not in vid_name:
            label=1  
        video_feature = np.load(os.path.join(self.feature_path[0],
                                vid_name )).astype(np.float32)
        if self.mode == "Train":
            new_feature = np.zeros((self.num_segments,self.len_feature)).astype(np.float32)
            sample_index = utils.random_perturb(video_feature.shape[0],self.num_segments)
            for i in range(len(sample_index)-1):
                if sample_index[i] == sample_index[i+1]:
                    new_feature[i,:] = video_feature[sample_index[i],:]
                else:
                    new_feature[i,:] = video_feature[sample_index[i]:sample_index[i+1],:].mean(0)
                    
            video_feature = new_feature
        return video_feature, label    
