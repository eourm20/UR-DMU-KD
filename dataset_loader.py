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
        if self.mode == "Train":
            split_path = os.path.join('list','UCF_{}_center_svr3.list'.format(self.mode))
        elif self.mode == "Test":
            split_path = os.path.join('list','UCF_{}_svr3.list'.format(self.mode))
            # split_path = os.path.join('list','KD/pretrain/ucf-label-i3d_svr3_{}.list'.format(self.mode))
        # split_path = os.path.join('list','UCF_{}_svr3.list'.format(self.mode))
        split_file = open(split_path, 'r')
        self.vid_list = []

        # self.OHloss_list = []
        # self.TFloss_list = []
        for line in split_file:
            self.vid_list.append(line.split())
            # self.OHloss_list.append(line.split().replace(f'I3D_feature/{self.mode}/RGB','OHLoss_np3'))
            # self.TFloss_list.append(line.split().replace(f'I3D_feature/{self.mode}/RGB','TFLoss_np3'))
        split_file.close()
        if self.mode == "Train":
            if is_normal is True:
                self.vid_list = self.vid_list[810:]
            elif is_normal is False:
                self.vid_list = self.vid_list[:810]
            else:
                assert (is_normal == None)
                print("Please sure is_normal=[True/False]")
                self.vid_list=[]
                # self.OHloss_list=[]
                # self.TFloss_list=[]
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
        # ohloss_info = self.OHloss_list[index][0]
        # tfloss_info = self.TFloss_list[index][0]
        name = vid_info.split("/")[-1].split("_x264")[0]
        video_feature = np.load(vid_info).astype(np.float32)
        # ohloss = np.load(ohloss_info).astype(np.float32)
        # tfloss = np.load(tfloss_info).astype(np.float32)
        
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

class XDVideo(data.DataLoader):
    def __init__(self, root_dir, mode, modal, num_segments, len_feature, seed=-1, is_normal=None):
        if seed >= 0:
            utils.set_seed(seed)
        self.data_path=root_dir
        self.mode=mode
        self.modal=modal
        self.num_segments = num_segments
        self.len_feature = len_feature
        
        if self.mode == "Train":
            split_path = os.path.join('list','XD_{}_center_newdata.list'.format(self.mode))
        elif self.mode == "Test":
            split_path = os.path.join('list','XD_{}.list'.format(self.mode))
            # split_path = os.path.join('list','XD_{}_center.list'.format(self.mode))
        # split_path = os.path.join('list','UCF_{}_svr3.list'.format(self.mode))
        split_file = open(split_path, 'r', encoding='utf-8')
        self.vid_list = []
        
        for line in split_file:
            self.vid_list.append(line.split())
        split_file.close()
        if self.mode == "Train":
            if is_normal is True:
                self.vid_list = self.vid_list[1905:]
            elif is_normal is False:
                self.vid_list = self.vid_list[:1905]
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
        video_feature = np.load(vid_name).astype(np.float32)
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
