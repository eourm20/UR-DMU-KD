import math
import numpy as np
import torch
import cv2
import os
from feature_extract.i3dpt import I3D
from tqdm import tqdm
from model import WSAD
from time import time
import matplotlib.pyplot as plt


# 연속된 1의 구간을 찾는 함수
def find_continuous_ones(anomaly_info):
    intervals = []
    start = None
    
    for i, value in enumerate(anomaly_info):
        if value == 1 and start is None:
            start = i  # 연속 구간의 시작 지점
        elif value == 0 and start is not None:
            intervals.append((start, i - 1))  # 연속 구간의 끝 지점
            start = None  # 다음 구간을 위해 초기화
    # 마지막 요소가 1인 경우 처리
    if start is not None:
        intervals.append((start, len(anomaly_info) - 1))
    
    return intervals

def forward_batch(b_data,net):
    b_data = b_data.transpose([0, 4, 1, 2, 3])
    b_data = torch.from_numpy(b_data)   # b,c,t,h,w  # 40x3x16x224x224 
    with torch.no_grad():
        b_data = b_data.cuda().float()
        b_features,_ = net(b_data,feature_layer=5)
    b_features = b_features[:,:,0,0,0]
    return b_features
def load_video(path:str):
    frames=[]
    cap=cv2.VideoCapture(path)
    if not cap.isOpened():
        print("video capture open fail")
        exit(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("read over")
            break
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (340, 256)) 
        frame = np.array(frame)
        frame = frame.astype(float)
        frame = (frame * 2 / 255) - 1
        frame = frame[16:240, 58:282, :]
        frames.append(frame)
    return frames


def load_video_dir(path_dir):
    frames=[]
    fs = os.listdir(path_dir)
    fs=fs.sort()
    for f in fs:
        frame = cv2.imread(os.path.join(path_dir,f))
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224)) 
        frame = np.array(frame)
        frame = frame.astype(float)
        frame = (frame * 2 / 255) - 1
        frame = frame[16:240, 58:282, :]
        frames.append(frame)
    return frames
def batch_split(clipped_length,batch_size,chunk_size):
    frame_indices = [] 
    for i in range(clipped_length):
        frame_indices.append(
            [j for j in range(i * 16, i * 16 + chunk_size)])

    frame_indices = np.array(frame_indices)
    chunk_num = frame_indices.shape[0]
    batch_num = int(np.ceil(chunk_num / batch_size))   
    frame_indices = np.array_split(frame_indices, batch_num, axis=0)
    return frame_indices,batch_num

def cv2show(video_path, score_list, normal):
    if normal != True:
        video_info = video_path.split('/')[-1].split('.')[0] + '_frames.txt'
        with open(f'list/test/gt/{video_info}', 'r') as file:
            lines = file.readlines()
            anomaly_info = [int(line.strip()) for line in lines]
    else:
        anomaly_info = [0] * len(score_list)
        
    frame_num = 1
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("video capture open fail")
        exit(0)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height, width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    plot_height = 200
    plot_width = width
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    if normal:
        file_name = 'normal'
    else:
        file_name = 'abnormal'
    video_name = video_path.split('/')[-1].split('.')[0]
    file_name = file_name + '_' + video_name
    out = cv2.VideoWriter(f'video_score/{file_name}.mp4', fourcc, fps, (width, height + plot_height))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("read over")
            break
        
        intervals = find_continuous_ones(anomaly_info)
        
        figsize = (plot_width / 100, plot_height / 100)
        original_frame_indices = np.linspace(0, frame_count, num=len(score_list), endpoint=False)
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_ylim(0, 1)
        ax.set_xlim(0, frame_count)
        
        for start, end in intervals:
            ax.axvspan(start, end, color='red', alpha=0.2)
            
        ax.plot(original_frame_indices[:frame_num], score_list[:frame_num], label='Base Score')
        ax.legend(loc='upper right', fontsize='small')
        fig.canvas.draw()
        
        plot_img_np = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plot_img_np = plot_img_np.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        
        plot_img_resized = cv2.resize(plot_img_np, (width, plot_height))
        
        score = score_list[frame_num-1]
        left_x_up = 10
        left_y_up = 10
        right_x_down = int(left_x_up + 200)
        right_y_down = int(left_y_up + 60)
        word_x = left_x_up + 10
        word_y = left_y_up + 20
        cv2.rectangle(frame, (left_x_up, left_y_up), (right_x_down, right_y_down), (55, 255, 155), 2)
        cv2.putText(frame, 'frame_num:{}'.format(frame_num), (word_x, word_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (55, 255, 155), 1)
        if score > 0.5:
            cv2.putText(frame, 'frame_score:{:.2f}'.format(score), (word_x, word_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 155), 1)
        else:
            cv2.putText(frame, 'frame_score:{:.2f}'.format(score), (word_x, word_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (55, 255, 155), 1)
        
        combined_frame = np.vstack((frame, plot_img_resized))
        
        out.write(combined_frame)
        cv2.imshow('det_res', combined_frame)

        
        frame_num += 1
        key = cv2.waitKey(25)
        if key == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__=="__main__":
    start_time = time()
    batch_size = 10
    i3d = I3D(400, modality='rgb', dropout_prob=0, name='inception')
    i3d.eval()
    i3d.load_state_dict(torch.load("feature_extract/model_rgb.pth"))
    i3d.cuda()

    ad_net = WSAD(input_size = 1024, flag = "Test", a_nums = 60, n_nums = 60)
    ad_net.load_state_dict(torch.load("models/ucf_trans_300.pkl"))
    ad_net.cuda()
    input_dir = "/home/subin-oh/Nas-subin/SB-Oh/data/Anomaly-Detection-Dataset/Train/Burglary/Burglary021_x264.mp4"
    if "Normal" in input_dir.split('/')[-1].split('_')[0]:
        normal = True
    else:
        normal = False
    if os.path.isdir(input_dir):
        frames = load_video_dir(input_dir)
    else:
        frames = load_video(input_dir)
    frames_cnt = len(frames)
    clipped_length = math.ceil(frames_cnt /16)
    copy_length = (clipped_length *16)-frames_cnt
    if copy_length != 0:
        copy_img = [frames[frames_cnt-1]]*copy_length
        frames = frames+copy_img
    frame_indices, batch_num = batch_split(clipped_length, batch_size = batch_size, chunk_size = 16)
    full_features = torch.zeros(0).cuda()
    for batch_id in tqdm(range(batch_num)):
        batch_data = np.zeros(frame_indices[batch_id].shape + (224,224,3))      
        for i in range(frame_indices[batch_id].shape[0]):
            for j in range(frame_indices[batch_id].shape[1]):
                
                batch_data[i,j] = frames[frame_indices[batch_id][i][j]]
        full_features = torch.cat([full_features,forward_batch(batch_data,i3d)], dim = 0)
    print("{} has been extracted. Its shape:{}".format(input_dir,full_features.size()))
    print("---------------------start detecting---------------")
    full_features = full_features.unsqueeze(0)
    res = ad_net(full_features)
    scores = res["frame"].cpu().detach().numpy()
    scores = np.repeat(scores,16)[:-5]
    end_time = time()
    cost_time = end_time - start_time
    print("cost:{}".format(cost_time))
    print("fps:{}".format(frames_cnt/cost_time))
    cv2show(input_dir,scores, normal)
    cv2.destroyAllWindows()
    