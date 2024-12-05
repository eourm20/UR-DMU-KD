import sys, os, distutils.core
sys.path.insert(0, os.path.abspath('./detectron2'))
import cv2
import torch
import numpy as np
from matplotlib import pyplot as plt
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from sklearn.decomposition import PCA
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import euclidean
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

from tqdm import tqdm
import torch
import argparse
from PIL import Image
from torch.autograd import Variable
import numpy as np
from feature_extract.i3dpt import I3D
import math
from scipy.optimize import linear_sum_assignment

import warnings
warnings.filterwarnings("ignore")
import multiprocessing as mp
from multiprocessing import Pool, cpu_count

def filter_matches_by_shoulder_distance(matches, prev_keypoints, curr_keypoints, threshold=7):
    filtered_matches = []
    for match in matches:
        prev_idx, curr_idx = match
        # 왼쪽 어깨의 키포인트 위치를 가져옵니다. 여기서 인덱스 5는 왼쪽 어깨를 가리킵니다.
        prev_shoulder = np.array(prev_keypoints[prev_idx][6])
        curr_shoulder = np.array(curr_keypoints[curr_idx][6])
        
        # 위치 차이를 계산합니다.
        distance = np.linalg.norm(prev_shoulder - curr_shoulder)
        
        # 위치 차이가 임계값보다 작은 경우에만 결과 목록에 추가합니다.
        if distance <= threshold:
            filtered_matches.append(match)
        # print("distance:", distance)
    
    return filtered_matches

# def plot_flow(keypoints_flattened):
#     colors = plt.cm.rainbow(np.linspace(0, 1, len(keypoints_flattened)))
#     # 각 데이터 포인트를 해당 클러스터 색상으로 플롯
#     for point in keypoints_flattened:
#         plt.scatter(point[:, 0], point[:, 1], color=colors[i], label=f'person {i+1}')
#     # x, y축 숫자 크기 조정
#     plt.tick_params(axis='both', labelsize=6)  # 숫자 크기 조정
#     plt.title('Flow Keypoints', fontsize=10)
#     plt.xlabel('Dimension 1', fontsize=6)
#     plt.ylabel('Dimension 2', fontsize=6)
#     plt.legend(fontsize=6)
#     # plt.xlim(-5, 5)
#     # plt.ylim(-5, 5)
#     plt.tight_layout()
#     plt.savefig('plot_frame.png')
#     plt.close()
    
#     return colors
    
def flow_keypoints_with_loss(previous_keypoints, curr_keypoints):
    curr_keypoints_flattened = curr_keypoints.reshape(curr_keypoints.shape[0], -1)
    pre_keypoints_flattened = previous_keypoints.reshape(previous_keypoints.shape[0], -1)
    all_keypoints = np.vstack((pre_keypoints_flattened, curr_keypoints_flattened))
    pca = PCA(n_components=2)
    try:
        pca.fit(all_keypoints)
    except:
        # return 0.0
        
        # 시각화
        return 0.0, None, None  
        
    curr_keypoints_2d = pca.transform(curr_keypoints_flattened)
    pre_keypoints_2d = pca.transform(pre_keypoints_flattened)
    # 이전 프레임과 현재 프레임의 키포인트 간 거리 행렬 계산
    cost_matrix = torch.cdist(torch.tensor(pre_keypoints_2d), torch.tensor(curr_keypoints_2d), p=2).numpy()
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    filtered_matches = filter_matches_by_shoulder_distance(zip(row_ind, col_ind), previous_keypoints, curr_keypoints)
    if len(filtered_matches) == 0:
        # return 0.0
        
        # 시각화
        return 0.0, None, None 
        
    row_ind, col_ind = zip(*filtered_matches)
    row_ind, col_ind = np.array(row_ind), np.array(col_ind)
    # 매칭된 키포인트만 선택
    matched_prev_keypoints = pre_keypoints_2d[row_ind]
    matched_curr_keypoints = curr_keypoints_2d[col_ind]
    
    # 매칭된 키포인트의 변화량 계산
    TFLoss_list = []
    
    for prev, curr in zip(matched_prev_keypoints, matched_curr_keypoints):
        change = torch.norm(torch.tensor(curr - prev), dim=0)
        TFLoss_list.append(change)
    # print(TFLoss_list)
    TFLoss_= float(torch.mean(torch.stack(TFLoss_list)))
    
    # 시각화
    plt.figure(figsize=(2.24, 2.24))
    colors = plt.cm.rainbow(np.linspace(0, 1, max(len(row_ind), len(col_ind))))
    for i in range(len(row_ind)):
        point = pre_keypoints_2d[i]
        plt.scatter(point[0], point[1], color=colors[i], label=f'pre person {i+1}', s=10)
    
    for i in range(len(col_ind)):
        point = curr_keypoints_2d[i]
        plt.scatter(point[0], point[1], color=colors[i], label=f'curr person {i+1}', s=10)

    plt.title('Flow Keypoints', fontsize=10)
    plt.xlabel('Dimension 1', fontsize=6)
    plt.ylabel('Dimension 2', fontsize=6)
    # plt.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig('plot_frame.png')
    plt.close()
    
    return TFLoss_, colors, (row_ind, col_ind)
    
    # return TFLoss_
    
    


class Detectron2Pose:
    def __init__(self):
        # Set up the configuration and load a pre-trained model from Detectron2's model zoo
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml"))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml")
        self.predictor = DefaultPredictor(self.cfg)

    # 각각의 사람들의 행동을 추측할 수 있는 주요 키포인트를 사용하여 각각의 사람들의 임베딩값 추출
    def get_pose(self, frame):
        # Convert BGR image to RGB (Matplotlib uses RGB images)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # Make prediction on the current frame and extract keypoints predictions 
        outputs = self.predictor(frame)
        vis_person = []
        filtered_person = []
        if "instances" in outputs:
            instances = outputs["instances"].to(torch.device('cpu'))
            if instances.has("pred_keypoints"):
                keypoints_predictions = instances.pred_keypoints
                # print(keypoints_predictions)
                # [x,y,신뢰도]
                if len(keypoints_predictions) == 0:
                    # print('No person detected')
                    return filtered_person, vis_person
                    # return filtered_person
                else:
                    # 필터링할 keypoint 인덱스, COCO index는 0부터 시작하므로 1을 빼줍니다.
                    indices = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
                    # 왼쪽 어깨
                    # 오른쪽 어깨
                    # 왼쪽 팔꿈치
                    # 오른쪽 팔꿈치
                    # 왼쪽 손목
                    # 오른쪽 손목
                    # 왼쪽 골반
                    # 오른쪽 골반
                    # 왼쪽 무릎
                    # 오른쪽 무릎
                    # 왼쪽 발목
                    # 오른쪽 발목
                    for keypoints in keypoints_predictions:
                        # 신뢰도가 0.8 이상인 keypoint만 선택
                        keypoints = keypoints[indices]
                        # 모든 키포인트의 신뢰도가 0.8 이상인지 확인
                        if all(kp[2] >= 0.02 for kp in keypoints):
                            keypoints_xy = keypoints[:, :2]  # 첫 번째와 두 번째 요소(x, y)만 선택
                            center_point_indies = [0, 1, 6, 7] 
                            center_point = keypoints_xy[center_point_indies].mean(dim=0)
                            relative_keypoints = keypoints_xy - center_point  # 전체 키포인트에서 중심점 좌표를 뺌
                            
                            # Normalization
                            # 각 keypoint 사이의 최대 거리를 계산
                            max_distance = torch.max(torch.cdist(relative_keypoints, relative_keypoints, p=2))
                            normalized_keypoints = relative_keypoints / max_distance  # 모든 위치값을 최대 거리로 나눔
                            # 양옆 골반의 차이값 추가
                            left_distance = normalized_keypoints[0]-normalized_keypoints[6]
                            right_distance = normalized_keypoints[1]-normalized_keypoints[7]
                            # 총 데이터 포인트
                            normalized_keypoints = np.vstack((relative_keypoints,left_distance, right_distance))
                            
                            # 시각화
                            vis_person.append(keypoints_xy)
                            
                            filtered_person.append(normalized_keypoints)
                    if len(filtered_person) <= 1:
                        filtered_person = np.array(filtered_person)
                        # print('No person detected with high confidence keypoints')
                        # filtered_person = np.zeros((13, 3))
                        # return filtered_person
                        
                        # 시각화
                        return filtered_person, vis_person
                        
                    else:
                        filtered_person = np.array(filtered_person)
                        # return filtered_person
                        
                        # 시각화
                        return filtered_person, vis_person
                        
                
    def normalize(self, frame):
        # Find the minimum and maximum of the array
        min_val = frame.min()
        max_val = frame.max()

        # Normalize to [0, 1]
        normalized_frame = (frame - min_val) / (max_val - min_val)

        # Scale to [0, 255] and convert to uint8
        scaled_frame = (normalized_frame * 255).astype('uint8')  
        return scaled_frame

def load_frame_cv2(frame):
    # 프레임을 RGB로 변환
    # data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 크기 조정
    data = cv2.resize(frame, (340, 256), interpolation=cv2.INTER_LINEAR)
    # 데이터 타입 변환 및 정규화
    # data = data.astype(float)
    # data = (data * 2 / 255) - 1
    # 조건 확인
    # assert(data.max() <= 1.0)
    # assert(data.min() >= -1.0)
    return data

def oversample_data_single_img(data):
    # 단일 이미지 데이터의 형상 변경: (height, width, channels) -> (1, 1, height, width, channels)
    data = data.reshape((1, 1) + data.shape)
    # 데이터 오버샘플링 로직 적용
    data_flip = np.array(data[:, :, :, ::-1, :])

    # data_1 = np.array(data[:, :, :224, :224, :])# 왼쪽 위
    # data_2 = np.array(data[:, :, :224, -224:, :])# 오른쪽 위
    data_3 = np.array(data[:, :, 16:240, 58:282, :])#가운데
    # data_4 = np.array(data[:, :, -224:, :224, :])#왼쪽아래
    # data_5 = np.array(data[:, :, -224:, -224:, :])#오른쪽 아래

    # data_f_1 = np.array(data_flip[:, :, :224, :224, :])
    # data_f_2 = np.array(data_flip[:, :, :224, -224:, :])
    data_f_3 = np.array(data_flip[:, :, 16:240, 58:282, :])
    # data_f_4 = np.array(data_flip[:, :, -224:, :224, :])
    # data_f_5 = np.array(data_flip[:, :, -224:, -224:, :])

    # return [data_3, data_1, data_2, data_4, data_5, data_f_3, data_f_1, data_f_2, data_f_4, data_f_5]
    return [data_3, data_f_3]

# # 예제 데이터 생성
# data = np.random.rand(39, 16, 224, 224, 3)  # 3채널 RGB 데이터 예제

# # 배치 데이터 오버샘플링
# batch_data_ten_crop = oversample_data(data)

# # 첫 번째 프레임 시각화
# show_frames(batch_data_ten_crop, frame_idx=0)

def convert_to_bgr(person_colors):
    bgr_colors = []
    for color in person_colors:
        r, g, b, a = color  # RGBA 순서
        bgr = (int(b * 255), int(g * 255), int(r * 255))  # BGR 순서로 변환
        bgr_colors.append(bgr)
    return bgr_colors

def process_video(video, crop):
    # print(f"Processing {video}")
    pose_model = Detectron2Pose()
    # video_name = video.split('/')[-2]+'/'+video.split('/')[-1].split(".")[0]
    video_name = video.split('/')[-1].split(".mp4")[0]
    # 파일 이름에 video_name이 포함되어 있는지 확인
    path = f"/home/subin-oh/Nas-subin/SB-Oh/data/HPE/XD_ACT/"
    if os.path.exists(path) == False:
        os.makedirs(path)
    # name이 포함된 파일 리스트
    name = video_name
    list = os.listdir(path)
    crop_check=[]
    for line in list:
        if name in line:
            crop_check.append(line)
    if crop == 0:
        for crop_name in crop_check:
            if '__' not in crop_name:
                return
    elif crop == 1:
        for crop_name in crop_check:
            if '__' in crop_name:
                if int(crop_name.split('__')[-1].split('.')[0]) == 5:
                    return
    
    video_cap = cv2.VideoCapture(video)
    if not video_cap.isOpened():
        print("Error opening video file")
        return
    
    # 시각화
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    total_frame = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print("total frame:", total_frame)
    
    # #test만 사용 가능
    try:
        np_frame = np.load(f"/home/subin-oh/Nas-subin/SB-Oh/data/XD-Violence/i3d-features/RGBTest/{video_name}__0.npy")

        print("np_frame shape:", np_frame.shape)
        
        annotation_abnormal = open(f"/home/subin-oh/code/WVED_1101/main/UR-DMU-KD/list/XD_Annotation.txt", 'r')
        abnormals = [0, 0]
        for line in annotation_abnormal:
            if video_name in line:
                abnormals = line[:-1].split(" ")[1:]
                break
            else:
                abnormals = [0, 0]
    except:
        return
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height, width = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))*2-50, int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))*2
    out = cv2.VideoWriter(f'tfloss_vis/XD/{video_name}_16f_W0_01.mp4', fourcc, fps, (448, 398))
    
    weight = 0.01
    tfloss_results = {}
    previous_keypoints = {}
    curr_keypoints = {}
    tfloss_results[crop] = []
    previous_keypoints[crop] = None
    curr_keypoints[crop] = None
    
    # 시각화
    previous_vis_keypoints = {}
    curr_vis_keypoints = {}
    
    frame_count = 0
    while video_cap.isOpened():
        ret, frame = video_cap.read()
        if ret:
            if frame_count % 16 ==0 and frame_count != 0:
                # i = 1
                # 배치 데이터 오버샘플링
                data = load_frame_cv2(frame)
                oversampled_data = oversample_data_single_img(data)
                crop_frame = oversampled_data[crop][0][0]
                if previous_keypoints[crop] is None:
                    tfloss = 0.0 * weight
                    tfloss_results[crop].append(tfloss)
                    # previous_keypoints[crop] = pose_model.get_pose(crop_frame)
                    
                    # 시각화
                    previous_keypoints[crop], previous_vis_keypoints[crop] = pose_model.get_pose(crop_frame)
                    
                else:
                    # curr_keypoints[crop] = pose_model.get_pose(crop_frame)
                    
                    # 시각화
                    curr_keypoints[crop], curr_vis_keypoints[crop] = pose_model.get_pose(crop_frame)
                    
                    colors = None
                    if len(curr_keypoints[crop]) != 0 and len(previous_keypoints[crop]) != 0:
                        # tfloss = flow_keypoints_with_loss(previous_keypoints[crop], curr_keypoints[crop])
                        
                        # 시각화
                        tfloss, colors, matching_ind = flow_keypoints_with_loss(previous_keypoints[crop], curr_keypoints[crop])
                        
                        tfloss = tfloss * weight
                    else:
                        tfloss = 0.0 * weight
                    # 소수점 셋째 자리까지 반올림
                    tfloss = round(tfloss, 3)
                    tfloss_results[crop].append(tfloss)
                    previous_keypoints[crop] = curr_keypoints[crop]
                    
                    # 시각화
                    if colors is not None:
                        colors = convert_to_bgr(colors)
                        #index matching
                        for k in range(len(matching_ind[0])):
                            ind = matching_ind[0][k]
                            person = previous_vis_keypoints[crop][ind]
                            for kp in person:
                                cv2.circle(crop_frame, (int(kp[0]), int(kp[1])), 3, colors[k], -1)
                        for p in range(len(matching_ind[1])):
                            ind = matching_ind[1][p]
                            person = curr_vis_keypoints[crop][ind]
                            for kp in person:
                                cv2.circle(crop_frame, (int(kp[0]), int(kp[1])), 3, colors[p], -1)
                    previous_vis_keypoints[crop] = curr_vis_keypoints[crop]
                    

                    if os.path.exists('plot_frame.png'):
                        plot_image = cv2.imread('plot_frame.png')
                        os.remove('plot_frame.png')
                    else:
                        plt.figure(figsize=(2.24, 2.24))
                        # x, y축 숫자 크기 조정
                        plt.tick_params(axis='both', labelsize=6)  # 숫자 크기 조정
                        plt.title('Flow Keypoints', fontsize=10)
                        plt.xlabel('Dimension 1', fontsize=6)
                        plt.ylabel('Dimension 2', fontsize=6)
                        plt.xlim(-5, 5)
                        plt.ylim(-5, 5)
                        plt.tight_layout()
                        plt.savefig('plot_frame.png')
                        plt.close()
                        plot_image = cv2.imread('plot_frame.png')
                        os.remove('plot_frame.png')
                    # 필요하다면 plot_image의 크기를 조정
                    # plot_image_resized = cv2.resize(plot_image, (crop_frame.shape[1], crop_frame.shape[0]))
                    plot_image_resized = plot_image.copy()
                    # 프레임과 플롯 이미지를 수평으로 결합
                    combined_frame = np.hstack((crop_frame, plot_image_resized))
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    org = (10, 50)
                    fontScale = 1
                    colors = (255, 0, 0)
                    thickness = 2
                    cv2.putText(combined_frame, 'ACT: {:.2f}'.format(tfloss), org, font, fontScale, colors, thickness, cv2.LINE_AA)
                    
                    figsize = (crop_frame.shape[1]*2/100, ((crop_frame.shape[0])-50)/100)
                    fig, ax = plt.subplots(figsize=figsize)
                    ax.set_ylim(0, 1)
                    ax.set_xlim(0, np_frame.shape[0])
                    ax.plot(tfloss_results[crop])
                    for key in range(len(abnormals)//2):
                        ax.axvspan(int(abnormals[key])//16, int(abnormals[key+1])//16, color='red', alpha=0.2)
                    # ax.set_xlabel('Time')
                    # ax.set_ylabel('TFLoss')
                    ax.set_title('Action Change Tracking Loss', fontsize=10)
                    plt.tight_layout()
                    fig.canvas.draw()
                    
                    plot_img_np = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                    plot_img_np = plot_img_np.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    plt.close(fig)
                    plot_img_resized = cv2.resize(plot_img_np,  (crop_frame.shape[1]*2, crop_frame.shape[0]-50))
                    combined_frame = np.vstack((combined_frame, plot_img_resized))
                    out.write(combined_frame)
                    # 오버샘플링된 프레임을 개별 창에 표시
                    cv2.imshow(f'Frame - Crop {crop}', combined_frame)
                    
            
            key = cv2.waitKey(25)
            if key == ord('q'):
                break
            
            frame_count += 1
        else:
            break
    
    # 시각화
    out.release()
    cv2.destroyAllWindows()
    
    video_cap.release()

    if crop == 0:
        crop_ = ""
    else:
        crop_ = "__5"

    np.save(f"{path}{name}{crop_}.npy", tfloss_results[crop])

if __name__ == '__main__':
    crop = 0
    print(f"{str(crop)} crop")
    split_file = []
    # s = open('list/ucf-train.list', 'r')
    s = open('/home/subin-oh/code/WVED_1101/main/UR-DMU-KD/list/XD_Train.list', 'r')
    for line in s:
        line = line.strip()
        split_file.append(line)

    
    # split_file.reverse()
    vid_list = []
    
    for video_path in split_file:
        # 파일 이름에 video_name이 포함되어 있는지 확인
        video_name = line.split('/')[-1].split(".mp4")[0]
        path = f"/home/sb-oh/Nas-subin/SB-Oh/data/HPE/XD_ACD/"
        if os.path.exists(path) == False:
            os.makedirs(path)
        # name이 포함된 파일 리스트
        name = video_name
        list = os.listdir(path)
        # crop_num=[]
        crop_check=[]
        no_add = False
        for line in list:
            if name in line:
                crop_check.append(line)

        if crop == 0:
            for crop_name in crop_check:
                if '__' not in crop_name:
                    no_add = True
                    break
        elif crop == 1:
            for crop_name in crop_check:
                if '__' in crop_name:
                    if int(crop_name.split('__')[-1].split('.')[0]) == 5:
                        # crop_num.append(0)
                        no_add = True
                        break
                
        if no_add == False:
            vid_list.append(video_path)
        
    for q in range(len(vid_list)):
        video = vid_list[q]
        print(q,"/",len(vid_list),end="\r")
        process_video(video, crop)   
    
    '''
    annotation_abnormal = open(f"/home/subin-oh/code/WVED/HPE-Extract/UR-DMU-KD/list/UCF_Annotation.txt", 'r')
    test =[]
    for line in annotation_abnormal:
        file = line.split(" ")[0]
        if 'Normal' in file:
            path_file = '/home/subin-oh/Nas-subin/SB-Oh/data/Anomaly-Detection-Dataset/Test/'+file
        else:
            path_file = '/home/subin-oh/Nas-subin/SB-Oh/data/Anomaly-Detection-Dataset/Train/'+file
        mp4 = file.split("/")[1].split('.')[0]
        mp4 = f'{mp4}_16f_W0_2.mp4'
        if mp4 in os.listdir('ohloss_vis/NEW'):
            continue
        test.append(path_file)
    # test_list = ['Abuse/Abuse030_x264.mp4',
    #              'Arrest/Arrest024_x264.mp4',
    #              'Arson/Arson016_x264.mp4',
    #              'Assault/Assault011_x264.mp4',
    #              'Burglary/Burglary079_x264.mp4',
    #              'Explosion/Explosion017_x264.mp4',
    #              'Fighting/Fighting018_x264.mp4',
    #              'RoadAccidents/RoadAccidents002_x264.mp4',
    #              'Robbery/Robbery106_x264.mp4',
    #              'Shooting/Shooting048_x264.mp4',
    #              'Shoplifting/Shoplifting031_x264.mp4',
    #              'Stealing/Stealing036_x264.mp4',
    #              'Vandalism/Vandalism036_x264.mp4']
    for video in test:
        print(video.split('/')[-2]+'/'+video.split('/')[-1].split(".")[0])
        process_video(video, 0)
        
    '''
    # process_video('/home/subin-oh/Nas-subin/SB-Oh/data/Anomaly-Detection-Dataset/Train/Fighting/Fighting033_x264.mp4', 0)
    process_video('/home/subin-oh/Nas-subin/SB-Oh/data/XD-Violence/Test/videos/Bullet.in.the.Head.1990__#01-26-30_01-27-34_label_B2-0-0.mp4', 0)