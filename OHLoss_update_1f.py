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


import warnings
warnings.filterwarnings("ignore")
import multiprocessing as mp
from multiprocessing import Pool, cpu_count

def dunn_index(keypoints_2d, labels, centers):
    """
    Dunn 지수를 계산하는 함수
    """
    # 클러스터 간 최소 거리 계산
    min_intercluster_distance = np.inf
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            distance = euclidean(centers[i], centers[j])
            if distance < min_intercluster_distance:
                min_intercluster_distance = distance
    
    # 클러스터 내 최대 거리 계산
    max_intracluster_distance = -np.inf
    for i in range(len(centers)):
        cluster_points = keypoints_2d[labels == i]
        for point in cluster_points:
            for other_point in cluster_points:
                distance = euclidean(point, other_point)
                if distance > max_intracluster_distance:
                    max_intracluster_distance = distance
    # 분모가 0이 되는 상황 처리
    if max_intracluster_distance == 0:
        return np.inf  # 또는 적절한 오류 메시지 반환 또는 다른 값을 반환
    # Dunn 지수 계산
    dunn = min_intercluster_distance / max_intracluster_distance
    return dunn

def find_optimal_clusters(data):
    """
    데이터에 대해 최적의 클러스터 수를 찾는 함수
    data: 클러스터링할 데이터
    max_clusters: 최대 클러스터 수
    """
    max_clusters = data.shape[0]
    if max_clusters <= 2:
        if max_clusters == 1:
            return 1
        else:
            distance = np.linalg.norm(data[0] - data[1])
            if distance < 0.5:
                return 1
            else:
                return 2
    else:
        # 모든 데이터 사이의 유사도 계산
        # 모든 관계에서 70이하의 거리가 있으면 return 1
        for i in range(max_clusters):
            for j in range(i+1, max_clusters):
                distance = np.linalg.norm(data[i] - data[j])
                if distance >= 0.5:
                    break
                else:
                    if i == max_clusters-1 and j == max_clusters-1:
                        return 1

        
    iters = range(2, max_clusters)
    dunn_scores = []
    
    for k in iters:
        kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_
        dunn = dunn_index(data, labels, centers)
        dunn_scores.append(dunn)
        # s_scores.append(silhouette_score(data, kmeans.labels_))
    
    # 실루엣 스코어가 최대가 되는 클러스터 수 선택
    optimal_clusters = iters[np.argmax(dunn_scores)]
    return optimal_clusters


# 시각화
def plot_clusters(keypoints_flattened, labels, centers):
    """
    클러스터링 결과를 시각화하는 함수
    keypoints_flattened: 평탄화된 키포인트 데이터 배열
    labels: 각 데이터 포인트의 클러스터 레이블 배열
    centers: 클러스터 중심점 배열
    """
    colors = plt.cm.rainbow(np.linspace(0, 1, len(centers)))
    # 각 데이터 포인트를 해당 클러스터 색상으로 플롯
    for i in range(len(centers)):
        cluster_points = keypoints_flattened[labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[i], label=f'Cluster {i+1}')
        plt.scatter(centers[i, 0], centers[i, 1], color=colors[i], marker='x', s=200)
    
    plt.title('Clustered Keypoints')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.savefig('plot_frame.png')
    plt.close()
    
    return colors

def cluster_keypoints_with_loss(keypoints):
    """
    키포인트 데이터를 클러스터링하고, 로스 함수를 계산하는 함수
    keypoints: 사람들의 키포인트 데이터를 담고 있는 배열, shape=(n_people, 17, 3)
               각 키포인트는 (x, y, confidence) 형태의 데이터를 가짐
    max_clusters: 최대 클러스터 수
    
    반환값: 각 사람이 속한 클러스터의 인덱스 배열, 로스 값
    """
    # 키포인트 데이터를 (n_people, 24) 형태의 배열로 변환
    keypoints_flattened = keypoints.reshape(keypoints.shape[0], -1)
    
    pca = PCA(n_components=2)
    try:
        keypoints_2d = pca.fit_transform(keypoints_flattened)
    except:
        # return 0
        
        # 시각화
        return 0, colors, labels
        
    # 최적의 클러스터 수 찾기
    optimal_clusters = find_optimal_clusters(keypoints_2d)
    
    
    # K-means 클러스터링 수행
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=0).fit(keypoints_2d)
    
    # 각 사람이 속한 클러스터의 인덱스 및 클러스터 중심점 계산
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    
    # 시각화
    colors = plot_clusters(keypoints_2d, labels, centers)
    
    if optimal_clusters == 1:
        # return 0
        
        # 시각화
        return 0, colors, labels
        
    cluster_counts = np.bincount(labels)
    
    # 가장 적은 인원을 가진 클러스터들의 인덱스와 그 인원 수 확인
    min_count_indices = np.where(cluster_counts == np.min(cluster_counts))[0]
    
    # 초기화
    max_loss = -np.inf
    abnormal_cluster = None
    
    # 모든 클러스터에 대해 반복
    for idx in min_count_indices:
        current_loss = 0
        # 현재 클러스터와 나머지 클러스터들과의 거리(로스) 계산
        for i, center in enumerate(centers):
            if i == idx:
                continue
            loss = np.linalg.norm(center - centers[idx])
            current_loss += loss
        
        # 현재 클러스터의 로스가 지금까지 계산된 최대 로스보다 크면 업데이트
        if current_loss > max_loss:
            max_loss = current_loss
            abnormal_cluster = idx
    
    # 최종적으로 선택된 abnormal_cluster와 나머지 클러스터들과의 평균 거리(로스) 계산
    final_loss = 0
    for i, center in enumerate(centers):
        if i == abnormal_cluster:
            continue
        final_loss += np.linalg.norm(center - centers[abnormal_cluster])
    
    # 평균 로스 반환    
    # return final_loss*(np.max(cluster_counts)) / len(centers)
    
    # 시각화
    return final_loss*(np.max(cluster_counts)) / len(centers), colors, labels
    
    # # Dunn 지수 계산
    # dunn = dunn_index(keypoints_2d, labels, centers)
    # # 같은 행동을 하는 사람들이 많은 경우 소수의 클러스터는 이상
    # OHLoss = dunn * np.max(cluster_counts)
    # return OHLoss, colors, labels

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
                    # return filtered_person
                    
                    # 시각화
                    return filtered_person, vis_person
                     
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
                        # if all(kp[2] >= 0.02 for kp in keypoints):
                        keypoints_xy = keypoints[:, :2]  # 첫 번째와 두 번째 요소(x, y)만 선택
                        center_point_indies = [0, 1, 6, 7] 
                        center_point = keypoints_xy[center_point_indies].mean(dim=0)
                        relative_keypoints = keypoints_xy - center_point  # 전체 키포인트에서 중심점 좌표를 뺌
                        
                        # Normalization
                        # 각 keypoint 사이의 최대 거리를 계산
                        max_distance = torch.max(torch.cdist(relative_keypoints, relative_keypoints, p=2))
                        normalized_keypoints = relative_keypoints / max_distance  # 모든 위치값을 최대 거리로 나눔
                        
                        # 시각화
                        vis_person.append(keypoints_xy)
                        
                        filtered_person.append(normalized_keypoints)
                    if len(filtered_person) <= 1:
                        # print('No person detected with high confidence keypoints')
                        # filtered_person = np.zeros((13, 3))
                        
                        # return filtered_person
                        
                        # 시각화
                        return filtered_person, vis_person
                        
                    else:
                        filtered_person = torch.stack(filtered_person, dim=0)
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

def convert_to_bgr(person_colors):
    bgr_colors = []
    for color in person_colors:
        r, g, b, a = color  # RGBA 순서
        bgr = (int(b * 255), int(g * 255), int(r * 255))  # BGR 순서로 변환
        bgr_colors.append(bgr)
    return bgr_colors

def process_video(video, crop):
    pose_model = Detectron2Pose()
    video_name = video.split('/')[-2]+'/'+video.split('/')[-1].split(".")[0]
    
    # 파일 이름에 video_name이 포함되어 있는지 확인
    path = f"/home/subin-oh/Nas-subin/SB-Oh/data/HPE/OHLoss_np4/{video_name.split('/')[0]}/"
    if os.path.exists(path) == False:
        os.makedirs(path)
    # name이 포함된 파일 리스트
    name = video_name.split('/')[1]
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
        np_frame = np.load(f"/home/subin-oh/Nas-subin/SB-OH/data/I3D_feature/Test/RGB/{video_name}.npy")

        print("np_frame shape:", np_frame.shape)
        
        annotation_abnormal = open(f"/home/subin-oh/code/WVED/HPE-Extract/UR-DMU-KD/list/UCF_Annotation.txt", 'r')
        abnormals = [0, 0]
        if 'Normal' not in video_name:
            for line in annotation_abnormal:
                if video_name in line:
                    abnormals = [int(line.split()[-4]), int(line.split()[-3])]
                    break
        else:
            abnormals = [0, 0]
    except:
        return
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height, width = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))*2-50, int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))*2
    out = cv2.VideoWriter(f'ohloss_vis/{video_name.split("/")[1]}_1f_W0_2.mp4', fourcc, fps, (448, 398))
    # print(width, height)
    
    weight = 0.2
    ohloss_results = {}
    ohloss_results[crop] = []
    OHLoss = []
    frame_count = 0
    while video_cap.isOpened():
        ret, frame = video_cap.read()
        if ret:
            # if frame_count % 4 == 0 and frame_count != 0:
            data = load_frame_cv2(frame)
            oversampled_data = oversample_data_single_img(data)
            crop_frame = oversampled_data[crop][0][0]
            # crop_frame = data
            # keypoints = pose_model.get_pose(crop_frame)
            
            # 시각화
            keypoints, vis_keypoints = pose_model.get_pose(crop_frame)
            

            if len(keypoints) > 1:
                
                # 시각화
                ohloss, colors, labels = cluster_keypoints_with_loss(keypoints)
                person_colors = [colors[label] for label in labels]
                person_colors = convert_to_bgr(person_colors)
                
                # ohloss = cluster_keypoints_with_loss(keypoints)
                ohloss = ohloss * weight
                # 소수점 3째 자리까지 반올림
                ohloss = round(ohloss, 3)
                
                # 시각화
                for k in range(len(vis_keypoints)):
                    person = vis_keypoints[k]
                    for kp in person:
                        cv2.circle(crop_frame, (int(kp[0]), int(kp[1])), 3, person_colors[k], -1)
                
            else:
                ohloss = 0 * weight
                
                # 시각화
                if len(vis_keypoints)==1:
                    person = vis_keypoints[0]
                    for kp in person:
                        cv2.circle(crop_frame, (int(kp[0]), int(kp[1])), 3, (0, 0, 255), -1)
                
            OHLoss.append(ohloss)
                
            if frame_count % 16 == 0 and frame_count != 0:
                fps_OHLoss = sum(OHLoss) / len(OHLoss)
                ohloss_results[crop].append(fps_OHLoss)
                OHLoss = []

            
            # 시각화
            if os.path.exists('plot_frame.png'):
                plot_image = cv2.imread('plot_frame.png')
                os.remove('plot_frame.png')
            else:
                plt.title('Clustered Keypoints')
                plt.xlabel('Dimension 1')
                plt.ylabel('Dimension 2')
                plt.xlim(-5, 5)
                plt.ylim(-5, 5)
                plt.savefig('plot_frame.png')
                plt.close()
                plot_image = cv2.imread('plot_frame.png')
                os.remove('plot_frame.png')
            # 필요하다면 plot_image의 크기를 조정
            plot_image_resized = cv2.resize(plot_image, (crop_frame.shape[1], crop_frame.shape[0]))

            # 프레임과 플롯 이미지를 수평으로 결합
            combined_frame = np.hstack((crop_frame, plot_image_resized))
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (50, 50)
            fontScale = 1
            colors = (255, 0, 0)
            thickness = 2
            cv2.putText(combined_frame, 'OHLoss: {:.2f}'.format(ohloss), org, font, fontScale, colors, thickness, cv2.LINE_AA)
            
            figsize = (crop_frame.shape[1]*2//50, (crop_frame.shape[0])//50)
            fig, ax = plt.subplots(figsize=figsize)
            ax.set_ylim(0, 1)
            ax.set_xlim(0, total_frame//16)
            ax.plot(ohloss_results[crop])
            ax.axvspan(abnormals[0]//16, abnormals[1]//16, color='red', alpha=0.2)
            ax.set_xlabel('Time')
            ax.set_ylabel('OHLoss')
            ax.set_title('OHLoss Over Time')
            fig.canvas.draw()
            
            plot_img_np = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            plot_img_np = plot_img_np.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            plot_img_resized = cv2.resize(plot_img_np,  (crop_frame.shape[1]*2, crop_frame.shape[0]-50))
            combined_frame = np.vstack((combined_frame, plot_img_resized))
            out.write(combined_frame)
            cv2.imshow(f'Frame - Crop {crop}', combined_frame)
                
            key = cv2.waitKey(25)
            if key == ord('q'):
                break
                
            frame_count += 1
        else:
            break
    video_cap.release()
    
    # 시각화
    out.release()
    cv2.destroyAllWindows()
    
    if crop == 0:
        crop_ = ""
    else:
        crop_ = "__5"
    
    np.save(f"{path}{name}{crop_}.npy", ohloss_results[crop])

if __name__ == '__main__':
    crop = 0
    print(f"{str(crop)} crop")
    split_file = []
    # s = open('list/ucf-train.list', 'r')
    # s = open('list/KD/pretrain/ucf-label-i3d_Train.list', 'r')
    # for line in s:
    #     line = line.strip()
    #     line = line.split('/')[-2] + '/' + line.split('/')[-1].split('_x')[0]
    #     split_file.append(line)
    # s25 = open('list/KD/label_25/ucf-label-i3d_Train.list', 'r')
    # for line in s25:
    #     line = line.strip()
    #     line = line.split('/')[-2] + '/' + line.split('/')[-1].split('_x')[0]
    #     split_file.append(line)
    # un25 = open('/home/sb-oh/WVED/UR-DMU/HPE/UR-DMU-KD/list/KD/unlabel_25/ucf-unlabel-i3d.list', 'r')
    # for line in un25:
    #     line = line.strip()
    #     line = line.split('/')[-2] + '/' + line.split('/')[-1].split('_x')[0]
    #     split_file.append(line)
    # un50 = open('/home/sb-oh/WVED/UR-DMU/HPE/UR-DMU-KD/list/KD/unlabel_50/ucf-unlabel-i3d.list', 'r')
    # for line in un50:
    #     line = line.strip()
    #     line = line.split('/')[-2] + '/' + line.split('/')[-1].split('_x')[0]
    #     split_file.append(line)
    # un75 = open('/home/sb-oh/WVED/UR-DMU/HPE/UR-DMU-KD/list/KD/unlabel_75/ucf-unlabel-i3d.list', 'r')
    # for line in un75:
    #     line = line.strip()
    #     line = line.split('/')[-2] + '/' + line.split('/')[-1].split('_x')[0]
    #     split_file.append(line)
    ts = open('list/KD/label_25/ucf-label-i3d_Test.list', 'r')
    for line in ts:
        line = line.strip()
        if "__" in line:
            continue
        line = line.split('/')[-2] + '/' + line.split('/')[-1].split('_x')[0]
        split_file.append(line)

    
    # split_file.reverse()
    vid_list = []
    
    for line in split_file:
        mp4_path = "/home/subin-oh/Nas-subin/SB-Oh/data/Anomaly-Detection-Dataset/Train"
        state = "Train"
        if "Testing" in line.split()[0]:
            state = "Test"
            mp4_path = "/home/subin-oh/Nas-subin/SB-Oh/data/Anomaly-Detection-Dataset/Test"
        video_path = os.path.join(mp4_path, line.split()[0]+"_x264.mp4")
        
        # 파일 이름에 video_name이 포함되어 있는지 확인
        video_name = video_path.split('/')[-2]+'/'+video_path.split('/')[-1].split(".")[0]
        path = f"/home/subin-oh/Nas-subin/SB-Oh/data/HPE/OHLoss_np4/{video_name.split('/')[0]}/"
        if os.path.exists(path) == False:
            os.makedirs(path)
        # name이 포함된 파일 리스트
        name = video_name.split('/')[1]
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
        
    # for q in range(len(vid_list)):
    #     video = vid_list[q]
    #     print(q,"/",len(vid_list),end="\r")
    #     process_video(video, crop)   
    process_video('/home/subin-oh/Nas-subin/SB-Oh/data/Anomaly-Detection-Dataset/Train/Fighting/Fighting033_x264.mp4', 0)