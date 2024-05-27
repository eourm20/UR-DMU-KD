import sys, os, distutils.core
sys.path.insert(0, os.path.abspath('./detectron2'))
import cv2
import torch
import numpy as np
from matplotlib import pyplot as plt
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from sklearn.manifold import TSNE

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

import warnings
warnings.filterwarnings("ignore")

def find_optimal_clusters(data):
    """
    데이터에 대해 최적의 클러스터 수를 찾는 함수
    data: 클러스터링할 데이터
    max_clusters: 최대 클러스터 수
    """
    max_clusters = data.shape[0]-1
    if max_clusters <= 2:
        return data.shape[0]
    iters = range(2, max_clusters + 1)
    s_scores = []
    
    for k in iters:
        kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
        s_scores.append(silhouette_score(data, kmeans.labels_))
    
    # 실루엣 스코어가 최대가 되는 클러스터 수 선택
    optimal_clusters = iters[np.argmax(s_scores)]
    return optimal_clusters

def cluster_keypoints_with_loss(keypoints):
    """
    키포인트 데이터를 클러스터링하고, 로스 함수를 계산하는 함수
    keypoints: 사람들의 키포인트 데이터를 담고 있는 배열, shape=(n_people, 17, 3)
               각 키포인트는 (x, y, confidence) 형태의 데이터를 가짐
    max_clusters: 최대 클러스터 수
    
    반환값: 각 사람이 속한 클러스터의 인덱스 배열, 로스 값
    """
    # 키포인트 데이터를 (n_people, 39) 형태의 배열로 변환
    keypoints_flattened = keypoints.reshape(keypoints.shape[0], -1)
    
    # 최적의 클러스터 수 찾기
    optimal_clusters = find_optimal_clusters(keypoints_flattened)
    
    if optimal_clusters == 1:
        return 0
    # K-means 클러스터링 수행
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=0).fit(keypoints_flattened)
    
    # 각 사람이 속한 클러스터의 인덱스 및 클러스터 중심점 계산
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    
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
    return final_loss / (len(centers) - 1)



class Detectron2Pose:
    def __init__(self):
        # Set up the configuration and load a pre-trained model from Detectron2's model zoo
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
        self.predictor = DefaultPredictor(self.cfg)

    # 각각의 사람들의 행동을 추측할 수 있는 주요 키포인트를 사용하여 각각의 사람들의 임베딩값 추출
    def get_pose(self, frame):
        # Convert BGR image to RGB (Matplotlib uses RGB images)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # Make prediction on the current frame and extract keypoints predictions 
        outputs = self.predictor(frame)
        
        if "instances" in outputs:
            instances = outputs["instances"].to(torch.device('cpu'))
            if instances.has("pred_keypoints"):
                keypoints_predictions = instances.pred_keypoints
                # print(keypoints_predictions)
                # [x,y,신뢰도]
                if len(keypoints_predictions) == 0:
                    # print('No person detected')
                    return False
                else:
                    # 필터링할 keypoint 인덱스, COCO index는 0부터 시작하므로 1을 빼줍니다.
                    indices = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
                    filtered_people = []
                    for keypoints in keypoints_predictions:
                        # 신뢰도가 0.8 이상인 keypoint만 선택
                        keypoints = keypoints[indices]
                        # 모든 키포인트의 신뢰도가 0.8 이상인지 확인
                        if all(kp[2] >= 0.02 for kp in keypoints):
                            filtered_people.append(keypoints)
                    if len(filtered_people) == 0:
                        # print('No person detected with high confidence keypoints')
                        # filtered_people = np.zeros((13, 3))
                        return False
                    else:
                        filtered_people = torch.stack(filtered_people, dim=0)
                        return filtered_people
                    
                
    def normalize(self, frame):
        # Find the minimum and maximum of the array
        min_val = frame.min()
        max_val = frame.max()

        # Normalize to [0, 1]
        normalized_frame = (frame - min_val) / (max_val - min_val)

        # Scale to [0, 255] and convert to uint8
        scaled_frame = (normalized_frame * 255).astype('uint8')  
        return scaled_frame

if __name__ == '__main__':
    pose_model = Detectron2Pose()
    video_cap = cv2.VideoCapture('/home/subin-oh/Nas-subin/SB-Oh/data/Anomaly-Detection-Dataset/Train/Fighting/Fighting033_x264.mp4')
    if not video_cap.isOpened():
        print("Error opening video file")

    # fps = video_cap.get(cv2.CAP_PROP_FPS)
    print("total frame:", video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    np_frame = np.load("/home/subin-oh/Nas-subin/SB-OH/data/I3D_feature/Test/RGB/Fighting/Fighting033_x264.npy")
    print("np_frame shape:", np_frame.shape)
    fps = 16
    weight = 30
    all_OHLoss = []
    # frame_interval = int(fps / 16)  # Calculate the interval to sample at 16 FPS

    # all_keypoints = []

    frame_count = 0
    while video_cap.isOpened():
        ret, frame = video_cap.read()
        
        if ret:
            OHLoss = []
            # if frame_count % frame_interval == 0:
            keypoints = pose_model.get_pose(frame)
            if keypoints is not False:
                OHLoss.append(cluster_keypoints_with_loss(keypoints))
                for person in keypoints:
                    for kp in person:
                        cv2.circle(frame, (int(kp[0]), int(kp[1])), 3, (255, 0, 0), -1)
                # print("OHLoss:", OHLoss[-1])
            else:
                OHLoss.append(0)
                
            if frame_count == 570:
                print("ABNORMAL FRAME START")
            if frame_count == 840:
                print("ABNORMAL FRAME END")
            if frame_count % 16 == 0:
                fps_OHLoss = sum(OHLoss)/(fps*weight)
                print('fps OHLoss: ',fps_OHLoss)
                all_OHLoss.append(fps_OHLoss)
                OHLoss = []
                
            cv2.imshow('frame', frame)

            frame_count += 1

            # Press Q on keyboard to exit (optional)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    video_cap.release()
    cv2.destroyAllWindows()

# np저장
np.save("OHLOSS_Fighting003_x264.npy", all_OHLoss)
