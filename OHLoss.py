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
from scipy.spatial.distance import euclidean
from tqdm import tqdm
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import multiprocessing as mp
from multiprocessing import Pool, cpu_count


def dunn_index(keypoints_2d, labels, centers):
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
    if max_intracluster_distance == 0:
        return np.inf  # 또는 적절한 오류 메시지 반환 또는 다른 값을 반환
    dunn = min_intercluster_distance / max_intracluster_distance
    return dunn

def find_optimal_clusters(data):
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
    optimal_clusters = iters[np.argmax(dunn_scores)]
    return optimal_clusters
    
def cluster_keypoints_with_loss(keypoints):
    keypoints_flattened = keypoints.reshape(keypoints.shape[0], -1)
    
    pca = PCA(n_components=2)
    keypoints_2d = pca.fit_transform(keypoints_flattened)
    optimal_clusters = find_optimal_clusters(keypoints_2d)
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=0).fit(keypoints_2d)
    
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    
    if optimal_clusters == 1:
        return 0
    
    cluster_counts = np.bincount(labels)
    
    # 가장 적은 인원을 가진 클러스터들의 인덱스와 그 인원 수 확인
    min_count_indices = np.where(cluster_counts == np.min(cluster_counts))[0]
    
    # 초기화
    max_loss = -np.inf
    abnormal_cluster = None
    
    # 모든 클러스터에 대해 반복
    for idx in min_count_indices:
        current_loss = 0
        for i, center in enumerate(centers):
            if i == idx:
                continue
            loss = np.linalg.norm(center - centers[idx])
            current_loss += loss

        if current_loss > max_loss:
            max_loss = current_loss
            abnormal_cluster = idx

    final_loss = 0
    for i, center in enumerate(centers):
        if i == abnormal_cluster:
            continue
        final_loss += np.linalg.norm(center - centers[abnormal_cluster])

    return final_loss*(np.max(cluster_counts)) / len(centers)

class Detectron2Pose:
    def __init__(self):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml"))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml")
        self.predictor = DefaultPredictor(self.cfg)

    def get_pose(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        outputs = self.predictor(frame)
        vis_person = []
        filtered_person = []
        if "instances" in outputs:
            instances = outputs["instances"].to(torch.device('cpu'))
            if instances.has("pred_keypoints"):
                keypoints_predictions = instances.pred_keypoints
                # [x,y,신뢰도]
                if len(keypoints_predictions) == 0:
                    return filtered_person, vis_person
                else:
                    indices = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
                    for keypoints in keypoints_predictions:
                        # 신뢰도가 0.8 이상인 keypoint만 선택
                        keypoints = keypoints[indices]
                        keypoints_xy = keypoints[:, :2]  # 첫 번째와 두 번째 요소(x, y)만 선택
                        center_point_indies = [0, 1, 6, 7] 
                        center_point = keypoints_xy[center_point_indies].mean(dim=0)
                        relative_keypoints = keypoints_xy - center_point  # 전체 키포인트에서 중심점 좌표를 뺌
                        
                        # Normalization
                        # 각 keypoint 사이의 최대 거리를 계산
                        max_distance = torch.max(torch.cdist(relative_keypoints, relative_keypoints, p=2))
                        normalized_keypoints = relative_keypoints / max_distance  # 모든 위치값을 최대 거리로 나눔
                        
                        vis_person.append(keypoints_xy)
                        filtered_person.append(normalized_keypoints)
                    if len(filtered_person) <= 1:
                        return filtered_person, vis_person
                    else:
                        filtered_person = torch.stack(filtered_person, dim=0)
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
    data = cv2.resize(frame, (340, 256), interpolation=cv2.INTER_LINEAR)
    return data

def oversample_data_single_img(data):
    data = data.reshape((1, 1) + data.shape)
    # 데이터 오버샘플링 로직 적용
    data_flip = np.array(data[:, :, :, ::-1, :])

    data_1 = np.array(data[:, :, :224, :224, :])
    data_2 = np.array(data[:, :, :224, -224:, :])
    data_3 = np.array(data[:, :, 16:240, 58:282, :])
    data_4 = np.array(data[:, :, -224:, :224, :])
    data_5 = np.array(data[:, :, -224:, -224:, :])

    data_f_1 = np.array(data_flip[:, :, :224, :224, :])
    data_f_2 = np.array(data_flip[:, :, :224, -224:, :])
    data_f_3 = np.array(data_flip[:, :, 16:240, 58:282, :])
    data_f_4 = np.array(data_flip[:, :, -224:, :224, :])
    data_f_5 = np.array(data_flip[:, :, -224:, -224:, :])

    # return [data_1, data_2, data_3, data_4, data_5, data_f_1, data_f_2, data_f_3, data_f_4, data_f_5]
    return [data_3, data_f_3]


def process_video(video):
    pose_model = Detectron2Pose()
    video_name = video.split('/')[-2]+'/'+video.split('/')[-1].split(".")[0]
    video_cap = cv2.VideoCapture(video)
    if not video_cap.isOpened():
        print("Error opening video file")
        return

    total_frame = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    weight = 0.2

    # Initialize loss lists
    all_OHLoss = [[] for _ in range(2)]
    OHLoss = [[] for _ in range(2)]
    frame_count = 0

    while video_cap.isOpened():
        ret, frame = video_cap.read()
        if ret:
            data = load_frame_cv2(frame)
            oversampled_data = oversample_data_single_img(data)
            for i in range(len(oversampled_data)):
                crop_frame = oversampled_data[i][0][0]
                keypoints, vis_keypoints = pose_model.get_pose(crop_frame)
                if len(keypoints) > 1:
                    ohloss = cluster_keypoints_with_loss(keypoints)
                    ohloss = ohloss * weight
                else:
                    ohloss = 0 * weight
                # print('OHLoss:', ohloss)
                OHLoss[i].append(ohloss)

                if frame_count % 16 == 0 and frame_count != 0:
                    fps_OHLoss = sum(OHLoss[i]) / len(OHLoss[i])
                    all_OHLoss[i].append(fps_OHLoss)
                    OHLoss[i] = []

            frame_count += 1
            key = cv2.waitKey(25)
            if key == ord('q'):
                break
        else:
            break

    video_cap.release()
    cv2.destroyAllWindows()
    print(f"{video_name} processing complete")
    # Save numpy arrays
    for j in range(2):
        if os.path.exists(f"OHLoss_np/{video_name.split('/')[0]}") == False:
            os.makedirs(f"OHLoss_np/{video_name.split('/')[0]}")
        np.save(f"OHLoss_np/{video_name}_{j}.npy", all_OHLoss[j])
        print(f"save: OHLoss_np/{video_name}_{j}.npy")
if __name__ == '__main__':
    # 특정 GPU만 사용하도록 환경 변수 설정
    mp.set_start_method('spawn')
    s = open('list/ucf-train.list', 'r')
    split_file = []
    for line in s:
        line = line.strip()
        split_file.append(line)
    split_file.reverse()
    vid_list = []
    mp4_path = "/home/subin-oh/Nas-subin/SB-Oh/data/Anomaly-Detection-Dataset/Train"
    for line in split_file:
        if os.path.exists(f"OHLoss_np/{line}_x264_0.npy"):
            print(f"{line} already processed. Skipping...")
            continue
        if "Testing" in line.split()[0]:
            mp4_path = "/home/subin-oh/Nas-subin/SB-Oh/data/Anomaly-Detection-Dataset/Test"
        video_path = os.path.join(mp4_path, line.split()[0]+"_x264.mp4")
        vid_list.append(video_path)

    # Use multiprocessing to process videos in parallel
    with Pool(processes=5) as p:
        list(tqdm(p.imap(process_video, vid_list), total=len(vid_list)))