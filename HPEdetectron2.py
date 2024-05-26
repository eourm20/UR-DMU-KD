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


from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


class detectron2_pose:
    def __init__(self):
        # Set up the configuration and load a pre-trained model from Detectron2's model zoo
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
        self.predictor = DefaultPredictor(self.cfg)

    def get_pose_vi(self, frame):
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert BGR image to RGB

        outputs = self.predictor(frame_rgb)
        
        if "instances" in outputs:
            instances = outputs["instances"].to(torch.device('cpu'))
            if instances.has("pred_keypoints"):
                keypoints_predictions = instances.pred_keypoints

                # Now you can use `keypoints_predictions` tensor as needed.
                if keypoints_predictions.shape[0] > 1:
                    v = Visualizer(frame_rgb[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
                    v_out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
                    # cv2.imshow('Visualized Frame', v_out.get_image()[:, :, ::-1])
                    
                    variances = keypoints_predictions.std(dim=1).sum(dim=1)
                    max_variance_idx = torch.argmax(variances)
                    selectes_person = keypoints_predictions[max_variance_idx].unsqueeze(0)
                    
                    # tsne = TSNE(n_components=3, random_state=0)
                    # transformed_data = tsne.fit_transform(keypoints_predictions)
                    # print(transformed_data)
                    # 사람이 여러명이면 하나의 의미로 합침
                    # Get the indices of the max confidence score along the 'people' dimension (0)
                    # conf_max_indices = keypoints_predictions[:,:,2].argmax(axis=0)
                    # Use these indices to select the corresponding rows for each keypoint
                    # keypoints_predictions = keypoints_predictions[conf_max_indices, torch.arange(keypoints_predictions.shape[1])]
                    # print("person more than 1")
                    for key in instances._fields.keys():
                        instances._fields[key] = instances._fields[key][int(max_variance_idx)]
                    instances._fields['scores'] = instances._fields['scores'].unsqueeze(0)
                    instances._fields['pred_classes'] = instances._fields['pred_classes'].unsqueeze(0)
                    instances._fields['pred_keypoints'] = instances._fields['pred_keypoints'].unsqueeze(0)
                    instances._fields['pred_keypoint_heatmaps'] = instances._fields['pred_keypoint_heatmaps'].unsqueeze(0)
                    v2 = Visualizer(frame_rgb[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
                    v_out2 = v2.draw_instance_predictions(instances)
                    
                    # cv2.imshow('Visualized Frame(one person)', v_out2.get_image()[:, :, ::-1])
                    
                    return keypoints_predictions, v_out.get_image()[:, :, ::-1], v_out2.get_image()[:, :, ::-1]
                else:
                    keypoints_predictions = keypoints_predictions[0]
                    v = Visualizer(frame_rgb[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
                    v_out = v.draw_instance_predictions(instances)
                    
                    return keypoints_predictions, v_out.get_image()[:, :, ::-1], v_out.get_image()[:, :, ::-1]
                
    def get_pose(self, frame):

        # BGR 이미지를 RGB 이미지로 변환 (Matplotlib은 RGB 이미지를 사용)

        # 이미지 출력
        # plt.imshow(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # Make prediction on the current frame and extract keypoints predictions 
        outputs = self.predictor(frame)
        
        if "instances" in outputs:
            instances = outputs["instances"].to(torch.device('cpu'))
            if instances.has("pred_keypoints"):
                keypoints_predictions = instances.pred_keypoints
                # print(keypoints_predictions)
                # [x,y,신뢰도]
                if len(keypoints_predictions)==0:
                    keypoints_predictions = np.zeros((17,3))
                    keypoints_predictions = keypoints_predictions.reshape(-1)
                    return keypoints_predictions
                else:
                    # Now you can use `keypoints_predictions` tensor as needed.
                    if keypoints_predictions.shape[0] > 1:
                        tsne = TSNE(n_components=3, random_state=0)
                        transformed_data = tsne.fit_transform(keypoints_predictions)
                        print(transformed_data)
                        # 사람이 여러명이면 하나의 의미로 합침
                        # Get the indices of the max confidence score along the 'people' dimension (0)
                        # conf_max_indices = keypoints_predictions[:,:,2].argmax(axis=0)
                        # Use these indices to select the corresponding rows for each keypoint
                        # keypoints_predictions = keypoints_predictions[conf_max_indices, torch.arange(keypoints_predictions.shape[1])]
                        # print("person more than 1")
                        return keypoints_predictions
                    else:
                        keypoints_predictions = keypoints_predictions[0]
                        keypoints_predictions = keypoints_predictions.reshape(-1)
                        # print("person 1")
                        return keypoints_predictions
                
            
    def normalize(self, frame):
        # Assuming `array` is your original array

        # Find the minimum and maximum of the array
        min_val = frame.min()
        max_val = frame.max()

        # Normalize to [0, 1]
        normalized_frame = (frame - min_val) / (max_val - min_val)

        # Scale to [0, 255] and convert to uint8
        scaled_frame = (normalized_frame * 255).astype('uint8')  
        return scaled_frame
    
    
if __name__ == '__main__':
    pose_model = detectron2_pose()
    video_cap = cv2.VideoCapture('/home/subin-oh/Nas-subin/SB-Oh/data/Anomaly-Detection-Dataset/Train/Shooting/Shooting008_x264.mp4')
    if not video_cap.isOpened():
        print("Error opening video file")

    # Read until video is completed
    while(video_cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = video_cap.read()
        
        if ret:
            
            keypoints, visualized_frame, one_person_detection = pose_model.get_pose_vi(frame)
        
            cv2.imshow('Visualized Frame', visualized_frame)
            cv2.imshow('Visualized Frame_one person', one_person_detection)
            # Press Q on keyboard to exit (optional)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        else: 
            break

    # After reading all frames, close the display window and release video capture object
    video_cap.release()
    cv2.destroyAllWindows()