import torch
import torch.nn as nn

from .mpp_loss import MPPLoss
from .normal_loss import NormalLoss

class LossComputer(nn.Module):
    def __init__(self, w_normal=1., w_mpp=1., HPLoss_w=1.):
        super().__init__()
        self.w_normal = w_normal
        self.w_mpp = w_mpp
        self.mppLoss = MPPLoss()
        self.normalLoss = NormalLoss()
        self.HPLoss_w = HPLoss_w

    def forward(self, result):
        loss = {}

        pre_normal_scores = result['pre_normal_scores']
        normal_loss = self.normalLoss(pre_normal_scores)
        loss['normal_loss'] = normal_loss

        anchors          = result['bn_results']['anchors']
        variances        = result['bn_results']['variances']
        select_normals   = result['bn_results']['select_normals']
        select_abnormals = result['bn_results']['select_abnormals']
        t = pre_normal_scores.size(1)
        anormaly = torch.topk(pre_normal_scores, t//16 + 1, dim = -1)[0].max(-1)[0]
        # HPLoss 추가
        oh_att = result["oh_att"]
        tf_att = result["tf_att"]
        
        oh = torch.topk(oh_att, t//16 + 1, dim = -1)[0].max(-1)[0]
        # min-max scaling
        # 최댓값이 1 이상이면 min-max scaling을 통해 0~1 사이의 값으로 변환
        if oh.max() > 1:
            oh = (oh - oh.min()) / (oh.max() - oh.min())

        # oh_loss = self.bce(oh, _label)
        tf = torch.topk(tf_att*2.5, t//16 + 1, dim = -1)[0].max(-1)[0]
        # tf = torch.topk(tf_att, t//16 + 1, dim = -1)[0].max(-1)[0]
        if tf.max() > 1:
            tf = (tf - tf.min()) / (tf.max() - tf.min())

        # tf_loss = self.bce(tf, _label)
        # 평균값  
        hp = torch.max(oh,tf)
        # hp_loss = self.bce(hp, predict_label)
        hp_loss = nn.MSELoss()(hp, anormaly)
        
        
        mpp_loss = self.mppLoss(anchors, variances, select_normals, select_abnormals)
        loss['mpp_loss'] = mpp_loss

        loss['total_loss'] = self.w_normal * normal_loss + self.w_mpp * mpp_loss

        # HPLoss 추가
        # loss['new_cost'] = 0.5*loss['total_loss'] + self.HPLoss_w * hp_loss#for UCF
        loss['new_cost'] = 0.9*loss['total_loss'] + self.HPLoss_w * hp_loss#for XD
        
        return loss['new_cost'], loss