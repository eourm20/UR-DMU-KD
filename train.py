import torch
import torch.nn as nn

def norm(data):
    l2=torch.norm(data, p = 2, dim = -1, keepdim = True)
    return torch.div(data, l2)

class AD_Loss(nn.Module):
    def __init__(self, HPLoss_w) -> None:
        super().__init__()
        self.bce = nn.BCELoss()
        self.HPLoss_w = HPLoss_w
        
        
    def forward(self, result, _label):
        loss = {}

        _label = _label.float()

        triplet = result["triplet_margin"]
        att = result['frame']
        A_att = result["A_att"]
        N_att = result["N_att"]
        A_Natt = result["A_Natt"]
        N_Aatt = result["N_Aatt"]
        kl_loss = result["kl_loss"]
        distance = result["distance"]
        # HPLoss 추가
        oh_att = result["oh_att"]
        tf_att = result["tf_att"]
        
        b = _label.size(0)//2
        t = att.size(1)      
        anomaly = torch.topk(att, t//16 + 1, dim=-1)[0].mean(-1)
        anomaly_loss = self.bce(anomaly, _label)
        # print('anomaly_loss:', anomaly_loss)
        # HP Loss 추가
        # 시그모이드 함수를 통과시켜서 0~1 사이의 값으로 변환
        oh_att = nn.Sigmoid()(oh_att)
        oh = torch.topk(oh_att, t//16 + 1, dim = -1)[0].mean(-1)
        oh_loss = self.bce(oh, _label)
        # print('oh_loss:', oh_loss)
        tf_att = nn.Sigmoid()(tf_att)
        tf = torch.topk(tf_att, t//16 + 1, dim = -1)[0].mean(-1)
        tf_loss = self.bce(tf, _label)
        # print('tf_loss:', tf_loss)
        hp_loss = torch.max(oh_loss, tf_loss)
        # print('hp_loss:', hp_loss)

        panomaly = torch.topk(1 - N_Aatt, t//16 + 1, dim=-1)[0].mean(-1)
        panomaly_loss = self.bce(panomaly, torch.ones((b)).cuda())
        
        A_att = torch.topk(A_att, t//16 + 1, dim = -1)[0].mean(-1)
        A_loss = self.bce(A_att, torch.ones((b)).cuda())

        N_loss = self.bce(N_att, torch.ones_like((N_att)).cuda())    
        A_Nloss = self.bce(A_Natt, torch.zeros_like((A_Natt)).cuda())
        
        

        # 기존의 AD_Loss
        cost = anomaly_loss + 0.1 * (A_loss + panomaly_loss + N_loss + A_Nloss) + 0.1 * triplet + 0.001 * kl_loss + 0.0001 * distance

        # HPLoss 추가
        new_cost = cost + self.HPLoss_w * hp_loss
        
        loss['total_loss'] = cost
        loss['att_loss'] = anomaly_loss
        loss['N_Aatt'] = panomaly_loss
        loss['A_loss'] = A_loss
        loss['N_loss'] = N_loss
        loss['A_Nloss'] = A_Nloss
        loss["triplet"] = triplet
        loss['kl_loss'] = kl_loss
        # HPLoss 추가
        loss['new_total_loss'] = new_cost
        loss['oh_loss'] = oh_loss
        loss['tf_loss'] = tf_loss
        loss['hp_loss'] = hp_loss
        
        return new_cost, loss



def train(net, normal_loader, abnormal_loader, optimizer, criterion, task_logger, index):
    net.train()
    net.flag = "Train"
    ninput, nlabel, nohloss, ntfloss = next(normal_loader)
    ainput, alabel, aohloss, atfloss = next(abnormal_loader)
    _data = torch.cat((ninput, ainput), 0)
    _label = torch.cat((nlabel, alabel), 0)
    _ohloss = torch.cat((nohloss, aohloss), 0)
    _tfloss = torch.cat((ntfloss, atfloss), 0)
    _data = _data.cuda()
    _label = _label.cuda()
    _ohloss = _ohloss.cuda()
    _tfloss = _tfloss.cuda()
    predict = net(_data, _ohloss, _tfloss)
    cost, loss = criterion(predict, _label)
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    if task_logger is not None:
        task_logger.report_scalar(title = 'Supervised Loss', series = 'total_loss', value = loss['total_loss'].item(), iteration = index)
        task_logger.report_scalar(title = 'Supervised Loss', series = 'new_total_loss', value = loss['new_total_loss'].item(), iteration = index)
        task_logger.report_scalar(title = 'Supervised Loss', series = 'OHLoss', value = loss['oh_loss'].item(), iteration = index)
        task_logger.report_scalar(title = 'Supervised Loss', series = 'TFLoss', value = loss['tf_loss'].item(), iteration = index)
        task_logger.report_scalar(title = 'Supervised Loss', series = 'HPLoss', value = loss['hp_loss'].item(), iteration = index)
