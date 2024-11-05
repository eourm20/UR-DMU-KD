import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def norm(data):
    l2=torch.norm(data, p = 2, dim = -1, keepdim = True)
    return torch.div(data, l2)

class AD_Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bce = nn.BCELoss()
      
        
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
        b = _label.size(0)//2
        t = att.size(1)      
        anomaly = torch.topk(att, t//16 + 1, dim=-1)[0].mean(-1)
        anomaly_loss = self.bce(anomaly, _label)

        panomaly = torch.topk(1 - N_Aatt, t//16 + 1, dim=-1)[0].mean(-1)
        panomaly_loss = self.bce(panomaly, torch.ones((b)).cuda())
        
        A_att = torch.topk(A_att, t//16 + 1, dim = -1)[0].mean(-1)
        A_loss = self.bce(A_att, torch.ones((b)).cuda())

        N_loss = self.bce(N_att, torch.ones_like((N_att)).cuda())    
        A_Nloss = self.bce(A_Natt, torch.zeros_like((A_Natt)).cuda())

        cost = anomaly_loss + 0.1 * (A_loss + panomaly_loss + N_loss + A_Nloss) + 0.1 * triplet + 0.001 * kl_loss + 0.0001 * distance

        loss['total_loss'] = cost
        loss['att_loss'] = anomaly_loss
        loss['N_Aatt'] = panomaly_loss
        loss['A_loss'] = A_loss
        loss['N_loss'] = N_loss
        loss['A_Nloss'] = A_Nloss
        loss["triplet"] = triplet
        loss['kl_loss'] = kl_loss
        return cost, loss


def update_ema_variables(teacher_model, student_model, initial_alpha, final_alpha, global_step, decay_rate):
    # teacher_model: EMA가 적용될 티쳐 모델
    # student_model: 현재 학습 중인 스튜던트 모델
    # alpha: EMA decay rate, 값이 클수록 과거 가중치가 더 크게 반영됨
    # global_step: 현재 학습 스텝
    alpha = initial_alpha
    # alpha = initial_alpha * np.exp(decay_rate * global_step)
    for teacher_param, student_param in zip(teacher_model.parameters(), student_model.parameters()):
        # 수정된 방식으로 add_ 사용
        teacher_param.data.mul_(alpha).add_(student_param.data, alpha=1 - alpha)
        
# # 이상 탐지 손실 함수(distillation loss)
# def consistency_loss(student_output, teacher_output, weight=2.0):
#     """
#     Consistency loss를 계산하는 함수. student 모델의 출력과 EMA를 적용한 teacher 모델의 출력 사이의 차이를 줄이는 목적.
#     """
#     return weight * F.mse_loss(student_output['frame'], teacher_output['frame'])

# 이상 탐지 손실 함수(distillation loss)
def consistency_loss(student_output, teacher_output, weight=0.3):
    """
    Consistency loss를 계산하는 함수. student 모델의 출력과 EMA를 적용한 teacher 모델의 출력 사이의 차이를 줄이는 목적.
    """
    psudo_label = torch.where(teacher_output['frame'] > 0.5, torch.ones_like(teacher_output['frame']), torch.zeros_like(teacher_output['frame']))
    return weight * nn.BCELoss()(student_output['frame'], psudo_label)
    # return weight * F.mse_loss(student_output['frame'], teacher_output['frame'])

# 어둡게 처리하는 함수
def darken_data(data, factor=0.5):
    # factor는 0과 1 사이의 값이며, 1에 가까울수록 원본에 가까워집니다.
    return data * factor


def add_noise(data, mean=0.0, std=0.1):
    """
    데이터에 Gaussian 노이즈를 추가하는 함수.
    mean은 노이즈의 평균값, std는 표준편차를 나타냅니다.
    """
    noise = torch.randn(data.size()).to(data.device) * std + mean
    return data + noise


def train(student_net, teacher_net, normal_loader, abnormal_loader, unlabel_loader, student_optimizer, criterion, task_logger, index, num_iters, KD_w):
    initial_alpha = 0.999
    final_alpha = 0.8

    decay_rate = np.log(final_alpha / initial_alpha) / num_iters
    
    ninput, nlabel = next(normal_loader)
    ainput, alabel = next(abnormal_loader)
    ulinput, aug_ulinput = next(unlabel_loader)
    
    _data = torch.cat((ninput, ainput), 0)
    _label = torch.cat((nlabel, alabel), 0)
    _data = _data.cuda()
    _label = _label.cuda()
    
    # ----------------- strong augmentation -----------------
    # --------------------------------------------------------
    # 데이터를 GPU로 이동
    # 화질 저하
    # pool = nn.AdaptiveAvgPool1d(512)
    # ulinput_lowres = pool(ulinput)
    # _unlabeled_data = ulinput_lowres.cuda()
    # _aug_unlabeled_data = aug_ulinput.cuda()
    
    # # 어둡게 처리
    ulinput_dark = darken_data(ulinput, factor=0.5)
    _unlabeled_data = ulinput_dark.cuda()
    _aug_unlabeled_data = aug_ulinput.cuda()
    
    # 노이즈 추가
    # ulinput_noisy = add_noise(ulinput)
    # _unlabeled_data = ulinput_noisy.cuda()
    # _aug_unlabeled_data = aug_ulinput.cuda()
    

    # ----------------- supervised loss -----------------
    student_net.train()
    student_net.flag = "Label_Train"
    predict = student_net(_data)
    supervised_loss, loss = criterion(predict, _label)
    student_optimizer.zero_grad()
    # cost.backward()
    # student_optimizer.step()
    
    update_ema_variables(teacher_net, student_net, initial_alpha, final_alpha, index+1, decay_rate)
    
    # ----------------- unsupervised loss -----------------
    with torch.no_grad():  # 교사 모델의 forward pass는 기울기 계산이 필요 없음
        teacher_net.eval()
        teacher_net.flag = "Unlabel_Train"
        teacher_output = teacher_net(_aug_unlabeled_data)  
    
    # autograd를 활성화하기 위해 student 모델의 forward pass는 torch.no_grad() 바깥에서 수행
    student_net.eval()  # student_net을 eval 모드로 설정
    student_net.flag = "Unlabel_Train"
    student_output = student_net(_unlabeled_data)

    # KD loss 계산
    unsupervised_loss = consistency_loss(student_output, teacher_output)
    # unsupervised_losses.append(unsupervised_loss.item())
    
    # student_optimizer.zero_grad()
    # unsupervised_loss.backward()
    # student_optimizer.step()
    
    # total_loss
    total_loss = supervised_loss*0.5 + KD_w * unsupervised_loss
    total_loss.backward()
    student_optimizer.step()
    
    # # EMA 업데이트
    # update_ema_variables(teacher_net, student_net, initial_alpha, final_alpha, index+2, decay_rate)
            
    if task_logger is not None:
        task_logger.report_scalar(title = 'Supervised Loss', series = 'total_loss', value = supervised_loss.item(), iteration = index)
        task_logger.report_scalar(title='Unsupervised Loss', series='total_loss', value=unsupervised_loss.item(), iteration=index)
        task_logger.report_scalar(title = 'Train Loss', series = 'total_loss', value = total_loss.item(), iteration = index)