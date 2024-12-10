import torch

def train(net, normal_loader, abnormal_loader, optimizer, criterion):
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
    res = net(_data, _ohloss, _tfloss)
    cost, loss = criterion(res)
    optimizer.zero_grad()
    cost.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
    optimizer.step()
    
    return loss