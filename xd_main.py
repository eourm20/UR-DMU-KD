import pdb
import numpy as np
import torch.utils.data as data
import utils
from options import *
from config import *
from train import *
from xd_test import test
from model import *
import clearml
from clearml import Task
import os
from dataset_loader import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        pdb.set_trace()

    config = Config(args)
    worker_init_fn = None
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    config.len_feature = 1024
    task_logger = None
    net = WSAD(config.len_feature,flag = "Train", a_nums = 60, n_nums = 60)
    net = net.cuda()

    normal_train_loader = data.DataLoader(
        XDVideo(root_dir = config.root_dir, mode = 'Train',modal = config.modal, num_segments = 200, len_feature = config.len_feature, is_normal = True),
            batch_size = 64,
            shuffle = True, num_workers = config.num_workers,
            worker_init_fn = worker_init_fn, drop_last = True)
    abnormal_train_loader = data.DataLoader(
        XDVideo(root_dir = config.root_dir, mode='Train', modal = config.modal, num_segments = 200, len_feature = config.len_feature, is_normal = False),
            batch_size = 64,
            shuffle = True, num_workers = config.num_workers,
            worker_init_fn = worker_init_fn, drop_last = True)
    test_loader = data.DataLoader(
        XDVideo(root_dir = config.root_dir, mode = 'Test', modal = config.modal, num_segments = config.num_segments, len_feature = config.len_feature),
            batch_size = 1,
            shuffle = False, num_workers = config.num_workers,
            worker_init_fn = worker_init_fn)

    test_info = {"step": [], "auc": [],"ap":[]}
    
    best_auc = 0
    best_auc_update = 0

    criterion = AD_Loss()
    
    optimizer = torch.optim.Adam(net.parameters(), lr = config.lr[0],
        betas = (0.9, 0.999), weight_decay = 0.00005)

    # wind = Visualizer(env = 'XD_URDMU', port = "2022", use_incoming_socket = False)
    test(net, config, test_loader, test_info, 0)
    for step in tqdm(
            range(1, config.num_iters + 1),
            total = config.num_iters,
            dynamic_ncols = True
        ):
        if step > 1 and config.lr[step - 1] != config.lr[step - 2]:
            for param_group in optimizer.param_groups:
                param_group["lr"] = config.lr[step - 1]
        if (step - 1) % len(normal_train_loader) == 0:
            normal_loader_iter = iter(normal_train_loader)

        if (step - 1) % len(abnormal_train_loader) == 0:
            abnormal_loader_iter = iter(abnormal_train_loader)
        train(net, normal_loader_iter,abnormal_loader_iter, optimizer, criterion, task_logger, step)
        # early stopping 20번동안 best auc가 갱신되지 않으면 종료
        test(net, config, test_loader, test_info, step)
        if test_info["auc"][-1] > best_auc:
            best_auc = test_info["auc"][-1]
            best_auc_update = 0
            utils.save_best_record(test_info, 
                os.path.join(config.output_path, "XD_ALL3_best_record.txt"))
            torch.save(net.state_dict(), os.path.join(args.model_path, \
                args.model_file.split('<')[0]+"_best.pkl"))
        else:
            best_auc_update += 1
            if best_auc_update == 100:
                break
