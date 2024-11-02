import pdb
import numpy as np
import torch.utils.data as data
import utils
from options import *
from config import *
from train import *
from ucf_test import test
from model import *
import clearml
from clearml import Task
import os
from dataset_loader import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


# ROC 커브와 PR 커브를 그리기 위한 코드
def plot_curve(fpr, tpr, roc_auc, recall, precision, pr_auc, epoch):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    lw = 2
    ax1.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    ax1.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Receiver Operating Characteristic')
    ax1.legend(loc="lower right")
    
    ax2.plot(recall, precision, color='darkorange',
             lw=lw, label='PR curve (area = %0.2f)' % pr_auc)
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision Recall Curve')
    ax2.legend(loc="lower right")
    fig.suptitle('ROC and PR curve')
    save_path = args.log_path+'curve_plot/'
    fig.savefig(save_path+'epoch_{}.png'.format(epoch))
    plt.close()
    
    
if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        pdb.set_trace()

    config = Config(args)
    worker_init_fn = None
    # gpus = [0]
    # torch.cuda.set_device('cuda:{}'.format(gpus[0]))
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    if config.seed >= 0:
        utils.set_seed(config.seed)
        worker_init_fn = np.random.seed(config.seed)

    config.len_feature = 1024

    Task.set_credentials(
        api_host="https://api.clear.ml",
        web_host="https://app.clear.ml",
        files_host="https://files.clear.ml",
        key='60B49RW4U8P2S7DS15DW',
        secret='ctQIyHsC0rxTyh8RR8I3aGFOD9ylMveWurwVcPkhGBoMMwHsX8'
    )
    task = clearml.Task.init(project_name="UR-DMU-HPE", task_name="KD75_res", task_type=Task.TaskTypes.training)
    task_logger = task.get_logger()
    
    student_net = Student_WSAD(config.len_feature, flag = "Train", a_nums = 60, n_nums = 60)
    student_net = student_net.cuda()
    
    teacher_net = Teacher_WSAD(config.len_feature, flag = "Train", a_nums = 60, n_nums = 60)
    teacher_net.load_state_dict(torch.load(f'{args.model_path}Teacher2000__best.pkl', map_location = 'cuda'))
    teacher_net = teacher_net.cuda()
    
    normal_train_loader = data.DataLoader(
        UCF_crime(root_dir = config.root_dir, mode = 'Train', modal = config.modal, num_segments = 200, len_feature = config.len_feature, is_normal = True),
            batch_size = 64,
            shuffle = True, num_workers = config.num_workers,
            worker_init_fn = worker_init_fn, drop_last = True)
    abnormal_train_loader = data.DataLoader(
        UCF_crime(root_dir = config.root_dir, mode = 'Train', modal = config.modal, num_segments = 200, len_feature = config.len_feature, is_normal = False),
            batch_size = 64,
            shuffle = True, num_workers = config.num_workers,
            worker_init_fn = worker_init_fn, drop_last = True)
    unlabeled_loader_loader = data.DataLoader(
        Unlabeled_UCF_crime(root_dir = config.root_dir, modal = config.modal, num_segments = 200, len_feature = config.len_feature),
            batch_size = 64,
            shuffle = True, num_workers = config.num_workers,
            worker_init_fn = worker_init_fn, drop_last = True)
    test_loader = data.DataLoader(
        UCF_crime(root_dir = config.root_dir, mode = 'Test', modal = config.modal, num_segments = config.num_segments, len_feature = config.len_feature),
            batch_size = 1,
            shuffle = False, num_workers = config.num_workers,
            worker_init_fn = worker_init_fn)

    test_info = {"step": [], "auc": [],"ap":[],"ac":[]}
    
    best_auc = 0

    criterion = AD_Loss()
    
    optimizer = torch.optim.Adam(student_net.parameters(), lr = config.lr[0],
        betas = (0.9, 0.999), weight_decay = 0.00005)

    # wind = Visualizer(env = 'UCF_URDMU', port = "2022", use_incoming_socket = False)
    test(student_net, config, test_loader, test_info, 0)
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
            # print("normal_loader_iter")

        if (step - 1) % len(abnormal_train_loader) == 0:
            abnormal_loader_iter = iter(abnormal_train_loader)
            # print('abnormal_loader_iter')
            
        if (step - 1) % len(unlabeled_loader_loader) == 0:
            unlabeled_loader_iter = iter(unlabeled_loader_loader)
            # print('unlabeled_loader_iter')
            
        train(student_net, teacher_net, normal_loader_iter,abnormal_loader_iter, unlabeled_loader_iter, optimizer, criterion, task_logger, step, config.num_iters)
        if step % 10 == 0 and step >= 10:
            test(student_net, config, test_loader, test_info, step)
            task_logger.report_scalar(title = "AUC",series = "AUC",value = test_info["auc"][-1], iteration = step//10)
            task_logger.report_scalar(title = "AP",series = "AP",value = test_info["ap"][-1], iteration = step//10)
            task_logger.report_scalar(title = "ACC",series = "ACC",value = test_info["ac"][-1], iteration = step//10)
            if test_info["auc"][-1] > best_auc:
                best_auc = test_info["auc"][-1]
                utils.save_best_record(test_info, 
                    os.path.join(config.output_path, "ucf_KD75_res_best_record.txt"))

                torch.save(student_net.state_dict(), os.path.join(args.model_path, \
                    args.model_file.split('<')[0]+"_best.pkl"))
            if step == config.num_iters:
                utils.save_best_record(test_info, 
                    os.path.join(config.output_path, "ucf_KD75_res_last_record_{}.txt".format(step)))

                torch.save(student_net.state_dict(), os.path.join(args.model_path, \
                    args.model_file.split('<')[0]+"{}_last.pkl".format(step)))
