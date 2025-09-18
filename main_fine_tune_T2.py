# coding=utf-8
import argparse
import random
import os
from model_fine_tune import WiFo_model, WiFo
from train import TrainLoop

import setproctitle
import torch

from DataLoader import data_load_task_1_CE, data_load_task_2_LoS_NLoS
from utils import *

import torch as th
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from scheduler import FakeLR

def setup_init(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.backends.cudnn.benchmark = False
    th.backends.cudnn.deterministic = True

def dev(device_id='0'):
    """
    Get the device to use for torch.distributed.
    # """
    if th.cuda.is_available():
        return th.device('cuda:{}'.format(device_id))
    return th.device("cpu")

def create_argparser():
    defaults = dict(
        # experimental settings
        note = 'SoM2025-baseline',
        task = 'Task2',
        file_load_path = 'please/input/dataset/path',
        dataset = 'SoM2025',
        used_data = '',
        process_name = 'process_name_fine_tune',
        his_len = 6,
        pred_len = 6,
        few_ratio = 0.0,
        stage = 0,

        # model settings
        mask_ratio = 0.5,
        patch_size = 4,
        t_patch_size = 4,
        size = 'base',
        no_qkv_bias = 0,
        pos_emb = 'SinCos_3D',
        conv_num = 3,

        # pretrain settings
        random=True,
        mask_strategy = 'fre',
        mask_strategy_random = 'none', # ['none','batch']
        
        # training parameters
        lr=1e-5,
        min_lr = 1e-6,
        epochs = 50,
        early_stop = 5,
        weight_decay=0,
        batch_size=64,
        log_interval=5,
        total_epoches = 10000,
        device_id='0',
        machine = 'machine_name',
        clip_grad = 0.05,  # 0.05
        lr_anneal_steps = 200,
        rgb_aided=True,
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser
    
torch.multiprocessing.set_sharing_strategy('file_system')

def main():

    th.autograd.set_detect_anomaly(True)
    
    args = create_argparser().parse_args()
    setproctitle.setproctitle("{}-{}".format(args.process_name, args.device_id))
    setup_init(100)  # 随机种子设定100

    train_data, test_data = data_load_task_2_LoS_NLoS(args)  # 加载数据

    args.folder = 'Dataset_{}_Task_{}_FewRatio_{}_{}_{}/'.format(args.dataset, args.task, args.few_ratio, args.size, args.note)

    if args.mask_strategy_random != 'batch':
        args.folder = '{}_{}'.format(args.mask_strategy, args.mask_ratio) + args.folder
    args.model_path = './experiments/{}'.format(args.folder)
    logdir = "./logs/{}".format(args.folder)
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
        os.makedirs(args.model_path+'model_save/')

    writer = SummaryWriter(log_dir = logdir, flush_secs=5)
    device = dev(args.device_id)

    model = WiFo(t_patch_size=4, patch_size=4, embed_dim=512, decoder_embed_dim=512,
                 depth=6, decoder_depth=4, num_heads=8, decoder_num_heads=8,
                 pos_emb='SinCos_3D', args=args).to(device)
    model.load_state_dict(torch.load('./wifo_base.pkl', map_location=device), strict=False)
    model = freeze_wifo_except(model)

    # 优化器 + lr调度器
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    scheduler = FakeLR(optimizer=optimizer)


    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total number of parameters: {total_params/1e6} M')
    total_learn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of learnable parameter: %.5fM" % (total_learn / 1e6))
    # if args.file_load_path != '':
    #     model.load_state_dict(torch.load('{}.pkl'.format(args.file_load_path),map_location=device), strict=False)
    #     print('pretrained model loaded'+args.file_load_path)

    # 循环微调
    TrainLoop(
        args=args,
        writer=writer,
        model=model,
        optimizer=optimizer, scheduler=scheduler,
        test_data=test_data,
        device=device,
        early_stop=args.early_stop,
    ).Train(train_data, args.epochs, seed=100, mask_ratio=0, mask_strategy='fre', task_id=2)


def freeze_wifo_except(model):
    # 先冻结所有参数
    for param in model.parameters():
        param.requires_grad = False
    # 1. Fine-Tune-T1
    # for param in model.decoder_blocks[-2:].parameters():
    #     param.requires_grad = True
    for param in model.Fine_Tune_Layer_T1_CE.parameters():
        param.requires_grad = True

    # 2. Fine-Tune-T2
    # for param in model.blocks[-2:].parameters():
    #     param.requires_grad = True
    for param in model.Fine_Tune_Layer_T2_LoS_NLoS.parameters():
        param.requires_grad = True
    for param in model.Fine_Tune_Layer0_T2_LoS_NLoS.parameters():
        param.requires_grad = True

    return model

if __name__ == "__main__":
    main()