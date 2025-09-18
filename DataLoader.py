# coding=utf-8
import numpy as np
import torch as th
import os
import json
import torch
import scipy.io
import datetime
import copy
import hdf5storage
import random
import math
from torch.utils.data import Dataset
import torch.nn.functional as F
from einops import rearrange


def LoadBatch(H):
    # H: ...     [tensor complex]
    # out: ..., 2  [tensor real]
    size = list(H.shape)
    H_real = np.zeros(size + [2])
    H_real[..., 0] = H.real
    H_real[..., 1] = H.imag
    H_real = torch.tensor(H_real, dtype=torch.float32)
    if len(H_real.shape) == 4:
        H_real = rearrange(H_real, 'b n k o -> b o n k')
    else:
        H_real = rearrange(H_real, 'b t n k o -> b o t n k')
    return H_real


class Dataset_task_1_CE(Dataset):
    def __init__(self, file_path, itr=10000, SEED=42, is_show=1, dataset_name="train"):
        super(Dataset_task_1_CE, self).__init__()
        # Shuffle and Segmentation Dateset
        db = hdf5storage.loadmat(file_path + f'/X_{dataset_name}.mat')[f'X_{dataset_name}']
        db2 = hdf5storage.loadmat(file_path + f'/X_pilot_{dataset_name}.mat')[f'X_pilot_{dataset_name}']
        Batch_num = db.shape[0]
        np.random.seed(SEED)
        idx = np.arange(Batch_num)
        np.random.shuffle(idx)
        idx = idx[:itr]

        # Load data for Channel Prediction
        self.H_full = torch.tensor(db, dtype=torch.complex128)
        self.H_pilot = torch.tensor(db2, dtype=torch.complex128)

        self.H_full = LoadBatch(self.H_full)
        self.H_pilot = LoadBatch(self.H_pilot)

        self.H_pilot = self.H_pilot[idx, ...]
        self.H_full = self.H_full[idx, ...]

        full_size = self.H_full.shape[-3:]
        self.H_inter = F.interpolate(self.H_pilot, size=full_size, mode='trilinear', align_corners=False)

        if is_show:
            print('Training Dataset info: ')
            print(
                f'H_pilot(Input) shape: {self.H_pilot.shape}\t'
                f'H_full(Label) shape: {self.H_full.shape}\n'
                f'H_inter(Input) shape: {self.H_inter.shape}\t'
            )

    def __getitem__(self, index):
        return {
            "h_pilot": self.H_pilot[index, ...].float(),
            "h_full": self.H_full[index, ...].float(),
            "h_inter": self.H_inter[index, ...].float(),
        }

    def __len__(self):
        return self.H_pilot.shape[0]


def data_load_task_1_CE(args):
    train_data = Dataset_task_1_CE(args.file_load_path + '/Task1-CE', itr=900, dataset_name="train")
    test_data = Dataset_task_1_CE(args.file_load_path + '/Task1-CE', dataset_name="val")

    train_data = th.utils.data.DataLoader(train_data, num_workers=8, batch_size=args.batch_size, shuffle=False,
                                          pin_memory=False, prefetch_factor=4)
    test_data = th.utils.data.DataLoader(test_data, num_workers=8, batch_size=args.batch_size, shuffle=False,
                                         pin_memory=False, prefetch_factor=4)

    return train_data, test_data


class Dataset_task_2_LoS_NLoS(Dataset):
    def __init__(self, file_path, itr=10000, SEED=42, is_show=1, dataset_name="train"):
        super(Dataset_task_2_LoS_NLoS, self).__init__()
        # Shuffle and Segmentation Dateset
        db = hdf5storage.loadmat(file_path + f'/X_{dataset_name}.mat')[f'X_{dataset_name}']
        db2 = hdf5storage.loadmat(file_path + f'/L_{dataset_name}.mat')[f'L_{dataset_name}']
        Batch_num = db.shape[0]
        np.random.seed(SEED)
        idx = np.arange(Batch_num)
        np.random.shuffle(idx)
        idx = idx[:itr]

        # Load data for Channel Prediction
        self.H_full = torch.tensor(db, dtype=torch.complex128)
        self.Label = torch.tensor(db2, dtype=torch.long)

        self.H_full = LoadBatch(self.H_full)

        self.H_full = self.H_full[idx, ...]
        self.Label = self.Label[idx, ...]

        if is_show:
            print('Training Dataset info: ')
            print(
                f'H_full(Input) shape: {self.H_full.shape}\t'
                f'Label shape: {self.Label.shape}\n'
            )

    def __getitem__(self, index):
        return {
            "label": self.Label[index].long(),
            "h_full": self.H_full[index, ...].float(),
        }

    def __len__(self):
        return self.H_full.shape[0]


def data_load_task_2_LoS_NLoS(args):
    train_data = Dataset_task_2_LoS_NLoS(args.file_load_path + '/Task2-Los-NLoS', itr=300, dataset_name="train")
    test_data = Dataset_task_2_LoS_NLoS(args.file_load_path + '/Task2-Los-NLoS', dataset_name="val")

    train_data = th.utils.data.DataLoader(train_data, num_workers=8, batch_size=args.batch_size, shuffle=False,
                                          pin_memory=False, prefetch_factor=4)
    test_data = th.utils.data.DataLoader(test_data, num_workers=8, batch_size=args.batch_size, shuffle=False,
                                         pin_memory=False, prefetch_factor=4)

    return train_data, test_data


class Dataset_task_3_V_WL(Dataset):
    def __init__(self, file_path, itr=10000, SEED=42, is_show=1, dataset_name="train"):
        super(Dataset_task_3_V_WL, self).__init__()
        # Shuffle and Segmentation Dateset
        db = hdf5storage.loadmat(file_path + f'/X_{dataset_name}.mat')[f'X_{dataset_name}']
        db2 = hdf5storage.loadmat(file_path + f'/imgs_{dataset_name}.mat')[f'imgs_{dataset_name}']
        db3 = hdf5storage.loadmat(file_path + f'/location_{dataset_name}.mat')[f'location_{dataset_name}']
        Batch_num = db.shape[0]
        np.random.seed(SEED)
        idx = np.arange(Batch_num)
        np.random.shuffle(idx)
        idx = idx[:itr]

        # Load data for Channel Prediction
        self.H_full = torch.tensor(db, dtype=torch.complex128)
        self.img = torch.tensor(db2, dtype=torch.float32).permute(0, 3, 1, 2)
        self.location = torch.tensor(db3, dtype=torch.float32)

        self.H_full = LoadBatch(self.H_full)
        self.img = F.interpolate(self.img, size=(224, 224), mode='bilinear', align_corners=False) / 255.0

        self.H_full = self.H_full[idx, ...]
        self.img = self.img[idx, ...]
        self.location = self.location[idx, ...]

        if is_show:
            print('Training Dataset info: ')
            print(
                f'H_full(Input) shape: {self.H_full.shape}\t'
                f'imgs(Input) shape: {self.img.shape}\n'
                f'location(Output) shape: {self.location.shape}\t'
            )

    def __getitem__(self, index):
        return {
            "imgs": self.img[index, ...].float(),
            "h_full": self.H_full[index, ...].float(),
            "location": self.location[index, ...].float(),
        }

    def __len__(self):
        return self.H_full.shape[0]


def data_load_task_3_V_WL(args):
    train_data = Dataset_task_3_V_WL(args.file_load_path + '/Task3-V-WL', itr=600, dataset_name="train")
    test_data = Dataset_task_3_V_WL(args.file_load_path + '/Task3-V-WL', dataset_name="val")

    train_data = th.utils.data.DataLoader(train_data, num_workers=8, batch_size=args.batch_size, shuffle=False,
                                          pin_memory=False, prefetch_factor=4)
    test_data = th.utils.data.DataLoader(test_data, num_workers=8, batch_size=args.batch_size, shuffle=False,
                                         pin_memory=False, prefetch_factor=4)

    return train_data, test_data
