# coding=utf-8
import torch
from torch.optim import AdamW, SGD, Adam
import random
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import time
import collections
import torch.nn as nn
from sklearn.metrics import f1_score

class TrainLoop:
    def __init__(self, args, writer, model, optimizer, scheduler,
                 test_data, device, early_stop=5, test_fre=1):
        self.args = args
        self.writer = writer
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.test_data = test_data
        self.device = device
        self.lr_anneal_steps = args.lr_anneal_steps
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        # self.opt = AdamW([p for p in self.model.parameters() if p.requires_grad==True], lr=args.lr, weight_decay=args.weight_decay)
        self.log_interval = args.log_interval
        self.best_nmse_random = 1e9
        self.warmup_steps = 5
        self.min_lr = args.min_lr
        self.best_nmse = 1e9
        self.early_stop = early_stop
        self.test_fre = test_fre
        self.criterion_t1 = nn.MSELoss().to(device)
        self.criterion_t2 = nn.CrossEntropyLoss().to(device)
        self.criterion_t3 = nn.MSELoss().to(device)

        self.mask_list = {'random': [0.85], 'temporal': [0.5], 'fre': [0.5]}

    def Train_iter(self, train_data, mask_ratio, mask_strategy, seed=None, dataset='', mode='backward',
                   is_output_mat=False, task_id=1):
        loss_list = []
        # start time
        for _, batch in enumerate(train_data):
            # print(batch.shape)
            pred = self.model_forward(batch, mask_ratio, mask_strategy, seed=seed,
                                      data=dataset, mode='forward', task_id=task_id)
            # print(pred.shape, batch['label'].shape)
            if mode == 'backward':
                if task_id == 1:
                    loss = self.criterion_t1(pred, batch['h_full'])
                elif task_id == 2:
                    loss = self.criterion_t2(pred, batch['label'].squeeze())
                elif task_id == 3:
                    loss = self.criterion_t3(pred, batch['location'])

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                loss_list.append(loss.item())

            else:
                if task_id == 1:
                    # cal NMSE
                    diff = torch.abs(pred - batch['h_full']) ** 2
                    power = torch.abs(batch['h_full']) ** 2
                    nmse = torch.sum(diff, dim=tuple(range(1, diff.ndim))) / torch.sum(power,
                                                                                       dim=tuple(range(1, power.ndim)))
                    result = torch.mean(nmse).item()
                elif task_id == 2:
                    # cal F1
                    pred_classes = pred.argmax(dim=1)  # [B]
                    # correct = (pred_classes == batch['label'].squeeze()).float()  # [B], 0/1
                    # result = correct.mean().item()
                    f1_macro = f1_score(batch['label'].squeeze().detach().cpu(), pred_classes.detach().cpu(),
                                        average='macro')
                    result = 1 - f1_macro
                elif task_id == 3:
                    # cal RMSE
                    result = torch.mean(torch.linalg.norm(pred - batch['location'], dim=1)).item()

                loss_list.append(result)

        return np.mean(np.array(loss_list))

    def Train(self, train_data, epochs, seed=None, mask_ratio=0.5, mask_strategy='fre', task_id=1):
        nmse_list = []
        self.best_nmse = 1e9
        for epoch in range(epochs):
            self.model.train()
            loss_iter = self.Train_iter(train_data, mask_ratio=mask_ratio, mask_strategy=mask_strategy, seed=seed,
                                        dataset='None', mode='backward', task_id=task_id)
            print(f'Epoch [{epoch + 1}/{epochs}] Training Loss: {loss_iter}')
            nmse_list.append(loss_iter)
            if epoch % self.test_fre == 0:
                with torch.no_grad():
                    self.model.eval()
                    result = self.Train_iter(self.test_data, mask_ratio=mask_ratio, mask_strategy=mask_strategy,
                                             seed=seed,
                                             dataset='None', mode='test', task_id=task_id)
                    if result < self.best_nmse:
                        is_break = self.best_model_save(epoch, result, None)
                    print(f'Epoch [{epoch + 1}/{epochs}] Testing Loss: {result}', end=' || ')
                    print(' Test_Result_best:{}\n'.format(self.best_nmse))

    def best_model_save(self, step, nmse, nmse_key_result):

        self.early_stop = 0
        # torch.save(self.model.state_dict(), self.args.model_path+'model_save/model_best_stage_{}.pkl'.format(self.args.stage))
        torch.save(self.model.state_dict(), self.args.model_path + 'model_save/model_best.pkl')
        self.best_nmse = nmse
        self.writer.add_scalar('Evaluation/NMSE_best', self.best_nmse, step)
        # print('\nNMSE_best:{}\n'.format(self.best_nmse))
        # print(str(nmse_key_result) + '\n')
        # with open(self.args.model_path+'result.txt', 'w') as f:
        #     f.write('stage:{}, epoch:{}, best nmse: {}\n'.format(self.args.stage, step, self.best_nmse))
        #     f.write(str(nmse_key_result) + '\n')
        # with open(self.args.model_path+'result_all.txt', 'a') as f:
        #     f.write('stage:{}, epoch:{}, best nmse: {}\n'.format(self.args.stage, step, self.best_nmse))
        #     f.write(str(nmse_key_result) + '\n')
        return 'save'

    def mask_select(self, name):
        if self.args.mask_strategy_random == 'none':  # 'none' or 'batch'
            mask_strategy = self.args.mask_strategy
            mask_ratio = self.args.mask_ratio
        else:
            mask_strategy = random.choice(['random', 'temporal', 'fre'])
            mask_ratio = random.choice(self.mask_list[mask_strategy])

        return mask_strategy, mask_ratio

    def mask_list_chosen(self, name):
        if self.args.mask_strategy_random == 'none':  # 'none' or 'batch'
            mask_list = self.mask_list
        else:
            mask_list = {key: self.mask_list[key] for key in ['random', 'temporal', 'fre']}
        return mask_list

    def model_forward(self, batch, mask_ratio, mask_strategy, seed=None, data=None, mode='backward',
                      task_id=1):
        for value, key in batch.items():
            batch[value] = key.to(self.device)
        if task_id == 1:
            pred = self.model(
                batch['h_inter'],
                mask_ratio=0,
                mask_strategy='fre',
                seed=seed,
                data=data, task_id=task_id,
            )
        elif task_id == 2:
            pred = self.model(
                batch['h_full'],
                mask_ratio=0,
                mask_strategy='fre',
                seed=seed,
                data=data, task_id=task_id,
            )
        elif task_id == 3:
            pred = self.model(
                batch['h_full'], batch['imgs'],
                mask_ratio=0,
                mask_strategy='fre',
                seed=seed,
                data=data, task_id=task_id,
            )
        return pred
