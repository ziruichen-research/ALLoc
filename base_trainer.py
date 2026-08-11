import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from calflops import calculate_flops
from model import *

class BaseTrainer:
    def __init__(self, exp_conf):
        self.exp_conf = exp_conf
        self.exp_dir = self.exp_conf['exp_dir']
        self.task = self.exp_conf['task']
        self.scheme = self.exp_conf['scheme']
        self.use_seq_dataset_test = (self.task == 'train')
        self.iterative = (self.task == 'test' and self.scheme in ['alloc', 'iclloc'] and self.exp_conf['init_loc_dir'] is not None)
        self.init_model()

    def init_model(self):
        assert self.scheme in ['res_mlp', 'ad_cnn', 'ad_cnn_multitask', 'cnn', 'mfc_net', 'alloc', 'iclloc']
        model_name = {
            'res_mlp': 'ResidualMLP',
            'ad_cnn': 'CNN',
            'ad_cnn_multitask': 'MultitaskCNN',
            'cnn': 'CNN',
            'mfc_net': 'MFCNet',
            'alloc': 'ALLoc',
            'iclloc': 'ICLLoc'
        }[self.scheme]
        
        model = eval(model_name + '(self.exp_conf)')
        self.model = model.cuda()

        if self.scheme in ['ad_cnn', 'ad_cnn_multitask']:
            num_ant = self.exp_conf['num_ant']
            num_car = self.exp_conf['num_car']
            i = torch.arange(num_car).unsqueeze(dim=1) # [num_car, 1]
            j = torch.arange(num_car).unsqueeze(dim=0) # [1, num_car]
            f_delay = 1 / np.sqrt(num_car) * torch.exp(-1j * 2 * np.pi / num_car * i * j)
            i = torch.arange(num_ant).unsqueeze(dim=1) # [num_ant, 1]
            j = torch.arange(num_ant).unsqueeze(dim=0) # [1, num_ant]
            f_angle = 1 / np.sqrt(num_ant) * torch.exp(-1j * 2 * np.pi / num_ant * i * (j - num_ant / 2))
            self.f_delay = f_delay.cuda()
            self.f_angle = f_angle.cuda()

    def load_model(self, model, exp_dir, epoch, load_backbone=False):
        if epoch is None:
            pth_path = os.path.join(exp_dir, 'model.pth')
            assert os.path.exists(pth_path)
            state_dict = torch.load(pth_path, map_location=torch.device('cuda'))
        else:
            pth_path = os.path.join(exp_dir, 'ckpts', f'ckpt_{epoch:06d}.pth')
            assert os.path.exists(pth_path)
            ckpt = torch.load(pth_path, map_location=torch.device('cuda'))
            state_dict = ckpt['model']
        if load_backbone:
            backbone_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('conv_model.'):
                    backbone_state_dict[key[11:]] = value
            model.conv_model.load_state_dict(backbone_state_dict)
        else:
            model.load_state_dict(state_dict)

    def forward(self, k, data, loc, init_loc):
        '''
        data: [batch_size, num_ant, num_car] or [batch_size, max_len_nei + 1, num_ant, num_car], complex
        loc, init_loc: [batch_size, 2] or [batch_size, max_len_nei + 1, 2], float
        
        Return
        loss: [], float
        rmse: [batch_size], float
        '''
        inv_std = self.exp_conf['inv_std']

        if self.task == 'test':
            noise_dev = self.exp_conf['noise_dev']
            if noise_dev > 0:
                noise = torch.randn_like(data, dtype=torch.float32) * noise_dev
                data = data * (1 + noise)
            if self.scheme in ['alloc', 'iclloc'] and not self.iterative:
                loc_error_scale = self.exp_conf['loc_error_scale']
                if loc_error_scale > 0:
                    error = (torch.rand_like(init_loc) * 2 - 1) * loc_error_scale
                    init_loc = init_loc + error

        if self.scheme == 'res_mlp':
            data = torch.view_as_real(data) * inv_std # [batch_size, num_ant, num_car, 2]
            loc_hat = self.model(data) # [batch_size, 2]
            if self.model.training:
                loss = F.mse_loss(loc_hat, loc)
            else:
                rmse = torch.linalg.norm(loc - loc_hat, dim=-1) # [batch_size]

        elif self.scheme == 'ad_cnn':
            data = torch.matmul(torch.matmul(self.f_angle, data), self.f_delay.conj().t())
            data = torch.view_as_real(data) * inv_std # [batch_size, num_ant, num_car, 2]
            loc_hat = self.model(data) # [batch_size, 2]
            if self.model.training:
                loss = F.mse_loss(loc_hat, loc)
            else:
                rmse = torch.linalg.norm(loc - loc_hat, dim=-1) # [batch_size]

        elif self.scheme == 'ad_cnn_multitask':
            data = torch.matmul(torch.matmul(self.f_angle, data), self.f_delay.conj().t())
            data = torch.view_as_real(data) * inv_std # [batch_size, num_ant, num_car, 2]
            loc_hat = self.model(data, k) # [batch_size, 2]
            if self.model.training:
                loss = F.mse_loss(loc_hat, loc)
            else:
                rmse = torch.linalg.norm(loc - loc_hat, dim=-1) # [batch_size]
        
        elif self.scheme == 'cnn':
            data = torch.view_as_real(data) * inv_std # [batch_size, num_ant, num_car, 2]
            loc_hat = self.model(data) # [batch_size, 2]
            if self.model.training:
                loss = F.mse_loss(loc_hat, loc)
            else:
                rmse = torch.linalg.norm(loc - loc_hat, dim=-1) # [batch_size]
        
        elif self.scheme == 'mfc_net':
            data = torch.view_as_real(data) * inv_std # [batch_size, num_ant, num_car, 2]
            loc_hat = self.model(data) # [batch_size, num_car, 2]
            if self.model.training:
                num_car = self.exp_conf['num_car']
                loc = loc.unsqueeze(dim=1) # [batch_size, 1, 2]
                weight = torch.arange(1, num_car + 1).cuda() # [num_car]
                weight = weight / torch.sum(weight) * num_car # [num_car]
                weight = weight.reshape(1, num_car, 1) # [1, num_car, 1]
                loss = F.mse_loss(loc_hat * weight, loc * weight)
            else:
                loc_hat = loc_hat[:, -1] # [batch_size, 2]
                rmse = torch.linalg.norm(loc - loc_hat, dim=-1) # [batch_size]
        
        elif self.scheme in ['alloc', 'iclloc']:
            if self.model.training:
                # data, loc: [batch_size, max_len_nei + 1, num_ant, num_car], [batch_size, max_len_nei + 1, 2]
                if self.scheme == 'alloc':
                    data = torch.view_as_real(data) * inv_std # [batch_size, max_len_nei + 1, num_ant, num_car, 2]
                    len_nei_total = self.exp_conf['len_nei_total']
                    len_nei_least = self.exp_conf['len_nei_least']
                    len_nei = random.choice(list(range(len_nei_least, len_nei_total + 1)))
                    len_cur = random.choice(list(range(1, len_nei_total + 1)))
                    indices_nei = random.sample(list(range(len_nei_total + 1)), len_nei)
                    indices_cur = random.sample(list(range(len_nei_total + 1)), len_cur)
                    data_nei = data[:, indices_nei] # [batch_size, len_nei, num_ant, num_car, 2]
                    data_cur = data[:, indices_cur] # [batch_size, len_cur, num_ant, num_car, 2]
                    loc_nei = loc[:, indices_nei] # [batch_size, len_nei, 2]
                    loc_cur = loc[:, indices_cur] # [batch_size, len_cur, 2]
                    loc_hat = self.model(data_nei, loc_nei, data_cur) # [batch_size, len_cur, 2]
                    loss = F.mse_loss(loc_hat, loc_cur)
                elif self.scheme == 'iclloc':
                    iclloc_scheme = self.exp_conf['iclloc_scheme']
                    assert iclloc_scheme in ['generative', 'discriminative']
                    # data, loc: [batch_size, max_len_nei + 1, num_ant, num_car], [batch_size, max_len_nei + 1, 2]
                    data = torch.view_as_real(data) * inv_std # [batch_size, max_len_nei + 1, num_ant, num_car, 2]
                    len_nei_total = self.exp_conf['len_nei_total']
                    len_nei_least = self.exp_conf['len_nei_least']
                    if iclloc_scheme == 'discriminative':
                        len_nei = random.choice(list(range(len_nei_least, len_nei_total + 1)))
                        indices = list(range(len_nei_total + 1))
                        random.shuffle(indices)
                        indices = indices[:len_nei + 1]
                        data_nei = data[:, indices[:-1]] # [batch_size, len_nei, num_ant, num_car]
                        data_cur = data[:, indices[-1:]] # [batch_size, 1, num_ant, num_car]
                        loc_nei = loc[:, indices[:-1]] # [batch_size, len_nei, 2]
                        loc_cur = loc[:, indices[-1:]] # [batch_size, 1, 2]
                        loc_hat = self.model(data_nei, loc_nei, data_cur) # [batch_size, len_nei * 2 + 1, 2]
                        loc_hat = loc_hat[:, -1:] # [batch_size, 1, 2]
                        loss = F.mse_loss(loc_hat, loc_cur)
                    elif iclloc_scheme == 'generative':
                        indices = list(range(len_nei_total + 1))
                        random.shuffle(indices)
                        data_nei = data[:, indices[:-1]] # [batch_size, len_nei_total, num_ant, num_car]
                        data_cur = data[:, indices[-1:]] # [batch_size, 1, num_ant, num_car]
                        loc_nei = loc[:, indices[:-1]] # [batch_size, len_nei_total, 2]
                        loc_cur = loc[:, indices[-1:]] # [batch_size, 1, 2]
                        loc_total = loc[:, indices] # [batch_size, len_nei_total + 1, 2]
                        loc_hat = self.model(data_nei, loc_nei, data_cur) # [batch_size, len_nei_total * 2 + 1, 2]
                        loss = F.mse_loss(loc_hat[:, len_nei_least * 2::2], loc_total[:, len_nei_least:])
                    
            else:
                len_nei = self.exp_conf['len_nei_test']
                if self.use_seq_dataset_test:
                    # data, loc: [batch_size, max_len_nei + 1, num_ant, num_car], [batch_size, max_len_nei + 1, 2]
                    data = torch.view_as_real(data) * inv_std # [batch_size, max_len_nei + 1, num_ant, num_car, 2]
                    data_nei = data[:, :len_nei] # [batch_size, len_nei, num_ant, num_car, 2]
                    data_cur = data[:, -1:] # [batch_size, 1, num_ant, num_car, 2]
                    loc_nei = loc[:, :len_nei] # [batch_size, len_nei, 2]
                    loc_hat = self.model(data_nei, loc_nei, data_cur) # [batch_size, 1, 2]
                    loc_hat = loc_hat[:, -1] # [batch_size, 2]
                    loc_cur = loc[:, -1] # [batch_size, 2]
                    rmse = torch.linalg.norm(loc_cur - loc_hat, dim=-1) # [batch_size]
                else:
                    # data, loc: [batch_size, num_ant, num_car], [batch_size, 2]
                    loc_hat_list = []
                    batch_size = data.shape[0]
                    for i in range(batch_size):
                        select = self.exp_conf['select']
                        data_seq, loc_seq = self.train_datasets[k].find_sequence(data[i], init_loc[i], select) # [max_len_nei + 1, num_ant, num_car], [max_len_nei + 1, 2]
                        data_seq = torch.view_as_real(data_seq) * inv_std # [max_len_nei + 1, num_ant, num_car, 2]
                        data_seq = data_seq.unsqueeze(dim=0) # [1, max_len_nei + 1, num_ant, num_car, 2]
                        loc_seq = loc_seq.unsqueeze(dim=0) # [1, max_len_nei + 1, 2]
                        data_nei = data_seq[:, :len_nei] # [1, len_nei, num_ant, num_car, 2]
                        data_cur = data_seq[:, -1:] # [1, 1, num_ant, num_car, 2]
                        loc_nei = loc_seq[:, :len_nei] # [1, len_nei, 2]
                        loc_hat = self.model(data_nei, loc_nei, data_cur) # [1, 1, 2]
                        loc_hat = loc_hat[:, -1] # [1, 2]
                        loc_hat_list.append(loc_hat)
                    loc_hat = torch.cat(loc_hat_list, dim=0) # [batch_size, 2]
                    rmse = torch.linalg.norm(loc - loc_hat, dim=-1) # [batch_size]

        if self.model.training:
            return loss
        else:
            if self.task == 'test':
                return rmse, init_loc, loc_hat
            else:
                return rmse
    
    def compute_flops(self):
        assert self.task == 'test'
        num_ant = self.exp_conf['num_ant']
        num_car = self.exp_conf['num_car']
        
        if self.scheme in ['res_mlp', 'ad_cnn', 'cnn', 'mfc_net']:
            inputs = [
                torch.randn((1, num_ant, num_car, 2)),
            ]

        elif self.scheme == 'ad_cnn_multitask':
            inputs = [
                torch.randn((1, num_ant, num_car, 2)),
                torch.zeros((), dtype=torch.int64)
            ]
        
        elif self.scheme in ['alloc', 'iclloc']:
            len_nei = self.exp_conf['len_nei_test']
            inputs = [
                torch.randn((1, len_nei, num_ant, num_car, 2)),
                torch.randn((1, len_nei, 2)),
                torch.randn((1, 1, num_ant, num_car, 2))
            ]
        
        flops, macs, params = calculate_flops(model=self.model, args=inputs, print_results=False)
        return flops, macs, params