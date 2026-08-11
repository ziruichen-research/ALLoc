import os
import random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from base_trainer import BaseTrainer
from dataset import Dataset

class Trainer(BaseTrainer):
    def __init__(self, exp_conf, writer):
        super().__init__(exp_conf)
        self.writer = writer
        
        data_dirs = self.exp_conf['data_dirs']
        init_lr = self.exp_conf['init_lr']
        batch_size = self.exp_conf['batch_size']
        finetune = self.exp_conf['finetune']
        if self.scheme == 'ad_cnn_multitask' and finetune:
            self.model.conv_model.requires_grad_(False)
            self.optimizer = optim.Adam(self.model.fc_models.parameters(), lr=init_lr)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=init_lr)
        
        self.train_datasets = [Dataset(exp_conf, data_dir, 'train') for data_dir in data_dirs]
        self.test_datasets = [Dataset(exp_conf, data_dir, 'test') for data_dir in data_dirs]
        if self.scheme in ['alloc', 'iclloc']:
            gen_seq_datasets = self.exp_conf['gen_seq_datasets']
            for train_dataset, test_dataset in zip(self.train_datasets, self.test_datasets):
                if gen_seq_datasets:
                    train_dataset.generate_sequence_dataset()
                    Dataset.generate_sequence_dataset_test(train_dataset, test_dataset)
                else:
                    test_dataset.train_dataset_obj = train_dataset
        self.train_loaders = [DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True) for dataset in self.train_datasets]
        self.test_loaders = [DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True) for dataset in self.test_datasets]
        
        self.epoch = 0
        self.global_step = 0
        self.loss_cache = []
        
        os.makedirs(os.path.join(self.exp_dir, 'ckpts'), exist_ok=True)
        ckpts = os.listdir(os.path.join(self.exp_dir, 'ckpts'))
        epochs = [int(name[5:-4]) for name in ckpts]
        if len(epochs) > 0:
            epochs.sort()
            print(f'Find checkpoint: epoch {epochs[-1]}')
            self.load_checkpoint(epochs[-1])
        else:
            if finetune:
                exp_dir = self.exp_conf['finetune_model']['exp_dir']
                epoch = self.exp_conf['finetune_model']['epoch']
                self.load_model(self.model, exp_dir, epoch, load_backbone=(self.scheme == 'ad_cnn_multitask'))
            self.save_checkpoint()

    def load_checkpoint(self, epoch):
        pth_path = os.path.join(self.exp_dir, 'ckpts', f'ckpt_{epoch:06d}.pth')
        assert os.path.exists(pth_path)
        ckpt = torch.load(pth_path, map_location=torch.device('cuda'))
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.epoch = ckpt['epoch']
        self.global_step = ckpt['global_step']

    def save_checkpoint(self):
        ckpt = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'global_step': self.global_step
        }
        pth_path = os.path.join(self.exp_dir, 'ckpts', f'ckpt_{self.epoch:06d}.pth')
        torch.save(ckpt, pth_path)
    
    def update_lr(self):
        init_lr = self.exp_conf['init_lr']
        lr_decay_steps = self.exp_conf['lr_decay_steps']
        lr_decay_factor = self.exp_conf['lr_decay_factor']
        lr = init_lr
        for step in lr_decay_steps:
            if self.global_step >= step:
                lr = lr * lr_decay_factor
        for g in self.optimizer.param_groups:
            g['lr'] = lr

    def log(self, loss):
        log_step_inter = self.exp_conf['log_step_inter']
        self.loss_cache.append(loss.item())
        if self.global_step % log_step_inter == 0:
            mean_loss = sum(self.loss_cache) / len(self.loss_cache)
            self.writer.add_scalar('Train/loss', mean_loss, global_step=self.global_step)
            self.loss_cache.clear()
            self.writer.add_scalar('Train/lr', self.optimizer.param_groups[0]['lr'], global_step=self.global_step)
            self.writer.flush()
            
    def train_step(self, k, data, loc, init_loc):
        self.global_step += 1
        data = data.cuda()
        loc = loc.cuda()
        init_loc = init_loc.cuda()
        self.optimizer.zero_grad()
        loss = self.forward(k, data, loc, init_loc)
        loss.backward()
        self.optimizer.step()
        if self.writer is not None:
            self.log(loss)
        self.update_lr()

    def train(self):
        num_epochs = self.exp_conf['num_epochs']
        num_max_steps = self.exp_conf['num_max_steps']
        batch_size = self.exp_conf['batch_size']
        mix_train_datasets = self.exp_conf['mix_train_datasets']
        num_train_data = self.exp_conf['num_train_data']
        save_epoch_inter = self.exp_conf['save_epoch_inter']
        test_epoch_inter = self.exp_conf['test_epoch_inter']

        num_scenarios = len(self.train_datasets)
        scenario_indices = list(range(num_scenarios))
        start_epoch = self.epoch
        for _ in range(num_epochs - start_epoch):
            self.epoch += 1
            self.model.train()
            total_steps = num_scenarios * ((num_train_data + batch_size - 1) // batch_size)
            pbar = tqdm(desc=f'[Train] epoch {self.epoch}/{num_epochs}, global step {self.global_step}/{num_max_steps}', total=total_steps)
            if mix_train_datasets:
                for items in zip(*self.train_loaders):
                    random.shuffle(scenario_indices)
                    for k in scenario_indices:
                        data, loc, init_loc = items[k]
                        self.train_step(k, data, loc, init_loc)
                        pbar.set_description(f'[Train] epoch {self.epoch}/{num_epochs}, global step {self.global_step}/{num_max_steps}')
                        pbar.update()
                        if self.global_step >= num_max_steps:
                            break
                    if self.global_step >= num_max_steps:
                        break
            else:
                random.shuffle(scenario_indices)
                for k in scenario_indices:
                    loader = self.train_loaders[k]
                    for data, loc, init_loc in loader:
                        self.train_step(k, data, loc, init_loc)
                        pbar.set_description(f'[Train] epoch {self.epoch}/{num_epochs}, global step {self.global_step}/{num_max_steps}')
                        pbar.update()
                        if self.global_step >= num_max_steps:
                            break
                    if self.global_step >= num_max_steps:
                        break
            pbar.close()
            if self.epoch % save_epoch_inter == 0:
                self.save_checkpoint()
            if self.epoch % test_epoch_inter == 0:
                self.model.eval()
                metric = []
                for k, loader in enumerate(self.test_loaders):
                    for data, loc, init_loc in tqdm(loader, desc=f'[Test] epoch {self.epoch}/{num_epochs}, scenario {k + 1}/{num_scenarios}'):
                        data = data.cuda()
                        loc = loc.cuda()
                        init_loc = init_loc.cuda()
                        with torch.no_grad():
                            rmse = self.forward(k, data, loc, init_loc)
                        metric.append(rmse.cpu().numpy())
                metric = np.concatenate(metric, axis=0)
                if self.writer is not None:
                    self.writer.add_scalar(f'Test/rmse', np.mean(metric), global_step=self.epoch)
                    self.writer.flush()
            if self.global_step >= num_max_steps:
                break
        pth_path = os.path.join(self.exp_dir, 'model.pth')
        torch.save(self.model.state_dict(), pth_path)

class Tester(BaseTrainer):
    def __init__(self, exp_conf):
        super().__init__(exp_conf)
        batch_size = self.exp_conf['batch_size']
        data_dirs = self.exp_conf['data_dirs']
        if self.scheme in ['alloc', 'iclloc']:
            self.train_datasets = [Dataset(exp_conf, data_dir, 'train') for data_dir in data_dirs]
        
        # self.test_datasets = [Dataset(exp_conf, data_dir, 'test') for data_dir in data_dirs]
        self.test_datasets = []
        for k, data_dir in enumerate(data_dirs):
            if self.iterative:
                init_loc_dir = self.exp_conf['init_loc_dir']
                npz_path = os.path.join(init_loc_dir, f'loc_{k}.npz')
                init_loc = np.load(npz_path)['loc_hat']
                self.test_datasets.append(Dataset(exp_conf, data_dir, 'test', init_loc))
            else:
                self.test_datasets.append(Dataset(exp_conf, data_dir, 'test'))
        self.test_loaders = [DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True) for dataset in self.test_datasets]
        
        exp_dir = self.exp_conf['model']['exp_dir']
        epoch = self.exp_conf['model']['epoch']
        self.load_model(self.model, exp_dir, epoch)
    
    def test(self):
        flops, macs, params = self.compute_flops()
        print(f'FLOPs: {flops}   MACs: {macs}   Params: {params}')
        
        num_scenarios = len(self.test_loaders)
        self.model.eval()
        for k, loader in enumerate(self.test_loaders):
            metric = []
            loc_list = []
            init_loc_list = []
            loc_hat_list = []
            for data, loc, init_loc in tqdm(loader, desc=f'[Test] scenario {k + 1}/{num_scenarios}'):
                data = data.cuda()
                loc = loc.cuda()
                init_loc = init_loc.cuda()
                with torch.no_grad():
                    rmse, init_loc, loc_hat = self.forward(k, data, loc, init_loc)
                metric.append(rmse.cpu().numpy())
                loc_list.append(loc.cpu().numpy())
                init_loc_list.append(init_loc.cpu().numpy())
                loc_hat_list.append(loc_hat.cpu().numpy())
            metric = np.concatenate(metric, axis=0)
            loc = np.concatenate(loc_list, axis=0)
            init_loc = np.concatenate(init_loc_list, axis=0)
            loc_hat = np.concatenate(loc_hat_list, axis=0)
            print(f'rmse: {np.mean(metric)}')
            npy_path = os.path.join(self.exp_dir, f'rmse_{k}.npy')
            np.save(npy_path, metric)
            npz_path = os.path.join(self.exp_dir, f'loc_{k}.npz')
            np.savez(npz_path, loc=loc, init_loc=init_loc, loc_hat=loc_hat)