import os
import argparse
import yaml
import torch
from trainer import Trainer, Tester

def get_exp_conf():
    _conf_path = f'_{os.getpid()}.yaml'
    os.system(f'cp {args.conf} {_conf_path}')
    if args.mod is None:
        args.mod = []
    mod_dict = {}
    for item_str in args.mod:
        item = item_str.split('=')
        mod_dict[item[0]] = '='.join(item[1:])
    with open(_conf_path, 'r') as file:
        lines = file.readlines()
    names = []
    with open(_conf_path, 'w') as file:
        for line in lines:
            if ':' in line:
                spaced_name = line.split(':')[0]
                name = spaced_name.lstrip()
                count = (len(spaced_name) - len(name)) // 2
                names = names[:count] + [name]
                key = '.'.join(names)
                if key in mod_dict:
                    line = f'{spaced_name}: {mod_dict[key]}\n'
            file.write(line)
    del lines
    with open(_conf_path) as file:
        exp_conf = yaml.safe_load(file)
    exp_dir = exp_conf['exp_dir']
    task = exp_conf['task']
    conf_path = os.path.join(exp_dir, f'{task}.yaml')
    if os.path.exists(conf_path):
        print(f'Load existing configurations: {conf_path}')
        os.system(f'rm {_conf_path}')
        with open(conf_path) as file:
            exp_conf = yaml.safe_load(file)
    else:
        os.makedirs(exp_dir, exist_ok=True)
        os.system(f'mv {_conf_path} {conf_path}')
    return exp_conf

if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage='%(prog)s [options]')
    parser.add_argument('--conf', type=str, default='conf.yaml')
    parser.add_argument('--cuda', type=str, default='0')
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--mod', nargs='*', type=str)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    os.environ['OMP_NUM_THREADS'] = str(args.cpu)
    os.environ['MKL_NUM_THREADS'] = str(args.cpu)
    torch.set_num_threads(args.cpu)
    
    exp_conf = get_exp_conf()
    task = exp_conf['task']
    assert task in ['train', 'test']
    
    if task == 'train':
        exp_dir = exp_conf['exp_dir']
        if exp_conf['tensorboard_writer']:
            log_dir = os.path.join(exp_dir, 'log')
            os.makedirs(log_dir, exist_ok=True)
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir)
        else:
            writer = None
        trainer = Trainer(exp_conf, writer)
        trainer.train()
    
    if task == 'test':
        tester = Tester(exp_conf)
        tester.test()
        