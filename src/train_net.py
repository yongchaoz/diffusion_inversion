import os
import re
import math
import wandb
import random
import time
from absl import logging

import sys
import ml_collections
import tensorflow as tf
import PIL
from PIL import Image

from torch.utils.data import Dataset

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import InterpolationMode

import medmnist
from medmnist import INFO

import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dataset import get_dataset  # NOQA
from models import ResNet18, ResNet50, VGG, MobileNetV2, EfficientNetB0, ShuffleNetV2  # NOQA

tf.config.experimental.set_visible_devices([], "GPU")


PIL_INTERPOLATION = {
    "linear": PIL.Image.Resampling.BILINEAR,
    "bilinear": PIL.Image.Resampling.BILINEAR,
    "bicubic": PIL.Image.Resampling.BICUBIC,
    "lanczos": PIL.Image.Resampling.LANCZOS,
    "nearest": PIL.Image.Resampling.NEAREST,
}

data_stats = {
    'cifar10': ([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
    'cifar100': ([0.5071, 0.4866, 0.4409], [0.2673, 0.2564, 0.2762]),
    'imagenette': ([0.4608, 0.4570, 0.4233], [0.2838, 0.2789, 0.3012]),
    'stl10': ([0.4467, 0.4398, 0.4066], [0.2603, 0.2566, 0.2713]),
    'imagenet': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    'pathmnist': ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    'bloodmnist': ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    'dermamnist': ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
}


def D(**kwargs):
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


class RealDataset(Dataset):
    def __init__(self, x, y, transform):
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return self.x.shape[0]

    def __repr__(self) -> str:
        return "Real Dataset\n" + f"Size: ({self.x.shape[0]})"

    def __getitem__(self, idx):
        img = self.x[idx].astype(np.uint8)
        image = Image.fromarray(img)
        return self.transform(image), self.y[idx]


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--dataset-name', metavar='DIR',
                    help='name of the dataset')
parser.add_argument('--wandb-name', metavar='NAME',
                    help='name of the wandb')
parser.add_argument('--data-dir', metavar='DIR',
                    help='path to dataset (root dir)')
parser.add_argument('--syn-data-dir', default='', type=str, metavar='DIR',
                    help='path to synthetic dataset (root dir)')
parser.add_argument('--syn-pattern', default='', type=str, metavar='NAME',
                    help='regular expression for the wanted synthetic dataset')
parser.add_argument('--output', default='', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')
parser.add_argument('--model', default='resnet18', type=str, metavar='NAME',
                    help='architecture of neural networks')
parser.add_argument('--optimizer', default='sgd', type=str, metavar='NAME',
                    help='architecture of neural networks')
parser.add_argument('--group-size', default=100, type=int, help='group size')
parser.add_argument('--batch-size', default=128, type=int, help='batch size')
parser.add_argument('--real-bs', default=64, type=int, help='batch size')
parser.add_argument('--syn-bs', default=64, type=int, help='batch size')
parser.add_argument('--subset', default=1, type=int, help='data subset')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--weight-decay', default=5e-4,
                    type=float, help='learning rate')
parser.add_argument('--num-steps', default=50000,
                    type=int, help='num training steps')
parser.add_argument('--num-data', default=1000,
                    type=int, help='num of training data')
parser.add_argument('--warmup-steps', default=1000,
                    type=int, help='num training steps')
parser.add_argument('--num-evals', default=20,
                    type=int, help='Compute eval metrics every n steps')
parser.add_argument('--seed', default=0,
                    type=int, help='random seed')
parser.add_argument('--log-wandb', action='store_true', default=False,
                    help='log training and validation metrics to wandb')


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_network(model, num_classes=10, resolution=32):
    if model == 'resnet18':
        net = ResNet18(num_classes=num_classes, resolution=resolution)
    elif model == 'resnet50':
        net = ResNet50(num_classes=num_classes, resolution=resolution)
    elif model == 'vgg16':
        net = VGG('VGG16', num_classes=num_classes)
    elif model == 'mobilenetv2':
        net = MobileNetV2(num_classes=num_classes)
    elif model == 'efficientnetb0':
        net = EfficientNetB0(num_classes=num_classes)
    elif model == 'shufflenetv2':
        net = ShuffleNetV2(
            net_size=0.5, num_classes=num_classes)
    else:
        raise NotImplementedError
    return net


def test(net, testloader, criterion, device):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return test_loss/total, 100.*correct/total


def main():
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if device == 'cuda':
        cudnn.benchmark = True

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Data
    print('==> Preparing data..')
    dataset_config = D(
        name=args.dataset_name,
        data_dir='~/tensorflow_datasets')

    if args.dataset_name in ["pathmnist", "bloodmnist", "dermamnist"]:
        resolution = 28
    elif args.dataset_name in ['cifar10', 'cifar100']:
        resolution = 32
    elif args.dataset_name in ['stl10']:
        resolution = 96
    elif args.dataset_name in ['imagenette', 'imagenet']:
        resolution = 256
    else:
        raise NotImplementedError

    mean, std = data_stats[args.dataset_name]
    transform_train = transforms.Compose([
        transforms.Resize(
            resolution, interpolation=InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(resolution),
        transforms.RandomCrop(resolution, padding=resolution//8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(
            resolution, interpolation=InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    if args.dataset_name in ["pathmnist", "bloodmnist", "dermamnist"]:
        info = INFO[args.dataset_name]
        num_classes = len(info['label'])
        DataClass = getattr(medmnist, info['python_class'])
        train_ds = DataClass(split='train', transform=None, download=True)
        test_ds = DataClass(split='test', transform=None, download=True)
        x_train, y_train = train_ds.imgs, train_ds.labels
        x_test, y_test = test_ds.imgs, test_ds.labels
        y_train, y_test = y_train.squeeze(), y_test.squeeze()

        # Subset of data
        group_size = args.group_size
        cls_idx = {}
        for i in range(num_classes):
            cls_idx[i] = np.where(y_train == i)[0][0:group_size]

        x_t = np.concatenate([x_train[idx]
                              for k, idx in cls_idx.items()], axis=0)
        y_t = np.concatenate([y_train[idx]
                              for k, idx in cls_idx.items()], axis=0)

        real_train = RealDataset(x_t, y_t, transform=transform_train)
        real_test = RealDataset(x_test, y_test, transform=transform_test)
        real_loader = torch.utils.data.DataLoader(
            real_train, batch_size=args.real_bs, shuffle=True, num_workers=4, pin_memory=True)
        real_testloader = torch.utils.data.DataLoader(
            real_test, batch_size=100, shuffle=False, num_workers=4)
        print(f'Train Dataset size {len(real_train)}')
        print(f'Test Dataset size {len(real_test)}')
        print(f'==> Real Training data loaded.. Size: {len(real_train)}')
    elif args.dataset_name == 'imagenet':
        imagenet_path_train = '/scratch/ssd002/datasets/imagenet/train'
        num_classes=1000
        class_map = {i: [] for i in range(num_classes)}
        train_dataset = ImageFolder(imagenet_path_train, transform=transform_train)
        
        for i, y in enumerate(train_dataset.targets):
            class_map[y].append(i)
        print({i: len(class_map[i]) for i in range(num_classes)})
        
        subset_index = []
        for i in range(num_classes):
            subset_index.extend(class_map[i][:args.num_data//num_classes]) 
        group_ids = [i for i in range(args.num_data//num_classes)]
        
        real_train = torch.utils.data.Subset(train_dataset, subset_index)
        print(f'Train Dataset size {len(real_train)}')
        imagenet_path_val = '/scratch/ssd002/datasets/imagenet/val'
        real_test = ImageFolder(imagenet_path_val, transform=transform_test)
        print(f'Validation Data subset size {len(real_test)}')
        real_testloader = torch.utils.data.DataLoader(
            real_test, batch_size=100, shuffle=False, num_workers=4)
    else:
        x_train, y_train, x_test, y_test = get_dataset(
            dataset_config, return_raw=True, resolution=resolution, train_only=False)

        num_classes = dataset_config.num_classes
        group_size = args.group_size
        group_num = x_train.shape[0] // group_size if x_train.shape[0] % group_size == 0 else x_train.shape[0] // group_size + 1
        group_ids = np.random.choice(
            list(range(group_num)), min(args.num_data // group_size, group_num), replace=False).tolist()
        print('>> Group Number', len(group_ids), 'Group IDs: ', group_ids)

        x_list = []
        y_list = []
        for group_id in group_ids:
            x_list.append(x_train[group_id*group_size:(group_id+1)*group_size])
            y_list.append(y_train[group_id*group_size:(group_id+1)*group_size])
        x_train = np.concatenate(x_list, axis=0)
        y_train = np.concatenate(y_list, axis=0)

        real_train = RealDataset(x_train, y_train, transform=transform_train)
        real_test = RealDataset(x_test, y_test, transform=transform_test)
        real_loader = torch.utils.data.DataLoader(
            real_train, batch_size=args.real_bs, shuffle=True, num_workers=4, pin_memory=True)
        real_testloader = torch.utils.data.DataLoader(
            real_test, batch_size=100, shuffle=False, num_workers=4)

    if args.syn_data_dir != '':
        print("Synthtic pattern: ", args.syn_pattern)
        def is_valid_file(path):
            folder = path.split('/')[-2]
            # group_id = int(path.split('/')[-1][5:7])
            # check if folder matches the pattern
            if re.fullmatch(pattern=args.syn_pattern, string=folder) is not None:
                # if group_id in group_ids:
                #     return True
                return True
            else:
                return False
            # if group_id in group_ids:
            #     return True

        syn_train = ImageFolder(
            args.syn_data_dir, is_valid_file=is_valid_file, transform=transform_train)
        print(f'==> Synthetic Training data loaded.. Size: {len(syn_train)}')
        syn_loader = torch.utils.data.DataLoader(
            syn_train, batch_size=args.syn_bs, shuffle=True, num_workers=4, pin_memory=True)
        train_dataset = syn_train
        
        if args.real_bs == 0:
            # As we loop over the real loader, we need to replace it with the synthetic loader
            real_train = syn_train
            real_loader = syn_loader
            syn_train = None
            syn_loader = None
            args.real_bs = args.syn_bs
    else:
        syn_train = None
        train_dataset = real_train

    # Model
    print('==> Building model..')
    net = get_network(args.model, num_classes=num_classes,
                      resolution=resolution).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=args.lr,
                              momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(net.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay)
    else:
        raise NotImplementedError
    scheduler = torch.optim.lr_scheduler.ChainedScheduler([
        torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.001, total_iters=args.warmup_steps),
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_steps, eta_min=args.lr * 0.001)])

    if args.syn_data_dir != "":
        data_name = f'{args.syn_pattern}_ngroup{len(group_ids)}_ndata{len(train_dataset)}'
    else:
        data_name = f'real_ngroup{len(group_ids)}_ndata{len(train_dataset)}'

    opt_name = f'{args.model}_opt{args.optimizer}_lr{args.lr}_wd{args.weight_decay}'
    # output_dir = f'{args.output}/{args.dataset_name}/{data_name}/{opt_name}/seed{args.seed}'
    output_dir = f'{args.output}/{args.dataset_name}/{data_name}/{opt_name}_synbs{args.syn_bs}_realbs{args.real_bs}/seed{args.seed}'

    args.data_name = data_name
    args.opt_name = opt_name
    args.ngroup = len(group_ids)
    args.ndata = len(train_dataset)
    args.group_ids = group_ids
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print(f'==> Output directory: {output_dir}')

    try:
        checkpoint = torch.load(f'{output_dir}/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        best_acc = checkpoint['best_acc']
        start_step = checkpoint['step'] + 1
        print('==> Resuming from checkpoint. start_step: ',
              start_step, 'best_acc: ', best_acc)
    except:
        start_step = 0
        best_acc = 0  # best test accuracy
        print('==> No checkpoint found. Train from scratch.')

    if args.log_wandb:
        wandb.init(
            project=f"{args.wandb_name}", config=args)

    n_steps_per_epoch = math.ceil(len(real_train) / args.real_bs)
    eval_interval = args.num_steps // args.num_evals
    print(f'Number of steps per epoch is {n_steps_per_epoch}')

    start_t = time.time()
    step = start_step

    syn_iter = iter(syn_loader)

    while start_step < args.num_steps:
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(real_loader):
            step = start_step + batch_idx
            if step == args.num_steps:
                break

            inputs, targets = inputs.to(device), targets.to(device)
            if syn_train is not None:
                try:
                    syn_inputs, syn_targets = next(syn_iter)
                except StopIteration:
                    syn_iter = iter(syn_loader)
                    syn_inputs, syn_targets = next(syn_iter)
                syn_inputs, syn_targets = syn_inputs.to(
                    device), syn_targets.to(device)
                inputs = torch.cat([inputs, syn_inputs], dim=0)
                targets = torch.cat([targets, syn_targets], dim=0)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if step % 100 == 0:
                elapsed = (time.time() - start_t)/100
                if args.log_wandb:
                    wandb.log({'train/train_loss': train_loss/total, 'train/acc': 100. *
                               correct/total, 'train/lr': get_lr(optimizer), 'monitor/time_per_step': elapsed}, step=step)
                else:
                    logging.info(
                        f'Step: {step}, Time per step: {elapsed:.4f}, Loss: {train_loss/total:.4f}, Acc: {100.*correct/total:.4f}, LR: {get_lr(optimizer):.4f}')
                start_t = time.time()

            if step % eval_interval == 0:
                print(f'Running eval at step {step}')
                test_loss, acc = test(net, real_testloader, criterion, device)
                if args.log_wandb:
                    wandb.log({'val/test_loss': test_loss,
                              'val/acc': acc}, step=step)
                else:
                    logging.info(
                        f'Val Loss (Real): {test_loss:.4f}, Val Acc (Real): {acc:.4f}')

                net.train()

                state = {
                    'net': net.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_acc': best_acc,
                    'step': step,
                }
                torch.save(state, f'{output_dir}/ckpt.pth')
                logging.info(
                    f'==> Saving Checkpoint. Step: {step}, Acc: {acc:.4f}')

                if acc > best_acc:
                    logging.info(
                        f'==> Saving Best Checkpoint. Step: {step}, Acc: {acc:.4f}')
                    state = {
                        'net': net.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_acc': acc,
                        'step': step,
                    }
                    torch.save(state, f'{output_dir}/best_ckpt.pth')
                    best_acc = acc

        start_step += n_steps_per_epoch

    # Final Evaluation
    test_loss, acc = test(net, real_testloader, criterion, device)
    if args.log_wandb:
        wandb.log({'val/test_loss': test_loss, 'val/acc': acc}, step=step)
    else:
        logging.info(
            f'Val Loss (Real): {test_loss:.4f}, Val Acc (Real): {acc:.4f}')

    state = {
        'net': net.state_dict(),
        'scheduler': scheduler.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_acc': best_acc,
        'step': step,
    }
    torch.save(state, f'{output_dir}/final_ckpt.pth')
    logging.info(
        f'==> Saving Checkpoint. Step: {step}, Acc: {acc:.4f}')
    wandb.finish()


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    main()
