"""
LOAD DATA from file.
"""

# pylint: disable=C0301,E1101,W0622,C0103,R0902,R0915

##
import os
import glob
import torch
import numpy as np
import torchvision.datasets as datasets
from torchvision.datasets import CIFAR10
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
'''
class Custom_dataset(Dataset):
    def __init__(self, train_flag,root, transforms_ano=None, transforms_ori=None):
        self.transform_A = transforms.Compose(transforms_ano)
        self.transform_B = transforms.Compose(transforms_ori)
        #self.unaligned = unaligned
        if train_flag:
            self.files_A = sorted(glob.glob(os.path.join(root, 'train/anotation' ) + '/*.*'))
            self.files_B = sorted(glob.glob(os.path.join(root, 'train/original') + '/*.*'))
        else:
            self.files_A = sorted(glob.glob(os.path.join(root, 'test/anotation') + '/*.*'))
            self.files_B = sorted(glob.glob(os.path.join(root, 'test/original') + '/*.*'))

    def __getitem__(self, index):
        # a = Image.open(self.files_A[index])
        # b = Image.open(self.files_B[index])
        item_A = self.transform_A(Image.open(self.files_A[index]))
        item_B = self.transform_B(Image.open(self.files_B[index]))
        return {'anotation': item_A, 'original': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
'''
def load_data(args,train_flag = False,valid_flag=False,test_flag=False):
    train_list = [transforms.Resize(args.img_size), transforms.RandomHorizontalFlip(p=0.8),transforms.ToTensor(), transforms.Normalize((0.5), (0.5))] #transforms.Grayscale(),
    valid_list = [transforms.Resize(args.img_size), transforms.ToTensor(), transforms.Normalize((0.5), (0.5))]
    test_list = [transforms.Resize(args.img_size), transforms.ToTensor(), transforms.Normalize((0.5), (0.5))]


    dataloader=None
    if train_flag:
        train_data = ImageFolder(root=args.dataroot + 'train', transform=transforms.Compose(train_list))
        dataloader = DataLoader(
            train_data,
            batch_size=args.batch_size, shuffle=True, num_workers=0,drop_last=True)
    elif valid_flag:
        valid_data = ImageFolder(root=args.dataroot + 'valid', transform=transforms.Compose(valid_list))
        dataloader = DataLoader(
            valid_data,
            batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True)
    elif test_flag:
        test_data = ImageFolder(root=args.dataroot + 'test', transform=transforms.Compose(test_list))
        dataloader = DataLoader(
            test_data,
            batch_size=1, shuffle=False, num_workers=0,drop_last=False)
    return dataloader

'''
def get_normal_data(train_ds):
    anomaly_count = train_ds.targets.count(0)
    train_ds.targets = train_ds.targets[anomaly_count:]
    train_ds.imgs = train_ds.imgs[anomaly_count:]
    train_ds.samples = train_ds.samples[anomaly_count:]
    train_ds.classes = train_ds.classes[1]
    del (train_ds.class_to_idx['anomaly'])
    return train_ds


def change_N_AN(test_dl):
    anomaly_count = test_dl.targets.count(0)
    test_dl.class_to_idx['anomaly'] = 1
    test_dl.class_to_idx['normal'] = 0
    test_dl.targets[:anomaly_count] = [x + 1 for x in test_dl.targets[:anomaly_count]]
    test_dl.targets[anomaly_count:] = [x - 1 for x in test_dl.targets[anomaly_count:]]
    return test_dl
'''
