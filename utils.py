import torch
import collections
import os
from torch.utils.data import DataLoader, ConcatDataset
from models_resnet import mono_res, mono_vit, mono_vit_mlp, mono_res_mlp, \
    stereo_res, stereo_vit, stereo_vit_mlp, stereo_res_mlp, stereo_res_lstm, stereo_vit_lstm
from senseiloader import SENSEIDataset
import numpy as np


def to_device(input, device):
    if torch.is_tensor(input):
        return input.to(device=device)
    elif isinstance(input, str):
        return input
    elif isinstance(input, collections.Mapping):
        return {k: to_device(sample, device=device) for k, sample in input.items()}
    elif isinstance(input, collections.Sequence):
        return [to_device(sample, device=device) for sample in input]
    else:
        raise TypeError(f"Input must contain tensor, dict or list, found {type(input)}")


def get_model(model, pretrained=False):
    '''
    Mono setting
    '''
    if model == 'mono_res':
        out_model = mono_res(3)
    elif model == 'mono_vit':
        out_model = mono_vit(3)
    elif model == 'mono_vit_mlp':
        out_model = mono_vit_mlp(3)
    elif model == 'mono_res_mlp':
        out_model = mono_res_mlp(3)
        '''
        Stereo setting
        '''
    elif model == 'stereo_res':
        out_model = stereo_res(6)
    elif model == 'stereo_vit':
        out_model = stereo_vit(6)
    elif model == 'stereo_vit_mlp':
        out_model = stereo_vit_mlp(6)
    elif model == 'stereo_res_mlp':
        out_model = stereo_res_mlp(6)
    elif model == 'stereo_res_lstm':
        out_model = stereo_res_lstm(6)
    elif model == 'stereo_vit_lstm':
        out_model = stereo_vit_lstm(6)
    else:
        print('model selected does not exist')
    return out_model


def prepare_dataloader(data_directory, is_stereo, mode_flag, batch_size, num_workers, size):
    datasets = [SENSEIDataset(data_directory, size, is_stereo, mode_flag)]
    dataset = ConcatDataset(datasets)
    n_img = len(dataset)
    print('mode:', mode_flag, ': Use a dataset with ', n_img, 'images')
    if mode_flag == 'train':
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=num_workers,
                            pin_memory=True, drop_last=True)
    else:
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=True, drop_last=False)
    return n_img, loader


def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines

def transform2Dto3D(Z, pt2d, alpha, beta, ox, oy):  # (u, v) is 2D laser point and changing for each image
    u, v = pt2d[0]
    X = (Z * (u - ox)) / alpha
    Y = (Z * (v - oy)) / beta

    return np.array([X, Y, Z])