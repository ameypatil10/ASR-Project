from __future__ import print_function, division
import os
import json
import csv
import torch
import h5py
import random
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from PIL import ImageFilter
from nb_SparseImageWarp import *
import code

from hparams import hparams

epsilon = 0.0000000001

def time_warp(spec, W=5):
    num_rows = spec.shape[1]
    spec_len = spec.shape[2]
    device = 'cpu'

    y = num_rows//2
    horizontal_line_at_ctr = spec[0][y]
    assert len(horizontal_line_at_ctr) == spec_len

    point_to_warp = horizontal_line_at_ctr[random.randrange(W, spec_len - W)]
    assert isinstance(point_to_warp, torch.Tensor)

    # Uniform distribution from (0,W) with chance to be up to W negative
    dist_to_warp = random.randrange(-W, W)
    src_pts, dest_pts = (torch.tensor([[[y, point_to_warp]]], device=device),
                         torch.tensor([[[y, point_to_warp + dist_to_warp]]], device=device))
    warped_spectro, dense_flows = sparse_image_warp(spec, src_pts, dest_pts)
    return warped_spectro.squeeze(3)


def freq_mask(spec, F=30, num_masks=1, replace_with_zero=False):
    cloned = spec.clone()
    num_mel_channels = cloned.shape[1]

    for i in range(0, num_masks):
        f = random.randrange(0, F)
        f_zero = random.randrange(0, num_mel_channels - f)

        # avoids randrange error if values are equal and range is empty
        if (f_zero == f_zero + f): return cloned

        mask_end = random.randrange(f_zero, f_zero + f)
        if (replace_with_zero): cloned[0][f_zero:mask_end] = 0
        else: cloned[0][f_zero:mask_end] = cloned.mean()

    return cloned

def time_mask(spec, T=40, num_masks=1, replace_with_zero=False):
    cloned = spec.clone()
    len_spectro = cloned.shape[2]

    for i in range(0, num_masks):
        t = random.randrange(0, T)
        t_zero = random.randrange(0, len_spectro - t)

        # avoids randrange error if values are equal and range is empty
        if (t_zero == t_zero + t): return cloned

        mask_end = random.randrange(t_zero, t_zero + t)
        if (replace_with_zero): cloned[0][:,t_zero:mask_end] = 0
        else: cloned[0][:,t_zero:mask_end] = cloned.mean()
    return cloned


class AudioData(Dataset):

  def __init__(self, data_csv, data_file, scalar_file=hparams.scalar_file, augment=True, aug_prob=0.75, transform=None, input_shape=hparams.input_shape, pre_process=None, ds_type=''):
        'Initialization'
        self.data_csv = data_csv
        self.data_file = data_file
        self.input_shape = hparams.input_shape
        self.ds_type = ds_type
        self.transform = transform
        self.pre_process = pre_process
        self.aug_prob = aug_prob
        self.data_frame = pd.read_csv(data_csv)
        scalar = h5py.File(scalar_file)
        self.mean = scalar['mean'][:]
        self.std = scalar['std'][:]
        data = h5py.File(data_file)
        self.feature_data = {}
        self.augment = augment
        for idx in range(len(data['audio_name'])):
            self.feature_data[str(data['audio_name'][idx].decode("utf-8"))] = { 'feature': data['feature'][idx], 'scene_label': str(data['scene_label'][idx].decode("utf-8")),'source_label': str(data['source_label'][idx].decode("utf-8"))}
        self.data_ind = {i: [] for i in range(hparams.num_classes)}
        for idx in self.feature_data.keys():
            self.data_ind[hparams.class_to_id[self.feature_data[idx]['scene_label']]].append(idx)
        self.data_cnt = {lbl: len(self.data_ind[lbl]) for lbl in range(hparams.num_classes)}
        # print(self.feature_data)
        # print(self.data_frame.iloc[0,:])
        # code.interact(local=locals())


  def __len__(self):
        'Denotes the total number of samples'
        return len(self.data_frame)

  def __getitem__(self, index):
        'Generates one sample of data'

        file_name = str(self.data_frame.iloc[index, 0].split('/')[1])
        label = torch.tensor(self.data_frame.iloc[index, 1])
        inp = self.feature_data[file_name]['feature']
        inp = (inp-self.mean)/self.std
        if self.augment and np.random.uniform() > self.aug_prob:
            inp = torch.tensor(inp.reshape(1, hparams.input_shape[1], hparams.input_shape[0]))
            inp = time_mask(freq_mask(time_warp(inp), num_masks=2), num_masks=2)
            inp = inp.squeeze(0).T
            inp = inp.numpy()
            # augf = np.random.choice(self.data_ind[int(label.item())])
            # inp2 = self.feature_data[augf]['feature']
            # split = np.random.randint(0, int(7*hparams.input_shape[0]/8))
            # inp[split:split+int(hparams.input_shape[0]/8), :] = inp2[split:split+int(hparams.input_shape[0]/8), :]
        inp = inp.reshape(1, hparams.input_shape[1], hparams.input_shape[0])
        return (inp, label, file_name.split('.')[0])
