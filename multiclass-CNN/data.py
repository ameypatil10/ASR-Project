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

import code

from hparams import hparams

epsilon = 0.0000000001

class AudioData(Dataset):

  def __init__(self, data_csv, data_file, scalar_file=hparams.scalar_file, transform=None, input_shape=hparams.input_shape, pre_process=None, ds_type=''):
        'Initialization'
        self.data_csv = data_csv
        self.data_file = data_file
        self.input_shape = hparams.input_shape
        self.ds_type = ds_type
        self.transform = transform
        self.pre_process = pre_process
        self.data_frame = pd.read_csv(data_csv)
        scalar = h5py.File(scalar_file)
        self.mean = scalar['mean'][:]
        self.std = scalar['std'][:]
        data = h5py.File(data_file)
        self.feature_data = {}
        for idx in range(len(data['audio_name'])):
            self.feature_data[str(data['audio_name'][idx].decode("utf-8"))] = { 'feature': data['feature'][idx], 'scene_label': str(data['scene_label'][idx].decode("utf-8")),'source_label': str(data['source_label'][idx].decode("utf-8"))}
        # print(self.feature_data)
        # print(self.data_frame.iloc[0,:])


  def __len__(self):
        'Denotes the total number of samples'
        return len(self.data_frame)

  def __getitem__(self, index):
        'Generates one sample of data'

        file_name = str(self.data_frame.iloc[index, 0].split('/')[1])
        label = torch.tensor(self.data_frame.iloc[index, 1])
        inp = self.feature_data[file_name]['feature']
        inp = (inp-self.mean)/self.std

        return (inp, label, file_name.split('.')[0])
