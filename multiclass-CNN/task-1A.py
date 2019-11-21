#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import torch

class Hparams():
    def __init__(self):

        self.cuda = True if torch.cuda.is_available() else False

        """
        Data Parameters
        """

        # os.makedirs('../input', exist_ok=True)
        os.makedirs('../model', exist_ok=True)
        # os.makedirs('../data/', exist_ok=True)
        os.makedirs('../results/', exist_ok=True)

        self.train_csv = '/data1/amey/ASR-data/TAU-urban-acoustic-scenes-2019-development/evaluation_setup/train.csv'
        self.valid_csv = '/data1/amey/ASR-data/TAU-urban-acoustic-scenes-2019-development/evaluation_setup/valid.csv'
#         self.submit_csv = '../input/task1asubmitcsv/test.csv'
        self.submit_csv = '/data1/amey/ASR-data/TAU-urban-acoustic-scenes-2019-leaderboard/evaluation_setup/test.csv'

        self.dev_file = '../features/logmel_64frames_64melbins/TAU-urban-acoustic-scenes-2019-development.h5'
#         self.submit_file = '../input/task1aleaderboardh5/TAU-urban-acoustic-scenes-2019-leaderboard.h5'
        self.submit_file = '../features/logmel_64frames_64melbins/TAU-urban-acoustic-scenes-2019-leaderboard.h5'

        """
        Model Parameters
        """

        os.makedirs('../model/', exist_ok=True)

        self.input_shape = (640, 64)
        self.num_channel = 64
        self.num_classes = 10

        self.id_to_class = {
            0: 'airport',
            1: 'shopping_mall',
            2: 'metro_station',
            3: 'street_pedestrian',
            4: 'public_square',
            5: 'street_traffic',
            6: 'tram',
            7: 'bus',
            8: 'metro',
            9: 'park',
        }


        """
        Training parameters
        """

        self.gpu_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device_ids = [0,1]

        self.pretrained = False

        self.thresh = 0.5
        self.repeat_infer = 1

        self.num_epochs = 200
        self.batch_size = 32

        self.learning_rate = 0.00001

        self.momentum1 = 0.5
        self.momentum2 = 0.999

        self.avg_mode = 'micro'

        self.print_interval = 1000

        ################################################################################################################################################
        self.exp_name = 'multiclass-CNN/'
        ################################################################################################################################################

        self.result_dir = '../results/'+self.exp_name
        os.makedirs(self.result_dir, exist_ok=True)

        self.model_dir = '../model/' + self.exp_name
        os.makedirs(self.model_dir, exist_ok=True)

        self.model = self.model_dir + 'model'


hparams = Hparams()


# In[2]:


# from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
import torchvision


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # self.model = resnext101_64x4d(num_classes=1000, pretrained='imagenet')
        # num_ftrs = self.model.last_linear.in_features
        # self.model.last_linear = nn.Sequential(
        #                             nn.Linear(num_ftrs, hparams.num_classes),
        #                             nn.Sigmoid())
        self.model = models.densenet121(pretrained=False, progress=True)
        num_ftrs = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
                                    nn.Linear(num_ftrs, hparams.num_classes),
                                    nn.Softmax(dim=1))

        # self.model = models.squeezenet1_0(pretrained=True, progress=True)
        # self.model.classifier[1] = nn.Conv2d(512, 1, kernel_size=(1,1), stride=(1,1))
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        return x


# In[4]:


# from __future__ import print_function, division
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

class AudioData(Dataset):

  def __init__(self, data_csv, data_file, transform=None, input_shape=hparams.input_shape, pre_process=None, ds_type=''):
        'Initialization'
        self.data_csv = data_csv
        self.data_file = data_file
        self.input_shape = hparams.input_shape
        self.ds_type = ds_type
        self.transform = transform
        self.pre_process = pre_process
        self.data_frame = pd.read_csv(data_csv)
        data = h5py.File(data_file)
        self.feature_data = {}
        for idx in range(len(data['audio_name'])):
            self.feature_data[str(data['audio_name'][idx].decode("utf-8"))] = { 'feature': data['feature'][idx] }
        # print(self.feature_data)
        # print(self.data_frame.iloc[0,:])


  def __len__(self):
        'Denotes the total number of samples'
        return len(self.data_frame)

  def __getitem__(self, index):
        'Generates one sample of data'

        file_name = str(self.data_frame.iloc[index, 0].split('/')[1])
        label = 0#torch.tensor(self.data_frame.iloc[index, 1])
        inp = self.feature_data[file_name]['feature']

        return (inp, label, str(file_name.split('.')[0]))


# In[5]:


import time
import code
import os, torch, sys
import torch
import csv
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from torchvision.utils import save_image
from torch.autograd import Variable
from torch import optim
from skimage.util import random_noise
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

import sklearn.metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

epsilon = 0.0000000001

plt.switch_backend('agg')

def submit(model_path, data=(hparams.submit_csv, hparams.submit_file), plot_auc='submit', plot_path=hparams.result_dir+'valid', best_thresh=None):

    test_dataset = AudioData(data_csv=data[0], data_file=data[1], ds_type='submit',
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                        ]))

    test_loader = DataLoader(test_dataset, batch_size=hparams.batch_size,
                            shuffle=True, num_workers=2)


    discriminator = Discriminator().to(hparams.gpu_device)
    if hparams.cuda:
        discriminator = nn.DataParallel(discriminator, device_ids=hparams.device_ids)
    checkpoint = torch.load(model_path, map_location=hparams.gpu_device)
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

    discriminator = discriminator.eval()
    # print('Model loaded')

    Tensor = torch.cuda.FloatTensor if hparams.cuda else torch.FloatTensor

    print('Testing model on {0} examples. '.format(len(test_dataset)))

    with torch.no_grad():
        pred_logits_list = []
        labels_list = []
        img_names_list = []
        # for _ in range(hparams.repeat_infer):
        for (inp, labels, img_names) in tqdm(test_loader):
            inp = Variable(inp.float(), requires_grad=False)
            labels = Variable(labels.long(), requires_grad=False)

            inp = inp.to(hparams.gpu_device)
            labels = labels.to(hparams.gpu_device)

            inp = inp.view(-1, 1, 640, 64)
            inp = torch.cat([inp]*3, dim=1)

            pred_logits = discriminator(inp)

            pred_logits_list.append(pred_logits)
            labels_list.append(labels)
            img_names_list += list(img_names)

        pred_logits = torch.cat(pred_logits_list, dim=0)
        labels = torch.cat(labels_list, dim=0)

        pred_labels = pred_logits.max(1)[1]

        if hparams.cuda:
            pred_labels = pred_labels.cpu()
        pred_labels = pred_labels.numpy()

        data_frame = pd.DataFrame({'Id': img_names_list, 'Scene_label': pred_labels})
        data_frame.to_csv('submission.csv', index=False)

    print('Predictions saved to csv file.')


# In[6]:


submit('../model/multiclass-CNN/model.best')


# In[ ]:


# get_ipython().system('ls ../input/task1abasedensenet/model.best')


# In[ ]:
