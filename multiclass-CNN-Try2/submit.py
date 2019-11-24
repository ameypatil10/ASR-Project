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

from hparams import hparams
from data import AudioData
from model import Discriminator

epsilon = 0.0000000001

plt.switch_backend('agg')

def test(model_path, submit_csv=hparams.submit_file, submit_file=hparams.submit_file, best_thresh=None):

    test_dataset = AudioData(data_csv=submit_csv, data_file=submit_file, ds_type='submit',
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                        ]))

    test_loader = DataLoader(test_dataset, batch_size=hparams.batch_size,
                            shuffle=False, num_workers=2)


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
            img_names_list.append(img_names)

        pred_logits = torch.cat(pred_logits_list, dim=0)
        labels = torch.cat(labels_list, dim=0)

        pred_labels = pred_logits.max(1)[1]

        with open 
        for idx in range(len(test_dataset)):


    print('Predictions saved to csv file.')
