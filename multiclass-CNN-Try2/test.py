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
import seaborn as sn

import sklearn.metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

from hparams import hparams
from data import AudioData
from model import Discriminator
from metric import accuracy_metrics

epsilon = 0.0000000001

plt.switch_backend('agg')

def plot_cf(cf):
    fig = plt.figure()
    df_cm = pd.DataFrame(cf, range(hparams.num_classes), range(hparams.num_classes))
    sn.set(font_scale=1.4)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})
    return fig


def test(model_path, data=(hparams.valid_csv, hparams.dev_file), plot_auc='valid', plot_path=hparams.result_dir+'valid', best_thresh=None):

    test_dataset = AudioData(data_csv=data[0], data_file=data[1], ds_type='valid', augment=True,
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

            if hparams.dim3:
                inp = inp.view(-1, 1, 640, 64)
                inp = torch.cat([inp]*3, dim=1)

            pred_logits = discriminator(inp)

            pred_logits_list.append(pred_logits)
            labels_list.append(labels)
            img_names_list.append(img_names)

        pred_logits = torch.cat(pred_logits_list, dim=0)
        labels = torch.cat(labels_list, dim=0)

        auc, f1, acc, conf_mat = accuracy_metrics(labels, pred_logits, plot_auc=plot_auc, plot_path=plot_path, best_thresh=best_thresh)

        fig = plot_cf(conf_mat)
        plt.savefig(hparams.result_dir+'test_conf_mat.png')
        res = ' -- avg_acc - {0:.4f}'.format(acc['avg'])
        for it in range(10):
            res += ', acc_{}'.format(hparams.id_to_class[it])+' - {0:.4f}'.format(acc[it])
        print('== Test on -- '+model_path+ res)
        # print('== Test on -- '+model_path+' == \n\
        #     auc_{0} - {10:.4f}, auc_{1} - {11:.4f}, auc_{2} - {12:.4f}, auc_{3} - {13:.4f}, auc_{4} - {14:.4f}, auc_{5} - {15:.4f}, auc_{6} - {16:.4f}, auc_{7} - {17:.4f}, auc_{8} - {18:.4f}, auc_{9} - {19:.4f}, auc_micro - {20:.4f}, auc_macro - {21:.4f},\n\
        #     acc_{0} - {22:.4f}, acc_{1} - {23:.4f}, acc_{2} - {24:.4f}, acc_{3} - {25:.4f}, acc_{4} - {26:.4f}, acc_{5} - {27:.4f}, acc_{6} - {28:.4f}, acc_{7} - {29:.4f}, acc_{8} - {30:.4f}, acc_{9} - {31:.4f}, acc_avg - {32:.4f},\n\
        #     f1_{0} - {33:.4f}, f1_{1} - {34:.4f}, f1_{2} - {35:.4f}, f1_{3} - {36:.4f}, f1_{4} - {37:.4f}, f1_{5} - {38:.4f}, f1_{6} - {39:.4f}, f1_{7} - {40:.4f}, f1_{8} - {41:.4f}, f1_{9} - {42:.4f}, f1_micro - {42:.4f}, f1_macro - {43:.4f}, =='.\
        #     format([hparams.id_to_class[it] for it in range(10)]+[auc[it] for it in range(10)]+[auc['micro'], auc['macro']]+[acc[it] for it in range(10)]+[acc['avg']]+[f1[it] for it in range(10)]+[f1['micro'], f1['macro']]))
    return acc['avg']
