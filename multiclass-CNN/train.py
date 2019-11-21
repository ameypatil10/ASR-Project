import time
import code
import os, torch
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from torchvision.utils import save_image
from torch.autograd import Variable
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.sampler import WeightedRandomSampler
from tensorboardX import SummaryWriter
from functools import reduce
import operator
from tqdm import tqdm
import seaborn as sn
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from skimage.util import random_noise

from hparams import hparams
from data import AudioData
from model import Discriminator
from metric import accuracy_metrics

plt.switch_backend('agg')

def weights_init_normal(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Conv2d):
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def plot_cf(cf):
    fig = plt.figure()
    df_cm = pd.DataFrame(cf, range(hparams.num_classes), range(hparams.num_classes))
    sn.set(font_scale=1.4)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})
    return fig

def train(resume=False):

    writer = SummaryWriter('../runs/'+hparams.exp_name)

    for k in hparams.__dict__.keys():
        writer.add_text(str(k), str(hparams.__dict__[k]))

    train_dataset = AudioData(data_csv=hparams.train_csv, data_file=hparams.dev_file, ds_type='train',
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                        ]))

    validation_dataset = AudioData(data_csv=hparams.valid_csv, data_file=hparams.dev_file, ds_type='valid',
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                        ]))

    # train_sampler = WeightedRandomSampler()

    train_loader = DataLoader(train_dataset, batch_size=hparams.batch_size,
                            shuffle=True, num_workers=2)

    validation_loader = DataLoader(validation_dataset, batch_size=hparams.batch_size,
                            shuffle=True, num_workers=2)

    print('loaded train data of length : {}'.format(len(train_dataset)))

    adversarial_loss = torch.nn.CrossEntropyLoss().to(hparams.gpu_device)
    discriminator = Discriminator().to(hparams.gpu_device)

    if hparams.cuda:
        discriminator = nn.DataParallel(discriminator, device_ids=hparams.device_ids)

    params_count = 0
    for param in discriminator.parameters():
        params_count += np.prod(param.size())
    print('Model has {0} trainable parameters'.format(params_count))

    if not hparams.pretrained:
        discriminator.apply(weights_init_normal)

    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=hparams.learning_rate)

    scheduler_D = ReduceLROnPlateau(optimizer_D, mode='min', factor=0.3, patience=4, verbose=True, cooldown=0)

    Tensor = torch.cuda.FloatTensor if hparams.cuda else torch.FloatTensor

    def validation(discriminator, send_stats=False, epoch=0):
        print('Validating model on {0} examples. '.format(len(validation_dataset)))
        discriminator_ = discriminator.eval()

        with torch.no_grad():
            pred_logits_list = []
            labels_list = []

            for (inp, labels, imgs_names) in tqdm(validation_loader):
                inp = Variable(inp.float(), requires_grad=False)
                labels = Variable(labels.long(), requires_grad=False)

                if hparams.dim3:
                    inp = inp.view(-1, 1, 640, 64)
                    inp = torch.cat([inp]*3, dim=1)

                inp = inp.to(hparams.gpu_device)
                labels = labels.to(hparams.gpu_device)

                pred_logits = discriminator_(inp)

                pred_logits_list.append(pred_logits)
                labels_list.append(labels)

            pred_logits = torch.cat(pred_logits_list, dim=0)
            labels = torch.cat(labels_list, dim=0)

            val_loss = adversarial_loss(pred_logits, labels)

        return accuracy_metrics(labels.long(), pred_logits), val_loss #, plot_auc='train_val_'+str(epoch+1), plot_path=hparams.result_dir+'train_val_{}_'.format(epoch)), val_loss

    print('Starting training.. (log saved in:{})'.format(hparams.exp_name))
    start_time = time.time()
    best_valid_acc = 0

    # print(model)
    for epoch in range(hparams.num_epochs):
        for batch, (inp, labels, imgs_name) in enumerate(tqdm(train_loader)):

            inp = Variable(inp.float(), requires_grad=False)
            labels = Variable(labels.long(), requires_grad=False)

            inp = inp.to(hparams.gpu_device)
            labels = labels.to(hparams.gpu_device)

            if hparams.dim3:
                inp = inp.view(-1, 1, 640, 64)
                inp = torch.cat([inp]*3, dim=1)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            pred_logits = discriminator(inp)

            d_loss = adversarial_loss(pred_logits, labels)

            d_loss.backward()
            optimizer_D.step()

            writer.add_scalar('d_loss', d_loss.item(), global_step=batch+epoch*len(train_loader))

            # if batch % hparams.print_interval == 0:
            #     pred_labels = (pred_logits >= hparams.thresh)
            #     pred_labels = pred_labels.float()
            #     auc, f1, acc, _, _ = accuracy_metrics(pred_labels, labels.long(), pred_logits)
            #     print('[Epoch - {0:.1f}, batch - {1:.3f}, d_loss - {2:.6f}, acc - {3:.4f}, f1 - {4:.5f}, auc - {5:.4f}]'.\
            #     format(1.0*epoch, 100.0*batch/len(train_loader), d_loss.item(), acc['avg'], f1[hparams.avg_mode], auc[hparams.avg_mode]))

        (val_auc, val_f1, val_acc, val_conf_mat), val_loss = validation(discriminator, epoch=epoch)

        fig = plot_cf(val_conf_mat)
        writer.add_figure('val_conf', fig, global_step=epoch)
        plt.close(fig)
        for lbl in range(hparams.num_classes):
            writer.add_scalar('val_f1_{}'.format(hparams.id_to_class[lbl]), val_f1[lbl], global_step=epoch)
            writer.add_scalar('val_auc_{}'.format(hparams.id_to_class[lbl]), val_auc[lbl], global_step=epoch)
            writer.add_scalar('val_acc_{}'.format(hparams.id_to_class[lbl]), val_acc[lbl], global_step=epoch)
        writer.add_scalar('val_f1_{}'.format('micro'), val_f1['micro'], global_step=epoch)
        writer.add_scalar('val_auc_{}'.format('micro'), val_auc['micro'], global_step=epoch)
        writer.add_scalar('val_f1_{}'.format('macro'), val_f1['macro'], global_step=epoch)
        writer.add_scalar('val_auc_{}'.format('macro'), val_auc['macro'], global_step=epoch)
        writer.add_scalar('val_loss', val_loss, global_step=epoch)
        writer.add_scalar('val_f1', val_f1[hparams.avg_mode], global_step=epoch)
        writer.add_scalar('val_auc', val_auc[hparams.avg_mode], global_step=epoch)
        writer.add_scalar('val_acc', val_acc['avg'], global_step=epoch)
        scheduler_D.step(val_loss)
        writer.add_scalar('learning_rate', optimizer_D.param_groups[0]['lr'], global_step=epoch)

        # torch.save({
        #     'epoch': epoch,
        #     'discriminator_state_dict': discriminator.state_dict(),
        #     'optimizer_D_state_dict': optimizer_D.state_dict(),
        #     }, hparams.model+'.'+str(epoch))
        if best_valid_acc <= val_acc['avg']:
            best_valid_acc = val_acc['avg']
            fig = plot_cf(val_conf_mat)
            writer.add_figure('best_val_conf', fig, global_step=epoch)
            plt.close(fig)
            torch.save({
                'epoch': epoch,
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                }, hparams.model+'.best')
            print('best model on validation set saved.')

        print('[Epoch - {0:.1f} ---> val_auc - {1:.4f}, current_lr - {2:.6f}, val_loss - {3:.4f}, best_val_acc - {4:.4f}, val_acc - {5:.4f}, val_f1 - {6:.4f}] - time - {7:.1f}'\
            .format(1.0*epoch, val_auc[hparams.avg_mode], optimizer_D.param_groups[0]['lr'], val_loss, best_valid_acc, val_acc['avg'], val_f1[hparams.avg_mode], time.time()-start_time))
        start_time = time.time()
