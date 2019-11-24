from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
import torchvision
from resnext import resnext101_64x4d
from vgg import vggm

from hparams import hparams

# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         self.model = models.densenet121(pretrained=False, progress=True)
#         num_ftrs = self.model.classifier.in_features
#         self.model.classifier = nn.Sequential(
#                                     nn.Linear(num_ftrs, hparams.num_classes, bias=True),
#                                     )
#
#     def forward(self, x):
#         x = self.model(x)
#         return x
#
# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         self.model = models.vgg16(pretrained=False, progress=True)
#         num_ftrs = self.model.classifier[6].in_features
#         self.model.classifier[6] = nn.Sequential(
#                                     nn.Linear(num_ftrs, hparams.num_classes, bias=True),
#                                     )
#
#     def forward(self, x):
#         x = self.model(x)
#         return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = vggm(num_classes=hparams.num_classes, pretrained=None)

    def forward(self, x):
        x = self.model(x)
        return x


# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         self.model = models.resnet18(pretrained=False, progress=True)
#         # num_ftrs = self.model.classifier[6].in_features
#         # self.model.classifier[6] = nn.Sequential(
#         #                             nn.Linear(num_ftrs, hparams.num_classes, bias=True),
#         #                             )
#         num_ftrs = self.model.fc.in_features
#         self.model.fc = nn.Linear(num_ftrs, hparams.num_classes)
#
#     def forward(self, x):
#         x = self.model(x)
#         return x

#
# class ConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#
#         super(ConvBlock, self).__init__()
#
#         self.conv1 = nn.Conv2d(in_channels=in_channels,
#                               out_channels=out_channels,
#                               kernel_size=(3, 3), stride=(1, 1),
#                               padding=(1, 1), bias=False)
#
#         self.conv2 = nn.Conv2d(in_channels=out_channels,
#                               out_channels=out_channels,
#                               kernel_size=(3, 3), stride=(1, 1),
#                               padding=(1, 1), bias=False)
#
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#
#         self.init_weights()
#
#     def init_weights(self):
#
#         init_layer(self.conv1)
#         init_layer(self.conv2)
#         init_bn(self.bn1)
#         init_bn(self.bn2)
#
#     def forward(self, input, pool_size=(2, 2), pool_type='avg'):
#
#         x = input
#         x = F.relu_(self.bn1(self.conv1(x)))
#         x = F.relu_(self.bn2(self.conv2(x)))
#         if pool_type == 'max':
#             x = F.max_pool2d(x, kernel_size=pool_size)
#         elif pool_type == 'avg':
#             x = F.avg_pool2d(x, kernel_size=pool_size)
#         else:
#             raise Exception('Incorrect argument!')
#
#         return x
#
# def init_layer(layer, nonlinearity='leaky_relu'):
#     """Initialize a Linear or Convolutional layer. """
#     nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlinearity)
#
#     if hasattr(layer, 'bias'):
#         if layer.bias is not None:
#             layer.bias.data.fill_(0.)
#
#
# def init_bn(bn):
#     """Initialize a Batchnorm layer. """
#
#     bn.bias.data.fill_(0.)
#     bn.running_mean.data.fill_(0.)
#     bn.weight.data.fill_(1.)
#     bn.running_var.data.fill_(1.)
#
#
# class Discriminator(nn.Module):
#
#     def __init__(self, classes_num=hparams.num_classes):
#         super(Discriminator, self).__init__()
#
#         self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
#         self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
#         self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
#         self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
#         self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
#         self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)
#
#         self.fc = nn.Linear(2048, classes_num, bias=True)
#
#         self.init_weights()
#
#     def init_weights(self):
#         init_layer(self.fc)
#
#     def forward(self, input):
#         '''
#         Input: (batch_size, times_steps, freq_bins)'''
#
#         x = input[:, None, :, :]
#         '''(batch_size, 1, times_steps, freq_bins)'''
#
#         x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
#         x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
#         x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
#         x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
#         x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
#         x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
#
#         x = torch.mean(x, dim=3)
#         (x, _) = torch.max(x, dim=2)
#         output = self.fc(x)
#
#         return output
