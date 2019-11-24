from __future__ import print_function
import argparse
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
import torchvision
from resnext import resnext101_64x4d

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


# class ConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):

#         super(ConvBlock, self).__init__()

#         self.conv1 = nn.Conv2d(in_channels=in_channels,
#                               out_channels=out_channels,
#                               kernel_size=(3, 3), stride=(1, 1),
#                               padding=(1, 1), bias=False)

#         self.conv2 = nn.Conv2d(in_channels=out_channels,
#                               out_channels=out_channels,
#                               kernel_size=(3, 3), stride=(1, 1),
#                               padding=(1, 1), bias=False)

#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.bn2 = nn.BatchNorm2d(out_channels)

#         self.init_weights()

#     def init_weights(self):

#         init_layer(self.conv1)
#         init_layer(self.conv2)
#         init_bn(self.bn1)
#         init_bn(self.bn2)

#     def forward(self, input, pool_size=(2, 2), pool_type='avg'):

#         x = input
#         x = F.relu_(self.bn1(self.conv1(x)))
#         x = F.relu_(self.bn2(self.conv2(x)))
#         if pool_type == 'max':
#             x = F.max_pool2d(x, kernel_size=pool_size)
#         elif pool_type == 'avg':
#             x = F.avg_pool2d(x, kernel_size=pool_size)
#         else:
#             raise Exception('Incorrect argument!')

#         return x

# def init_layer(layer, nonlinearity='leaky_relu'):
#     """Initialize a Linear or Convolutional layer. """
#     nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlinearity)

#     if hasattr(layer, 'bias'):
#         if layer.bias is not None:
#             layer.bias.data.fill_(0.)


# def init_bn(bn):
#     """Initialize a Batchnorm layer. """

#     bn.bias.data.fill_(0.)
#     bn.running_mean.data.fill_(0.)
#     bn.weight.data.fill_(1.)
#     bn.running_var.data.fill_(1.)


# class Discriminator(nn.Module):

#     def __init__(self, classes_num=hparams.num_classes):
#         super(Discriminator, self).__init__()

#         self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
#         self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
#         self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
#         self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
#         self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
#         self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

#         self.fc = nn.Linear(2048, classes_num, bias=True)

#         self.init_weights()

#     def init_weights(self):
#         init_layer(self.fc)

#     def forward(self, input):
#         '''
#         Input: (batch_size, times_steps, freq_bins)'''

#         x = input[:, None, :, :]
#         '''(batch_size, 1, times_steps, freq_bins)'''

#         x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
#         x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
#         x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
#         x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
#         x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
#         x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')

#         x = torch.mean(x, dim=3)
#         (x, _) = torch.max(x, dim=2)
#         output = self.fc(x)

#         return output

# from .utils import load_state_dict_from_url


__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    count = 0
    for v in cfg:
        if v == 'M':
            count += 1
            if count > 3:
              layers += [nn.MaxPool2d(kernel_size=(2,1), stride=(2,1))]
            else:
              layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def vgg11(pretrained=False, progress=True, **kwargs):
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11', 'A', False, pretrained, progress, **kwargs)



def vgg11_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 11-layer model (configuration "A") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11_bn', 'A', True, pretrained, progress, **kwargs)



def vgg13(pretrained=False, progress=True, **kwargs):
    r"""VGG 13-layer model (configuration "B")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13', 'B', False, pretrained, progress, **kwargs)



def vgg13_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 13-layer model (configuration "B") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13_bn', 'B', True, pretrained, progress, **kwargs)



def vgg16(pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', 'D', True, pretrained, progress, **kwargs)



def vgg16_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16_bn', 'D', True, pretrained, progress, **kwargs)



def vgg19(pretrained=False, progress=True, **kwargs):
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19', 'E', False, pretrained, progress, **kwargs)



def vgg19_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19_bn', 'E', True, pretrained, progress, **kwargs)

def attention(q, k, v, d_k, mask=None, dropout=None):
    
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        scores = dropout(scores)
    
    output = torch.matmul(scores, v)

    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.q_linear = nn.Linear(self.d_k, d_model)
        self.v_linear = nn.Linear(self.d_k, d_model)
        self.k_linear = nn.Linear(self.d_k, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        
        bs = q.size(0)
        
        # perform linear operation and split into h heads
        
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * h * sl * d_model
       
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)# calculate attention using function we will define next

        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        concat = torch.sum(concat, axis=1)
        output = self.out(concat)
        # print(output.shape)
    
        return output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = vgg16(pretrained=False, progress=True)
        self.model.avgpool = nn.Sequential(nn.Flatten(start_dim=1, end_dim=2),
                                           )
        self.model.classifier = nn.Sequential(nn.LSTM(input_size=1024, hidden_size=160, bidirectional=True),
                                              )
        self.multiattention = MultiHeadAttention(heads=9, d_model=2880)
        # self.model.tplayer = nn.Sequential(nn.AdaptiveAvgPool2d((48,60)),
        #                                    nn.Flatten(),
        #                                    )
        self.model.endlayers = nn.Sequential(nn.Linear(in_features=2880, out_features=512),
                                              nn.ReLU(inplace=True),
                                              nn.Linear(in_features=512, out_features=512),
                                              nn.ReLU(inplace=True),
                                              nn.Linear(in_features=512, out_features=10),
                                              nn.Softmax(dim=1),
                                              )

    def forward(self, x):
      x = x.permute(0, 1, 3, 2)
      x = self.model.features(x)
      x = self.model.avgpool(x)
      x = x.transpose(1,2)
      x = self.model.classifier(x)
      x, (h, c) = x
      x = self.multiattention(x, x, x)
      x = self.model.endlayers(x)
      return x

