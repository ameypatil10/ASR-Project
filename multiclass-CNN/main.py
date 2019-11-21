import argparse
import os
import numpy as np
from train import *
from test import *
from hparams import hparams
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train')

args = parser.parse_args()

if args.mode == 'train':
    train()

if args.mode == 'test':
    auc = test(hparams.model+'.best', data=(hparams.valid_csv, hparams.dev_file), plot_auc='test', plot_path=hparams.result_dir+'test')
