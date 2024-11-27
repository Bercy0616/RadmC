# ------------------------------------------------------------------------------
# --coding='utf-8'--
# Written by czifan (czifan@pku.edu.cn)
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import logging
import numpy as np
import torch
import configparser
import SimpleITK as sitk
import pandas as pd
from tqdm import tqdm
from sklearn import metrics 
from sklearn.metrics import f1_score
#from lifelines.utils import concordance_index

def get_yoden_threshold(data,true = 'label',prob = 'out1'):
    fpr, tpr, thresholds=metrics.roc_curve(data[true],data[prob],pos_label=1)
    yoden_value = list(tpr-fpr)
    yoden_index = yoden_value.index(np.max(yoden_value))
    threshold = thresholds[yoden_index]
    return threshold,[fpr[yoden_index],tpr[yoden_index]]

def get_class(data,prob,threshold):
    probablity = data[prob].tolist()
    predict_class = []
    for v in probablity:
        if v >threshold:
            predict_class.append(1)
        else:
            predict_class.append(0)
    return predict_class

def sensitivity_specificity(df,label,prob):
    matrix = metrics.confusion_matrix(df[label],df[prob])
    sen = matrix[0][0]/(matrix[0][0]+matrix[0][1])
    spe = matrix[1][1]/(matrix[1][0]+matrix[1][1])
    return sen,spe


def get_result_binary(output_all, output_pred_all, label_all):
    acc = metrics.accuracy_score(label_all.cpu(), output_all.cpu())
    auc = metrics.roc_auc_score(label_all.cpu(), output_pred_all.cpu())
    return [acc,auc]

def adjust_learning_rate(optimizer, epoch, lr, lr_decay_rate, warm_epoch=10, warmup=False):
    ''' Adjusts learning rate according to (epoch, lr and lr_decay_rate)

    :param optimizer: (torch.optim object)
    :param epoch: (int)
    :param lr: (float) the initial learning rate
    :param lr_decay_rate: (float) learning rate decay rate
    :return lr_: (float) updated learning rate
    '''
    if warmup:
        if epoch < warm_epoch:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr / (warm_epoch-epoch)
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr / (1+(epoch-warm_epoch)*lr_decay_rate)
        #print(lr / (warm_epoch-epoch))
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr / (1+epoch*lr_decay_rate)
    return optimizer.param_groups[0]['lr']

def reindex(df):
    df.reset_index(inplace = True)
    del df['index']
    return df

def default_loader(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))