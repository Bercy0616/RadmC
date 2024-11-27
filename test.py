# ------------------------------------------------------------------------------
# --coding='utf-8'--
# Written by Bercy
# ------------------------------------------------------------------------------

import os
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from datasets import mission_Dataset
from model_cnn_3d.res_fpn import *
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils import get_result_binary
from utils import adjust_learning_rate,get_yoden_threshold,get_class,sensitivity_specificity


def test_model(model,loader,criterion):
    model.eval()
    loss_sum = 0
    total = 0
    output_all = torch.Tensor([]).to(device)
    output_pred_all = torch.Tensor([]).to(device)
    label_all = torch.IntTensor([]).to(device)

    for inputs, labels, f1, f2, f3 in loader:
        inputs, labels = inputs.float().to(device), labels.to(device)
        with torch.no_grad():
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss_sum += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            
            output_all = torch.cat((output_all, predicted.float()), 0)
            output_pred_all = torch.cat((output_pred_all,outputs[:,1]),0)
            label_all = torch.cat((label_all, labels.int()), 0)
            
    loss_sum = round(loss_sum/total,4)
    results = get_result_binary(output_all.cpu(), output_pred_all.cpu(),label_all.cpu())#
    
    results_df = pd.DataFrame()
    results_df['label'] = label_all.cpu()
    results_df['prob'] = output_pred_all.cpu()
    return loss_sum,results,results_df

def build_model(name,path_list,device,device_ids,m,layer_num,epochs_sum):
    outfolder = 'weight'
    # builds network|criterion|optimizer based on configuration
    num_class = 2

    if m == 'res10_rpn':
        model = ResFPN(BasicBlock,[1,1,1,1],fc_num = layer_num).to(device)
    elif m == 'res18_rpn':
        model = ResFPN(BasicBlock,[2,2,2,2],fc_num = layer_num).to(device)
    elif m == 'res50_rpn':
        model = ResFPN(BasicBlock,[3,4,6,3],fc_num = layer_num).to(device)
    elif m == 'shuffle':
        model = ShuffleNet(num_classes=2, groups=1, width_mult=1.,fc_num = layer_num).to(device)
    elif m == 'mobile':
        model = MobileNet(num_classes=2,fc_num = layer_num).to(device)
    elif m == 'mobilev2':
        model = MobileNetV2(num_classes=2,fc_num = layer_num).to(device)
    elif m == 'dense121':
        model = densenet121_3d(num_classes=2,fc_num = layer_num).to(device)
    else:
        model = densenetmini_3d(num_classes=2,fc_num = layer_num).to(device)

    EPOCHS = epochs_sum
    criterion = nn.CrossEntropyLoss().to(device)
    
    pretrained_path = r'weight.pth'
    batch_size = 256
    pretrained = True
    # load weight
    if pretrained == True:
        pretrained_dict = torch.load(pretrained_path)
        model_dict = model.state_dict()
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
        if len(pretrained_dict) > 1:
            print('sucessful')
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    #multi-GPU
    model = nn.DataParallel(model,device_ids=device_ids)

    # constructs data loaders based on configuration
    resize_size = (64,64,64)
    batch_size = 256
    test_dataset = mission_Dataset(path_list,augmentation = False)
    test_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    # valid step
    start = time.time()
    test_loss_sum,test_result,train_df = test_model(model,test_loader,criterion)
    end = time.time()
    
    # notes that, train loader and valid loader both have one batch!!!
    print('''
    Loss: {:.6f}
    acc: {:.3f}
    auc: {:.3f}
    '''
    .format(test_loss_sum,
    100*test_result[0],
    100*test_result[1],))

    print('**************************************************')
    print(end-start)
    #result_train_prob = pd.read_excel(path_list)
    #result_train_prob['prob'] = train_df['prob']
    #result_train_prob.to_csv(r'prob_save.csv',index = False)
    return

if __name__ == '__main__':
    save_excel_root = r'./table_save/'
    # global settings
    ## GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    device = torch.device('cuda:0')
    device_ids = [0,1]
    torch.backends.cudnn.benchmark = True
    
    ## others
    models_dir = os.path.join('models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    params = [('res10_rpn_wt1','res10_rpn', 128*2, 1e-6, 1e-2, 128, 50)]

    # training
    for name, m, bs, lr, wd, layer_num,epochs in params:
        print('Running {}...'.format(name))
        
        path = r'table/test.csv'

        build_model(name,path,device,device_ids,m,layer_num,epochs)

