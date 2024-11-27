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

def build_model(name,device,device_ids,m,path_list,layer_num,epochs_sum):
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

    lr = path_list[3]
    batch_size = path_list[2]
    weight_decay = path_list[4]
    EPOCHS = epochs_sum

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,weight_decay=weight_decay)
    
    pretrained = False
    pretrained_path = r'weight.pth'

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
    optimizer = nn.DataParallel(optimizer,device_ids=device_ids)

    # constructs data loaders based on configuration
    resize_size = (64,64,64)
    train_dataset = mission_Dataset(path_list[0],augmentation = True)
    train_dataset2 = mission_Dataset(path_list[0])
    valid_dataset = mission_Dataset(path_list[1])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_loader2 = DataLoader(train_dataset2, batch_size=batch_size)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

    # training
    train = pd.DataFrame()
    valid = pd.DataFrame()

    train_loss,valid_loss = [],[]
    train_acc,valid_acc = [],[]
    train_auc,valid_auc = [],[]
    train_sen,valid_sen = [],[]
    train_spe,valid_spe = [],[]

    for epoch in range(1, EPOCHS+1):
        lr = adjust_learning_rate(optimizer.module, epoch,
                                  lr,
                                  weight_decay)

        # train step
        model.train()
        k = 0
        sum_loss = 0
        total = 0
        length = len(train_loader)
        optimizer.zero_grad()
        k_times = 1
        kkk = 0
        for inputs, labels, f1, f2, f3 in train_loader: 
            inputs, labels = inputs.float().to(device), labels.to(device)
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            loss = loss/k_times
            loss.backward()

            optimizer.module.step()       
            optimizer.module.zero_grad()

            sum_loss += loss.item()
            total += labels.size(0)

            if kkk%k_times == 0:
                print('[epoch:%d, iter:%d] Loss: %.03f'
                            % (epoch, (k + 1 + epoch * length), sum_loss / total))
            k+=1

        # valid step
        train_loss_sum,train_result,train_df = test_model(model,train_loader2,criterion)
        valid_loss_sum,valid_result,valid_df = test_model(model,valid_loader,criterion)

        prob_c = 'prob'
        t,r = get_yoden_threshold(train_df,'label',prob_c)
        train_c = get_class(train_df,prob_c,t)
        valid_c = get_class(valid_df,prob_c,t)

        train_df[prob_c+'_ch'] = train_c
        valid_df[prob_c+'_ch'] = valid_c

        sen_train,spe_train = sensitivity_specificity(train_df,'label','prob_ch')
        sen_valid,spe_valid = sensitivity_specificity(valid_df,'label','prob_ch')

        # add result
        train_loss.append(train_loss_sum)
        train_acc.append(train_result[0])
        train_auc.append(train_result[1])
        train_sen.append(sen_train)
        train_spe.append(spe_train)
        
        valid_loss.append(valid_loss_sum)
        valid_acc.append(valid_result[0])
        valid_auc.append(valid_result[1])
        valid_sen.append(sen_valid)
        valid_spe.append(spe_valid)

        # saves the best model
        weight_root = r'weights'
        if os.path.exists(os.path.join(weight_root,name)) == False:
            os.mkdir(os.path.join(weight_root,name))
        
        torch.save(model.state_dict(), '%s/%s/net_%03d.pth' % (weight_root,name,epoch))
        # notes that, train loader and valid loader both have one batch!!!
        print('''
        Epoch: {}  Loss: {:.6f}({:.6f})
        acc: {:.3f}({:.3f})
        auc: {:.3f}({:.3f})
        sen: {:.3f}({:.3f}) 
        spe: {:.3f}({:.3f}) 
        '''

        .format(epoch, train_loss_sum,valid_loss_sum,
        100*train_result[0],100*valid_result[0], 
        100*train_result[1],100*valid_result[1],
        sen_train,sen_valid,spe_train,spe_valid))

        print('**************************************************')

        result_train_prob = pd.read_csv(os.path.join('/data/hebingxi/disk/mission_result',str(name),str(name)+r'_train_prob.csv'))
        result_valid_prob = pd.read_csv(os.path.join('/data/hebingxi/disk/mission_result',str(name),str(name)+r'_valid_prob.csv'))

        result_train_prob[str(epoch)] = train_df['prob']
        result_valid_prob[str(epoch)] = valid_df['prob']

        result_train_prob.to_csv(os.path.join('/data/hebingxi/disk/mission_result',str(name),str(name)+r'_train_prob.csv'),index = False)
        result_valid_prob.to_csv(os.path.join('/data/hebingxi/disk/mission_result',str(name),str(name)+r'_valid_prob.csv'),index = False)

    train['loss'] = train_loss
    train['acc'] = train_acc
    train['auc'] = train_auc
    train['sen'] = train_sen
    train['spe'] = train_spe

    valid['loss'] = valid_loss
    valid['acc'] = valid_acc
    valid['auc'] = valid_auc
    valid['sen'] = valid_sen
    valid['spe'] = valid_spe

    return train,valid

if __name__ == '__main__':
    #Random
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
    for name, m, bs, lr, wd, layer_num,epochs_sum in params:
        print('Running {}...'.format(name))
        
        train_path = r'table/train.csv'
        valid_path = r'table/valid.csv'

        path_list = [train_path,
                    valid_path,
                    bs, 
                    lr,
                    wd]

        result_train_prob = pd.DataFrame()
        result_valid_prob = pd.DataFrame()

        train_ls = pd.read_excel(train_path)
        valid_ls = pd.read_excel(valid_path)

        result_train_prob['pid'] = train_ls['pid']
        result_valid_prob['pid'] = valid_ls['pid']

        result_train_prob['cohort'] = train_ls['cohort']
        result_valid_prob['cohort'] = valid_ls['cohort']

        result_train_prob['label'] = train_ls['label']
        result_valid_prob['label'] = valid_ls['label']
        
        if os.path.exists(os.path.join('/data/hebingxi/disk/mission_result',str(name))) == False:
            os.mkdir(os.path.join('/data/hebingxi/disk/mission_result',str(name)))

        result_train_prob.to_csv(os.path.join('/data/hebingxi/disk/mission_result',str(name),str(name)+r'_train_prob.csv'),index = False)
        result_valid_prob.to_csv(os.path.join('/data/hebingxi/disk/mission_result',str(name),str(name)+r'_valid_prob.csv'),index = False)
        
        del train_ls
        del valid_ls

        train,valid,test = build_model(name,device,device_ids,m,path_list,layer_num,epochs_sum)
        
        result_csv = pd.DataFrame()
        result_csv['train_loss'] = train['loss']
        result_csv['valid_loss'] = valid['loss']

        result_csv['train_acc'] = train['acc']
        result_csv['valid_acc'] = valid['acc']
        
        result_csv['train_auc'] = train['auc']
        result_csv['valid_auc'] = valid['auc']

        result_csv['train_sen'] = train['sen']
        result_csv['valid_sen'] = valid['sen']

        result_csv['train_spe'] = train['spe']
        result_csv['valid_spe'] = valid['spe']

        result_csv.to_csv(os.path.join('results',str(name),str(name)+r'.csv'),index = False)

