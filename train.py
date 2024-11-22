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
from model_cnn_3d.densenet import densenet121_3d,densenetmini_3d
from model_cnn_3d.fcn import FCN_methylation
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils import get_result_binary
from utils import adjust_learning_rate,get_yoden_threshold,get_class,sensitivity_specificity


def test_model(model,loader,criterion,name):
    if 'multi' in name:
        model[0].eval()
        model[1].eval()
    elif 'radiomics' in name: 
        model[0].eval()
    elif 'methylation' in name: 
        model[1].eval()

    loss_sum = 0
    total = 0
    output_all = torch.Tensor([]).to(device)
    output_pred_all = torch.Tensor([]).to(device)
    label_all = torch.IntTensor([]).to(device)
    for inputs, labels, f1,f2,f3 in loader:
        inputs, labels = inputs.float().to(device), labels.to(device)
        f1,f2,f3 = f1.float().to(device),f2.float().to(device),f3.float().to(device)
        with torch.no_grad():
            if 'multi' in name:
                outputs_methylation,fea_methylation = model[1](f1,f2,f3)
                outputs = model[0](inputs,fea_methylation)
            elif 'radiomics' in name: 
                outputs = model[0](inputs,inputs)
            elif 'methylation' in name: 
                outputs,fea_methylation = model[1](f1,f2,f3)

            loss = criterion(outputs, labels)
            loss_sum += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            
            output_all = torch.cat((output_all, predicted.float()), 0)
            output_pred_all = torch.cat((output_pred_all,outputs[:,1]),0)
            label_all = torch.cat((label_all, labels.int()), 0)
            
    loss_sum = round(loss_sum/total,4)
    results = get_result_binary(output_all.cpu(), output_pred_all.cpu(),label_all.cpu())
    
    results_df = pd.DataFrame()
    results_df['label'] = label_all.cpu()
    results_df['prob'] = output_pred_all.cpu()
    return loss_sum,results,results_df

def build_model(name,device,device_ids,m,path_list,layer_num,epochs_sum,pretrained):
    outfolder = 'weight'
    # builds network|criterion|optimizer based on configuration
    if m == 'dense121':
        model = densenet121_3d(num_classes=2,fc_num = layer_num, flat = name).to(device)
    else:
        model = densenetmini_3d(num_classes=2,fc_num = layer_num, flat = name).to(device)

    model2 = FCN_methylation().to(device)

    lr = path_list[3]
    batch_size = path_list[2]
    weight_decay = path_list[4]
    EPOCHS = epochs_sum
    criterion = nn.CrossEntropyLoss().to(device)
    if 'multi' in name:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr,weight_decay=weight_decay)
    elif 'radiomics' in name: 
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr,weight_decay=weight_decay)
    elif 'methylation' in name: 
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=lr,weight_decay=weight_decay)
    
    pretrained1 = pretrained[0]
    pretrained2 = pretrained[1]

    # load weights
    if pretrained1 == True:
        pretrained_dict = torch.load(r'weight/multi_omics/net2_001.pth')
        model_dict = model2.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        if len(pretrained_dict) > 1:
            print('Successful_1!')
        model_dict.update(pretrained_dict)
        model2.load_state_dict(model_dict)

    if pretrained2 == True:
        pretrained_dict = torch.load(r'weight/multi_omics/net1_001.pth')
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        if len(pretrained_dict) > 1:
            print('Successful_2!')
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    # multi-GPU
    #model = nn.DataParallel(model,device_ids=device_ids)
    #model2 = nn.DataParallel(model2,device_ids=device_ids)
    #optimizer = nn.DataParallel(optimizer,device_ids=device_ids)
    #if 'multi' in name:
    #    optimizer = nn.DataParallel(optimizer,device_ids=device_ids)
    #elif 'radiomics' in name: 
    #    optimizer = nn.DataParallel(optimizer,device_ids=device_ids)
    #elif 'methylation' in name: 
    #    optimizer2 = nn.DataParallel(optimizer2,device_ids=device_ids)

    # constructs data loaders based on configuration
    train_dataset = mission_Dataset(path_list[0],augmentation = False)
    train_dataset2 = mission_Dataset(path_list[0])
    valid_dataset = mission_Dataset(path_list[1])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_loader2 = DataLoader(train_dataset2, batch_size=batch_size)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

    # training
    train = pd.DataFrame()
    valid = pd.DataFrame()
    train_loss,valid_loss = [],[],[]
    train_acc,valid_acc = [],[],[]
    train_auc,valid_auc = [],[],[]
    train_sen,valid_sen = [],[],[]
    train_spe,valid_spe = [],[],[]

    kkkk = 0

    for epoch in range(1, EPOCHS+1):
        lr = adjust_learning_rate(optimizer, epoch,
                                  path_list[3],
                                  1e-4)

        # train step
        if 'multi' in name:
            model.train()
        elif 'radiomics' in name: 
            model.train()
        elif 'methylation' in name: 
            model2.train()

        k = 0
        sum_loss = 0
        total = 0
        length = len(train_loader)
        if 'multi' in name:
            optimizer.zero_grad()
        elif 'radiomics' in name: 
            optimizer.zero_grad()
        elif 'methylation' in name: 
            optimizer2.zero_grad()

        k_times = 1
        kkk = 0
        for inputs, labels,f1,f2,f3 in train_loader:
            inputs, labels = inputs.float().to(device), labels.to(device)
            f1 = f1.to(device)
            f2 = f2.to(device)
            f3 = f3.to(device)
            # makes predictions
            if 'multi' in name:
                outputs_methylation,fea_methylation = model2(f1,f2,f3)
                outputs = model(inputs,fea_methylation)
            elif 'radiomics' in name: 
                outputs = model(inputs,inputs)
            elif 'methylation' in name: 
                outputs,fea_methylation = model2(f1,f2,f3)
            
            loss = criterion(outputs, labels)
            loss = loss/k_times
            loss.backward()

            if kkk%k_times == 0:
                if 'multi' in name:
                    optimizer.step()
                    optimizer.zero_grad()
                elif 'radiomics' in name: 
                    optimizer.step()
                    optimizer.zero_grad()
                elif 'methylation' in name: 
                    optimizer2.step()
                    optimizer2.zero_grad()
            kkk+=1
            sum_loss += loss.item()
            total += labels.size(0)

            if kkk%k_times == 0:
                print('[epoch:%d, iter:%d] Loss: %.03f'
                            % (epoch, (k + 1 + epoch * length), sum_loss / total))
            k+=1

        # valid step
        train_loss_sum,train_result,train_df = test_model([model,model2],train_loader2,criterion,name)
        valid_loss_sum,valid_result,valid_df = test_model([model,model2],valid_loader,criterion,name)

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
        if os.path.exists(os.path.join(outfolder,name)) == False:
            os.mkdir(os.path.join(outfolder,name))
        
        if 'multi' in name:
            torch.save(model.state_dict(), '%s/%s/net1_%03d.pth' % (outfolder,name,epoch))
            torch.save(model2.state_dict(), '%s/%s/net2_%03d.pth' % (outfolder,name,epoch))
        elif 'radiomics' in name: 
            torch.save(model.state_dict(), '%s/%s/net1_%03d.pth' % (outfolder,name,epoch))
        elif 'methylation' in name: 
            torch.save(model2.state_dict(), '%s/%s/net2_%03d.pth' % (outfolder,name,epoch))

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
        sen_train,sen_valid,
        spe_train,spe_valid))
        
        print('**************************************************')

        result_train_prob = pd.read_csv(os.path.join('result',str(name),str(name)+r'_train_prob.csv'))
        result_valid_prob = pd.read_csv(os.path.join('result',str(name),str(name)+r'_valid_prob.csv'))

        result_train_prob[str(epoch)] = train_df['prob']
        result_valid_prob[str(epoch)] = valid_df['prob']

        result_train_prob.to_csv(os.path.join('result',str(name),str(name)+r'_train_prob.csv'),index = False)
        result_valid_prob.to_csv(os.path.join('result',str(name),str(name)+r'_valid_prob.csv'),index = False)

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
    save_excel_root = r'table/table_save/'
    # global settings
    ## GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device('cuda:0')
    device_ids = [0]
    torch.backends.cudnn.benchmark = True
    
    ## others
    models_dir = os.path.join('models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    params = [('multi_omics','dense', 86, 1e-6, 1e-2, 128, 100,[True,True])]

    # training
    for name, m, bs, lr, wd, layer_num,epochs_sum,pretrained in params:
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

        train_ls = pd.read_csv(train_path)
        valid_ls = pd.read_csv(valid_path)

        result_train_prob['pid'] = train_ls['pid']
        result_valid_prob['pid'] = valid_ls['pid']

        result_train_prob['center'] = train_ls['center']
        result_valid_prob['center'] = valid_ls['center']

        result_train_prob['label'] = train_ls['label']
        result_valid_prob['label'] = valid_ls['label']
        
        if os.path.exists(os.path.join('result',str(name))) == False:
            os.mkdir(os.path.join('result',str(name)))

        result_train_prob.to_csv(os.path.join('result',str(name),str(name)+r'_train_prob.csv'),index = False)
        result_valid_prob.to_csv(os.path.join('result',str(name),str(name)+r'_valid_prob.csv'),index = False)

        del train_ls
        del valid_ls

        train,tester = build_model(name,device,device_ids,m,path_list,layer_num,epochs_sum,pretrained)
        
        result_csv = pd.DataFrame()
        result_csv['train_loss'] = train['loss']
        result_csv['tester_loss'] = tester['loss']

        result_csv['train_acc'] = train['acc']
        result_csv['tester_acc'] = tester['acc']
        
        result_csv['train_auc'] = train['auc']
        result_csv['tester_auc'] = tester['auc']

        result_csv['train_sen'] = train['sen']
        result_csv['tester_sen'] = tester['sen']

        result_csv['train_spe'] = train['spe']
        result_csv['tester_spe'] = tester['spe']

        result_csv.to_csv(os.path.join('result',str(name),str(name)+r'.csv'),index = False)

