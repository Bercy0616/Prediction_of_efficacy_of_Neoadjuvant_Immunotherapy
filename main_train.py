# -*- encoding: utf-8 -*-
'''
@File    :   main_train.py
@Time    :   2022/10/17 18:03:54
@Author  :   Bercy 
@Version :   1.0
@Contact :   hebingxi0616@163.com
'''

import os
import torch
import pandas as pd
import torch.nn as nn
from datasets import NeoadjuvantImmunotherapy_Dataset
from torch.utils.data import Dataset, DataLoader
from utils import get_result_binary
from utils import adjust_learning_rate
from shufflenetv2 import shufflenet_v2_x0_5

def test_model(model,loader,criterion,criterion2,num_class):
    model.eval()
    loss_sum = 0
    total = 0
    output_outputs = torch.Tensor([]).to(device)
    output_pred = torch.Tensor([]).to(device)
    labels = torch.IntTensor([]).to(device) 
    for inputs, label in loader:
        inputs, label = inputs.float().to(device), label.to(device)
        # makes predictions
        with torch.no_grad():
            outputs,outputs2 = model(inputs)
            loss = criterion(outputs, label)
            _, predicted = torch.max(outputs.data, 1)
  
            total += label.size(0)
            loss_sum += loss.item()
            #print(predicted)
            output_outputs = torch.cat((output_outputs, outputs[:,0].float()), 0)
            output_pred = torch.cat((output_pred, predicted), 0)
            labels = torch.cat((labels, label.int()), 0)

    loss_sum = round(loss_sum/total,4)

    results = get_result_binary(output_pred.cpu(), output_outputs.cpu(), labels.cpu())

    return loss_sum,results

def build_model(parameters,mean_std,name,device,device_ids,m,pretrained):
    # reads configuration
    train_table_path = parameters[0]
    valid_table_path = parameters[1]
    lr = parameters[2]
    weight_decay = parameters[3]
    batch_size = parameters[4]
    EPOCHS = 500
    outfolder = 'weight'
    # builds network|criterion|optimizer based on configuration

    model = shufflenet_v2_x0_5(num_classes=2, mode = 0).to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)

    # 读取参数
    if pretrained != False:
        pretrained_dict = torch.load(r'/data1/hbx/pyproject/xfzmyzl/weight/mpr_shufflev205_4_1/net4_005.pth')
        model_dict = model.state_dict()
        # 将pretrained_dict里不属于model_dict的键剔除掉
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
        if len(pretrained_dict) > 1:
            print('加载预训练权重成功')
        # 更新现有的model_dict
        model_dict.update(pretrained_dict)
        # 加载我们真正需要的state_dict
        model.load_state_dict(model_dict)

    # multi-GPU
    model = nn.DataParallel(model,device_ids=device_ids)
    optimizer = nn.DataParallel(optimizer,device_ids=device_ids)

    # constructs data loaders based on configuration
    resize_size = (73,60,56)
    train_dataset = NeoadjuvantImmunotherapy_Dataset(train_table_path,mean_std,image_size = resize_size)
    train_dataset2 = NeoadjuvantImmunotherapy_Dataset(train_table_path,mean_std,image_size = resize_size)
    valid_dataset = NeoadjuvantImmunotherapy_Dataset(valid_table_path,mean_std,image_size = resize_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_loader2 = DataLoader(train_dataset2, batch_size=batch_size)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

    # training
    train = pd.DataFrame()
    valid = pd.DataFrame()
    train_loss,valid_loss = [],[]
    train_acc,valid_acc = [],[]
    train_auc,valid_auc = [],[]

    for epoch in range(1, EPOCHS+1):
        lr = adjust_learning_rate(optimizer.module, epoch, lr, 1e-3)
        # train step
        model.train()
        k = 0
        sum_loss = 0
        length = len(train_loader)
        for inputs, label in train_loader:
            inputs, label  = inputs.float().to(device), label.to(device)
            # makes predictions
            outputs,outputs2 = model(inputs)
            loss = criterion(outputs, label)

            # updates parameters
            optimizer.module.zero_grad()
            loss.backward()
            optimizer.module.step()
            sum_loss += loss.item()
            total = label.size(0)

            print('[epoch:%d, iter:%d] Loss: %.03f'
                          % (epoch, (k + 1 + epoch * length), sum_loss / total))
            k+=1

        # valid step
        train_loss_sum,train_result = test_model(model,train_loader2,criterion,criterion,2)
        valid_loss_sum,valid_result = test_model(model,valid_loader,criterion,criterion,2)

        # add result
        train_loss.append(train_loss_sum)
        train_acc.append(train_result[0])
        train_auc.append(train_result[1])

        valid_loss.append(valid_loss_sum)
        valid_acc.append(valid_result[0])
        valid_auc.append(valid_result[1])

        # saves the best model
        if os.path.exists(os.path.join(outfolder,name)) == False:
            os.mkdir(os.path.join(outfolder,name))
        
        torch.save(model.state_dict(), '%s/%s/net_%03d.pth' % (outfolder,name,epoch))

        # notes that, train loader and valid loader both have one batch!!!
        print('''
        Epoch: {}  Loss: {:.6f}({:.6f})
        acc: {:.3f}({:.3f})
        auc: {:.3f}({:.3f})
        **************************************************'''
        .format(epoch, train_loss_sum,valid_loss_sum, 
        100*train_result[0],100*valid_result[0],
        100*train_result[1],100*valid_result[1]))
        
        
    train['loss'] = train_loss
    train['acc'] = train_acc
    train['auc'] = train_auc

    valid['loss'] = valid_loss
    valid['acc'] = valid_acc
    valid['auc'] = valid_auc

    return train,valid

if __name__ == '__main__':
    #Random
    save_excel_root = r'/result_save/'

    # global settings
    ## GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
    device = torch.device('cuda:0')
    device_ids = [0,1]
    #torch.backends.cudnn.benchmark = True
    
    ## others
    models_dir = os.path.join('models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    configs_dir = 'configs'
    class_aim = 2
    
    mean_std = [,]#Calculate the mean and variance of the training cohort

    paras = [(1e-8,5e-1,86,'shufflev205','shufflev205_'+str(3),True)]
    for lrs,wd,bs,m,name,pretrained in paras:
        print('Running {}...'.format(name))

        #测试集患者信息表
        train_table = r'train.xlsx'
        validation_table = r'valid.xlsx'
            
        pretrained_weight_path = False
        
        parameters = [train_table,
                    validation_table,
                    lrs,
                    wd,
                    bs]

        train,valid = build_model(parameters,mean_std,name,device,device_ids,m,pretrained_weight_path)
        
        result_csv = pd.DataFrame()
        result_csv['train_loss'] = train['loss']
        result_csv['train_acc'] = train['acc']
        result_csv['train_auc'] = train['auc']

        result_csv['valid_loss'] = valid['loss']
        result_csv['valid_acc'] = valid['acc']
        result_csv['valid_auc'] = valid['auc']

        result_csv.to_csv(os.path.join('result',str(name)+r'.csv'),index = False)

