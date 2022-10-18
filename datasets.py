'''
Author: Bercy
Date: 2022-10-17 16:18:26
LastEditors: Bercy
LastEditTime: 2022-10-18 10:17:16
'''
# -*- encoding: utf-8 -*-
'''
@File    :   datasets.py
@Time    :   2022/10/17 18:03:44
@Author  :   Bercy 
@Version :   1.0
@Contact :   hebingxi0616@163.com
'''

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk

def standard(data,mu,sigma):
    data = (data - mu) / sigma
    return data

def default_loader(path):
    img = sitk.GetArrayFromImage(sitk.ReadImage(path))
    return img

def resize_3D(img,size,method = cv2.INTER_LINEAR):
    img_ori = img.copy()
    for i in range(len(size)):
        size_obj = list(size).copy()
        size_obj[i] = img_ori.shape[i]
        img_new = np.zeros(size_obj)
        for j in range(img_ori.shape[i]):
            if i == 0:
                img_new[j,:,:] = cv2.resize(img_ori[j,:,:].astype('float'), (size[2],size[1]), interpolation=method)
            elif i == 1:
                img_new[:,j,:] = cv2.resize(img_ori[:,j,:].astype('float'), (size[2],size[0]), interpolation=method)
            else:
                img_new[:,:,j] = cv2.resize(img_ori[:,:,j].astype('float'), (size[1],size[0]), interpolation=method)
        img_ori = img_new.copy()
    return img_ori

class NeoadjuvantImmunotherapy_Dataset(Dataset):
    def __init__(self, csvpath, parameter, image_size=(128,128,128), transform=resize_3D, standard=standard, loader=default_loader,num_class = 2, class_name = 'label3'):
        data_excel = pd.read_excel(csvpath)
        imgs = []
        for i in range(data_excel.shape[0]):
            path_part = data_excel.loc[i,'linux_root']+'/'+str(data_excel.loc[i,'patient_id'])+'_'+data_excel.loc[i,'hospital']+'_'+str(data_excel.loc[i,'notes'])+'.nrrd'
            label_part = data_excel.loc[i,class_name]
            imgs.append((path_part,float(label_part)))
        self.imgs = imgs
        self.mu,self.sigma = parameter
        self.transform = transform
        self.standard = standard
        self.loader = loader
        self.image_size = image_size

    def __getitem__(self, index):
        img_path,label = self.imgs[index]
        img = self.loader(img_path)
        img[img>=1024] = 1024
        img[img<-1024] = -1024
        img = self.standard(img,self.mu,self.sigma)
        if self.transform is not None:
            img = self.transform(img,size = self.image_size)
        img = img[np.newaxis,:]
        return img,int(label)

    def __len__(self):
        return len(self.imgs)