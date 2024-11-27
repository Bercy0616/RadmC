# ------------------------------------------------------------------------------
# --coding='utf-8'--
# Written by Bercy
# ------------------------------------------------------------------------------

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
#from Augmentor3D import Transforms,Transforms
import os
import random

def standard(data,mu,sigma):
    data = (data - mu) / sigma
    return data

def standard_single(data):
    mu = np.mean(data)
    sigma = np.std(data)
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

class mission_Dataset(Dataset):
    def __init__(self, csvpath, parameter=[-338.87,115.65], image_size=False, transform=resize_3D, augmentation = False, standard=standard, loader=default_loader, resize = True, window = 2):
        data_excel = pd.read_excel(csvpath)
        imgs = []
        for i in range(data_excel.shape[0]):
            path_part = data_excel.loc[i,'linux_path']    
            feature1 = data_excel.loc[i,'5mC_score']
            feature2 = data_excel.loc[i,'5hmC_score']
            feature3 = data_excel.loc[i,'ratio_score']
            label_part = data_excel.loc[i,'label']
            imgs.append((path_part,float(label_part),float(feature1),float(feature2),float(feature3)))
        self.window = window
        self.imgs = imgs
        self.mu,self.sigma = parameter
        self.transform = transform
        self.standard = standard
        self.loader = loader
        self.image_size = image_size
        self.augmentation = augmentation
        self.resize = resize

    def __getitem__(self, index):
        img_path,label,f1,f2,f3 = self.imgs[index]
        img = self.loader(img_path)
        if self.resize:
            img = resize_3D(img,[64,64,64])
        #f1 = torch.Tensor(f1)
        #f2 = torch.Tensor(f2)
        #f3 = torch.Tensor(f3)

        if self.window == 2:
            img[img>=1600] = 1600
            img[img<-400] = -400
        elif self.window == 3:
            img[img>=1000] = 1000
            img[img<-700] = -700
        else:
            img[img>=1024] = 1024
            img[img<-1024] = -1024

        img = self.standard(img,self.mu,self.sigma)
        img = img[np.newaxis,:]
        if self.augmentation == True:
            if random.choice([True, False]):
                img = self.flip3d(img)
            img = self.rotate3d(img, max_degree=3)
            img = self.translate3d(img, max_len=3)
        return img,int(label),f1,f2,f3

    def __len__(self):
        return len(self.imgs)
    
    def rotate3d(self, image, max_degree):
        channel, depth, height, width = image.shape
        angle = random.uniform(-max_degree, max_degree)
        rotate_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        rotated_channel = []
        for i in range(channel):
            rotated_channel.append(
                np.stack([cv2.warpAffine(image[i][j], rotate_matrix, (width, height), flags=cv2.INTER_CUBIC)
                          for j in range(depth)]))
        rotated_image = np.stack(rotated_channel)
        return rotated_image

    def translate3d(self, image, max_len):
        channel, depth, height, width = image.shape
        x, y = random.uniform(-max_len, max_len), random.uniform(-max_len, max_len)
        translate_matrix = np.float32([[1, 0, x], [0, 1, y]])
        translated_channel = []
        for i in range(channel):
            translated_channel.append(
                np.stack([cv2.warpAffine(image[i][j], translate_matrix, (width, height), flags=cv2.INTER_CUBIC)
                          for j in range(depth)]))
        translated_image = np.stack(translated_channel)
        return translated_image

    def flip3d(self, image):
        channel, depth, height, width = image.shape
        flip_code = random.choice([1, 0, -1])
        flipped_channel = []
        for i in range(channel):
            flipped_channel.append(
                np.stack([cv2.flip(image[i][j], flip_code) for j in range(depth)]))
        flipped_image = np.stack(flipped_channel)
        return flipped_image