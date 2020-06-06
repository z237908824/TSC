# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 12:59:26 2020

@author: z2379
"""

import torch.utils.data as Data
import pandas as pd
import numpy as np
import os
import torchvision.transforms as transforms
import time
import re
from torch import nn
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
import random
from tqdm import tqdm
import math
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
from tscModel.resNet import ResNet18

DEVICE = torch.device("cuda")  # 让torch判断是否使用GPU


# Data_set = pd.read_csv('new1.csv',header=None)
# wwww = np.asarray(Data_set.loc[1:2].values,'float32')
class MyDataset(Data.Dataset):

    def __init__(self, dataset, num_classes):
        self.dataset = dataset
        self.num_classes = num_classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        temp = np.asarray(self.dataset.loc[index].values, 'float32')
        # print(temp.shape)
        data_FCN = temp[1:-1]
        Label = temp[0]
        label = np.array(np.zeros(self.num_classes))
        label[int(Label) - 1] = 1
        # label[0]=float(index_key[2])

        return data_FCN, label


def train(model, device, dataiter, optimizer, train_set):  # 训练函数
    model.train()  # 将model设定为训练模式

    for iteration in range(math.ceil(len(train_set) / batch_size)):
        data, label = next(dataiter)
        # print(data.shape)
        label = label.clone().detach().float().to(device)
        data = data.clone().detach().float().to(device)

        output = model(data)
        # print(output.shape)
        loss_function = nn.MSELoss()
        loss = loss_function(output, label)

        optimizer.zero_grad()  # 清除旧的梯度信息
        loss.backward()  # 针对损失函数的后向传播
        optimizer.step()  # 反向传播后的梯度下降
        time_end = time.time()
        # print('训练时间：',time_end-time_start)

        time_start = time.time()
        # corrent=torch.eq(torch.argmax(output, dim=1),torch.argmax(a_label, dim=1))
        # acc = torch.mean(corrent.float())
        time_end = time.time()
        # print('善后时间：',time_end-time_start)
        # print('准确率为',acc)
        # print(loss)

    # print("1个EPOCH结束，loss=",loss)


def test(model, device, dataiter, test_set):  # 训练函数
    model.eval()  # 将model设定为训练模式
    acc = 0
    Alpha = []
    # aaa=0
    for iteration in tqdm(range(math.ceil(len(test_set) / batch_size))):
        data, label = next(dataiter)
        label = label.clone().detach().float().to(device)

        output = model(data)
        loss_function = nn.MSELoss()
        loss = loss_function(output, label)

        corrent = torch.eq(torch.argmax(output, dim=1), torch.argmax(label, dim=1))
        acc += torch.mean(corrent.float())
        # aaa+=1

    print("1个EPOCH结束，acc=", acc.cpu().numpy() / (iteration + 1))
    return acc.cpu().numpy() / (iteration + 1)


batch_size = 10
Num_workers = 0
epoch = 1
root = 'F:\\Dataset\\Univariate\\'


def themain(mission, num_classes):
    train_set = pd.read_csv(root + mission + '\\' + mission + '_TRAIN.csv', header=None)
    test_set = pd.read_csv(root + mission + '\\' + mission + '_TEST.csv', header=None)
    train_dataset = MyDataset(train_set, num_classes)
    Traindataloader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Num_workers)

    test_dataset = MyDataset(test_set, num_classes)
    Testdataloader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Num_workers)

    model = ResNet18(num_classes, DEVICE).to(DEVICE)

    lr = 0.002
    for i in range(epoch):
        Traindataiter = iter(Traindataloader)
        Testdataiter = iter(Testdataloader)
        optimizer = optim.Adam(model.parameters(), lr=lr * 0.9)  # 优化器
        train(model, DEVICE, Traindataiter, optimizer, train_set)
    acc = test(model, DEVICE, Testdataiter, test_set)
    return acc


A = pd.read_csv('F:\\Dataset\\Univariate\\dataset1.csv', header=None)
accs = 128 * [0]
for index, row in A.iterrows():
    print(row[1], row[7])  # 输出每一行
    accs[index] = themain(row[1], row[7])
    torch.cuda.empty_cache()
    A['8'] = pd.Series(accs)
    A.to_csv('F:\\Dataset\\Univariate\\Result1.csv', index=0)
