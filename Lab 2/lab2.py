#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 16:55:55 2019
Lab 2: Building a neural network to train and classify hand written digits
@author: nyamewaa
"""
#%% IMPORT PACKAGES
# http://pytorch.org/
from os import path
from wheel.pep425tags import get_abbr_impl, get_impl_ver, get_abi_tag
platform = '{}{}-{}'.format(get_abbr_impl(), get_impl_ver(), get_abi_tag())

accelerator = 'cu80' if path.exists('/opt/bin/nvidia-smi') else 'cpu'

import torch
print(torch.__version__)
print(torch.cuda.is_available())

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

#%%
# Img mean value of .13, and stdv of .31 were computed across entire train set
# in prior work
#for any set of imahes you need to know the mean and stdv of pixel values
#create the normalozation function
normalize_image = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                ])

#%% Importing images
# Dataset is loaded from torchvision
all_train = datasets.MNIST('data', train=True, download=True, transform=normalize_image)

num_train = int(len(all_train)*.8)
train = [all_train[i] for i in range(num_train)]
dev = [all_train[i] for i in range(num_train,len(all_train))]
test = datasets.MNIST('data', train=False, download=True, 
                      transform=normalize_image)
#%%
#plotting some of the images from the training set. This doesnt work because there is no
#function plot_images
all_train = datasets.MNIST('data', train=True, download=True)
# images = [tr[0] for tr in all_train[:9]]
num_examples = 9 
images, labels = [], []
for i in range(num_examples):
  images.append(all_train[i][0])
  labels.append(all_train[i][1])
    
imgplot=plt.imshow(images, labels)
train[0][0].size()
#%% BUILDING A MODEL

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # Learn weights for each pixel and perform linear regression
        self.fc = nn.Linear(28*28, 10)

    def forward(self, x):
        batch_size, num_channels, height, width = x.size()
        # Flatten image
        x = x.view(batch_size, -1)
        # Put it through linear classifier
        return self.fc(x)
#%%# Training settings
batch_size = 64
epochs = 10
lr = .01
momentum = 0.5
train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
dev_loader = torch.utils.data.DataLoader(dev, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True)
model = Model()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
#%% TRAINING MODEL
def train_epoch( model, train_loader, optimizer, epoch):
    model.train() # Set the nn.Module to train mode. 
    total_loss = 0
    correct = 0
    num_samples = len(train_loader.dataset)
    for batch_idx, (x, target) in enumerate(train_loader): #1) get batch
        # Reset gradient data to 0
        optimizer.zero_grad()
        # Get prediction for batch
        output = model(x)
        # 2) Compute loss
        loss = F.cross_entropy(output, target)
        #3) Do backprop
        loss.backward()
        #4) Update model
        optimizer.step()
        
        ## Do book-keeping to track accuracy and avg loss
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        total_loss += loss.detach() # Don't keep computation graph 

    print('Train Epoch: {} \tLoss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            epoch, total_loss / num_samples, 
            correct, 
            num_samples,
            100. * correct / num_samples))
#%% DEFINE EVALUATION LOOP
    def eval_epoch(model, test_loader, name):
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            output = model(data)
        test_loss += F.cross_entropy(output, target).item() # sum up batch loss
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        name,
        test_loss, 
        correct, 
        len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    #%% TRAINING MODEL
    for epoch in range(1, epochs + 1):
        train_epoch(model, train_loader, optimizer, epoch)
    eval_epoch(model,  dev_loader, "Dev")
    print("---")
    
#%% EXPERIMENT WITH MLP
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(28*28, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)
        

    def forward(self, x):
        batch_size, num_channels, height, width = x.size()
        x = x.view(batch_size, -1)
        hidden = F.relu(self.fc1(x))
        hidden = F.relu(self.fc2(hidden))
        logit = self.fc3(hidden)
        return logit
    
model = Model()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

for epoch in range(1, epochs + 1):
    train_epoch(model, train_loader, optimizer, epoch)
    eval_epoch(model,  dev_loader, "Dev")
    print("---")
    

#%% RUN MODEL ON TEST IMAGES
eval_epoch(model,  test_loader, "Test")

    