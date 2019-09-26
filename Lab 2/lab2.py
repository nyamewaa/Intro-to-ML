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
import torchvision
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

classes=('0','1','2','3','4','5','6','7','8','9')
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
trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=normalize_image)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True)
#%%
num_train = int(len(trainset)*.8)
dev = [trainset[i] for i in range(num_train,len(trainset))]



#%%
testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=normalize_image)



#%% plotting some of the images from the training set. This doesnt work because there is no
#function plot_images

# functions to show an image


def imshow(img):
    img = img / 2 + 0.3081    # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

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
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)
devloader= torch.utils.data.DataLoader(dev, batch_size=batch_size,
                                         shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=True)
model = Model()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
#%% TRAINING MODEL DEFINITION
def train_epoch( model, trainloader, optimizer, epoch):
    model.train() # Set the nn.Module to train mode. 
    total_loss = 0
    correct = 0
    num_samples = len(trainloader.dataset)
    for batch_idx, (x, target) in enumerate(trainloader): #1) get batch
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
def eval_epoch(model, testloader, name):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in testloader:
        output = model(data)
        test_loss += F.cross_entropy(output, target).item() # sum up batch loss
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(testloader.dataset)
    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        name,
        test_loss, 
        correct, 
        len(testloader.dataset),
        100. * correct / len(testloader.dataset)))
    
#%% TRAINING THE MODEL
    
for epoch in range(1, epochs + 1):
    train_epoch(model, trainloader, optimizer, epoch)
    eval_epoch(model,  devloader, "Dev")
    print("---")
    
#%% EXPERIMENT WITH MLP (MULTILAYER PERCEPTRON)
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
        train_epoch(model, trainloader, optimizer, epoch)
        eval_epoch(model,  devloader, "Dev")
        print("---")
    

#%% RUN MODEL ON TEST IMAGES
eval_epoch(model,  testloader, "Test")

#%% EXPERIMENT WITH MLP custom model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer1 = nn.Sequential(    #defines first layer
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2), #con2d is for images. conv2d(input channel, output channel, filter size, eg, 5x5, stride)
            nn.ReLU(), #relu activation. padding is calculated from formula (width out=widthin-Filtersize in W direction + 2Padding)/stride]+1
            nn.MaxPool2d(kernel_size=2, stride=2)) # pooling size=kernel size,  stride=downsampling by a factor 2, padding is zero by default if unspecified
        #from first layer we can calculate size of output images using above formula
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)) #we can calculate to realize that the output size will be 7x7 images with the 64 channels specified
        self.drop_out = nn.Dropout()  #drop out helps to avoid overfitting
        self.fc1 = nn.Linear(7 * 7 * 64, 1000) #first fully connected layer. inout size is the image wxhxchannels and output size is specified as 1000
        self.fc2 = nn.Linear(1000, 10) #ouput from fc1 is the number of nodes in fc2
        
    #the above defines the layers. now we specify how the data flows through the layers
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out=out.reshape(out.size(0),-1) #flattens the data dimensions  7x7x64 into 3164x1
        out = self.drop_out(out)
        out = self.fc1(out)
        out=self.fc2(out)
        return out
        
        
model = Model()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
#%%
for epoch in range(1, epochs + 1):
        train_epoch(model, trainloader, optimizer, epoch)
        eval_epoch(model,  devloader, "Dev")
        print("---")
        
#%% RUN MODEL ON TEST IMAGES
eval_epoch(model,  testloader, "Test")
