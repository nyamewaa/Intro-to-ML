#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 16:28:21 2019
Introduction to deep learning packages
@author: nyamewaa
"""
#%% IMPORT PACKAGES
# http://pytorch.org/
from os import path
from wheel.pep425tags import get_abbr_impl, get_impl_ver, get_abi_tag
platform = '{}{}-{}'.format(get_abbr_impl(), get_impl_ver(), get_abi_tag())

accelerator = 'cu80' if path.exists('/opt/bin/nvidia-smi') else 'cpu'

!pip install -q http://download.pytorch.org/whl/{accelerator}/torch-0.4.0-{platform}-linux_x86_64.whl torchvision==0.2.0
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

#%% SIMPLE PYTORCH ARITHMETIC
a=torch.rand(3,3)
print(a)
print(a.size())

b=torch.Tensor([[1,2,3],[4,5,6]])
print(b)
print(b.size)

a=a+4
print(a)
d=a*2
print(d)
e=a-d
print(e)

print(a.size(), b.size())

c = torch.matmul(b, a)
print(c.size())

#%% AUTOGRAD
a=torch.ones(1, requires_grad=False)
b=torch.ones(1, requires_grad=False)
x=torch.ones(1, requires_grad=True)
y=a*(x*x)+b
print (y)
y.backward()
print("x.grad={}".format(x.grad))