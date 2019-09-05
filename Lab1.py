#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 13:39:39 2019
Introduction to machine learning packages Lab 1
@author: nyamewaa
"""

import pickle
import re
from sklearn.linear_model import  LogisticRegression
from sklearn.linear_model import Perceptron 
from sklearn.linear_model  import PassiveAggressiveClassifier
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import warnings
warnings.filterwarnings("ignore")

#%% PREPROCESSING DATA
#Read in data
#conda install -c anaconda wget 
!wget https://raw.githubusercontent.com/yala/MLCodeLab/master/lab1/data/beer/overall_train.p
!wget https://raw.githubusercontent.com/yala/MLCodeLab/master/lab1/data/beer/overall_dev.p
!wget https://raw.githubusercontent.com/yala/MLCodeLab/master/lab1/data/beer/overall_test.p


#%%assign data to variables
train_path = "overall_train.p"
dev_path = "overall_dev.p"
test_path = "overall_test.p"

train_set = pickle.load(open(train_path,'rb'))
dev_set = pickle.load(open(dev_path,'rb'))
test_set = pickle.load(open(test_path, 'rb'))


def preprocess_data(data):
    for indx, sample in enumerate(data):
        text, label = sample['text'],sample['y']
        text = re.sub('\W+', ' ', text).lower().strip()
        data[indx] = text, label
    return data
    
train_set = preprocess_data(train_set)
dev_set = preprocess_data(dev_set)
test_set = preprocess_data(test_set)

print("Num Train: {}".format(len(train_set)))
print("Num Dev: {}".format(len(dev_set)))
print("Num Test: {}".format(len(test_set)))
print("Example Reviews:")
print(train_set[0])
print()
print(train_set[1])

#%% FEATURE ENGINEERING
#Extract tweets and labels into 2 lists
trainText = [t[0] for t in train_set] #text training data
trainY = [t[1] for t in train_set] #Label for training

devText = [t[0] for t in dev_set] #text development data
devY = [t[1] for t in dev_set] #label for development data

testText = [t[0] for t in test_set] #text for test data
testY = [t[1] for t in test_set] #label for test data

# Set that word has to appear at least 5 times to be in vocab
min_df = 5
max_features = 1000
countVec = CountVectorizer(min_df = min_df, max_features = max_features )

# Learn vocabulary from train set
countVec.fit(trainText)

# Transform list of review to matrix of bag-of-word vectors
trainX = countVec.transform(trainText)
devX = countVec.transform(devText)
testX = countVec.transform(testText)

print("Shape of Train X {}\n".format(trainX.shape))
print("Sample of the vocab:\n {}".format(np.random.choice(countVec.get_feature_names(), 20)))

#%% PICK A MODEL AND EXPERIMENT
lr = LogisticRegression()
passAgg    = PassiveAggressiveClassifier()
perceptron = Perceptron()

lr.fit(trainX, trainY)
print("Logistic Regression Train:", lr.score(trainX, trainY))
print("Logistic Regression Dev:", lr.score(devX, devY))
print("--")

passAgg.fit(trainX, trainY) 
print("Passive Aggressive Train:", passAgg.score(trainX, trainY))
print("Passive Aggressive Dev:", passAgg.score(devX, devY))
print("--")

perceptron.fit(trainX, trainY) 
print("Perceptron Train:", perceptron.score(trainX, trainY))
print("Perceptron Dev:", perceptron.score(devX, devY))
print("--")

#%% ANALYSIS AND DEBUGGING
lr = LogisticRegression()
lr.fit(trainX, trainY)
print("Logistic Regression Train:", lr.score(trainX, trainY))
print("Logistic Regression Dev:", lr.score(devX, devY))
print("--")

print("Intepreting LR")
for label in range(3):
    coefs = lr.coef_[label]
    vocab = np.array(countVec.get_feature_names())
    num_features = 10

    top = np.argpartition(coefs, -num_features)[-num_features:]
    # Sort top
    top = top[np.argsort(coefs[top])]
    s_coef = coefs[top]
    scored_vocab = list(zip(vocab[top], s_coef))
    print("Top weighted features for label {}:\n \n {}\n -- \n".format(label, scored_vocab))
    
    ## Find erronous dev errors
devPred = lr.predict(devX)
errors = []
for indx in range(len(devText)):
    if devPred[indx] != devY[indx]:
        error = "Review: \n {} \n Predicted: {} \n Correct: {} \n ---".format(
            devText[indx],
            devPred[indx],
            devY[indx])
        errors.append(error)

np.random.seed(2)
print("Random dev error: \n {} \n \n {} \n \n{}".format(
        np.random.choice(errors,1),
        np.random.choice(errors,1),
        np.random.choice(errors,1))
     )
 #%% REGULARIZATION
 lr = LogisticRegression(C=.5)
lr.fit(trainX, trainY)

print("Logistic Regression Train:", lr.score(trainX, trainY))
print("Logistic Regression Dev:", lr.score(devX, devY))
print("--")

lr = LogisticRegression(C=.1)
lr.fit(trainX, trainY)

print("Logistic Regression Train:", lr.score(trainX, trainY))
print("Logistic Regression Dev:", lr.score(devX, devY))
print("--")

lr = LogisticRegression(C=.01)
lr.fit(trainX, trainY)

print("Logistic Regression Train:", lr.score(trainX, trainY))
print("Logistic Regression Dev:", lr.score(devX, devY))
print("--")