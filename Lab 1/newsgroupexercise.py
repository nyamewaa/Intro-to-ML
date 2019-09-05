#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 11:19:47 2019
news group exercise

@author: nyamewaa
"""

import pickle
import re
from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.linear_model  import PassiveAggressiveClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import warnings
warnings.filterwarnings("ignore")

#%% READ IN NEWSGROUPS FROM DIFFERENT PLACES
categories=['alt.atheism',
 'comp.graphics',
 'comp.os.ms-windows.misc',
 'comp.sys.ibm.pc.hardware',
 'comp.sys.mac.hardware',
 'comp.windows.x',
 'misc.forsale',
 'rec.autos',
 'rec.motorcycles',
 'rec.sport.baseball',
 'rec.sport.hockey',
 'sci.crypt',
 'sci.electronics',
 'sci.med',
 'sci.space',
 'soc.religion.christian',
 'talk.politics.guns',
 'talk.politics.mideast',
 'talk.politics.misc',
 'talk.religion.misc']

#%% PREPROCESS DATA
#Strip data of non alpha numerics, and make everything lowercase
 full_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), categories=categories)
 test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), categories=categories)
def preprocess_data(data):
    processed_data = []
    for indx, sample in enumerate(data['data']):
        text, label = sample, data['target'][indx]
        label_name = data['target_names'][label]
        text = re.sub('\W+', ' ', text).lower().strip()
        processed_data.append( (text, label, label_name) )
    return processed_data


full_train_set = preprocess_data(full_train)
train_set = full_train_set[:-5000]
dev_set = full_train_set[-5000:]
test_set = preprocess_data(test)

print("Num Train: {}".format(len(train_set)))
print("Num Dev: {}".format(len(dev_set)))
print("Num Test: {}".format(len(test_set)))
print("Example Documents:")
print(train_set[0])
print()
print(train_set[1])

#%% FEATURE ENGINEERING
#Extract tweets and labels into 2 lists
trainText = [t[0] for t in train_set]
trainY = [t[1] for t in train_set]

devText = [t[0] for t in dev_set]
devY = [t[1] for t in dev_set]


testText = [t[0] for t in test_set]
testY = [t[1] for t in test_set]

#initialize counter vectorizer
min_df = 5
max_features = 1000
ngram_range = (1,5) #if not specified its (1,1)
countvec=CountVectorizer(min_df = min_df, ngram_range = ngram_range, max_features=max_features)
# Learn vocabulary from train set
countVec.fit(trainText)

# Transform list of review to matrix of bag-of-word vectors
trainX = countVec.transform(trainText)
devX = countVec.transform(devText)
testX = countVec.transform(testText)

print("Shape of Train X {}\n".format(trainX.shape))
print("Sample of the vocab:\n {}".format(np.random.choice(countVec.get_feature_names(), 20)))
#%% PICK A MODEL AND EXPERIMENT
lr = LogisticRegression(C=0.1)
passAgg    = PassiveAggressiveClassifier(C=0.1)
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

#%%ANALYSIS AND DEBUGGING
r = LogisticRegression()
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

np.random.seed(2) #what is this trying to do???
print("Random dev error: \n {} \n \n {} \n \n{}".format(
        np.random.choice(errors,1),
        np.random.choice(errors,1),
        np.random.choice(errors,1))
     )
        
#%% Step 5: Take best model, and report results on Test
lr = LogisticRegression(C=1)
lr.fit(trainX, trainY)
print("Logistic Regression Train:", lr.score(trainX, trainY))
print("Logistic Regression Dev:", lr.score(devX, devY))
print("--")
        
lr = LogisticRegression(C=0.5)
lr.fit(trainX, trainY)
print("Logistic Regression Train:", lr.score(trainX, trainY))
print("Logistic Regression Dev:", lr.score(devX, devY))
print("--")

lr = LogisticRegression(C=0.1)
lr.fit(trainX, trainY)
print("Logistic Regression Train:", lr.score(trainX, trainY))
print("Logistic Regression Dev:", lr.score(devX, devY))
print("--")

lr = LogisticRegression(C=0.01)
lr.fit(trainX, trainY)
print("Logistic Regression Train:", lr.score(trainX, trainY))
print("Logistic Regression Dev:", lr.score(devX, devY))
print("--")

lr = LogisticRegression(C=0.001)
lr.fit(trainX, trainY)
print("Logistic Regression Train:", lr.score(trainX, trainY))
print("Logistic Regression Dev:", lr.score(devX, devY))
print("--")