import numpy as np
import pandas as pd
import sklearn
import os
import math
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt
#####################
#Auther: Ruojun

def load_file_save(file_name):
    if not os.path.isfile('train.txt'):
        ####################
        #load file
        file = open(file_name)
        file_data = file.readlines()
        features = file_data[28:53]
        data = [line.strip().split(',') for line in file_data[145:545]]
        data.remove(data[369])
        dataframe = sklearn.utils.shuffle(pd.DataFrame(data))
        dataframe.drop(columns=25)
        ########################
        #drop missing data
        debug1 = dataframe == '?'
        drop_result = dataframe[debug1.sum(1) == 0]
        drop_result[24] = drop_result[24].map({'notckd': 0, 'ckd': 1})
        drop_result[5] = drop_result[5].map({'normal': 0, 'abnormal': 1})
        drop_result[6] = drop_result[6].map({'normal': 0, 'abnormal': 1})
        drop_result[7] = drop_result[7].map({'present': 0, 'notpresent': 1})
        drop_result[8] = drop_result[8].map({'present': 0, 'notpresent': 1})
        drop_result[18] = drop_result[18].map({'no': 0, 'yes': 1})
        drop_result[19] = drop_result[19].map({'no': 0, 'yes': 1})
        drop_result[20] = drop_result[20].map({'no': 0, 'yes': 1})
        drop_result[21] = drop_result[21].map({'poor': 0, 'good': 1})
        drop_result[22] = drop_result[22].map({'no': 0, 'yes': 1})
        drop_result[23] = drop_result[23].map({'no': 0, 'yes': 1})
        ########################
        #Split train and test
        train_data = drop_result[0:int(len(drop_result)*0.8)]
        test_data = drop_result[int(len(drop_result)*0.8):]
        train_data.to_csv('train.txt',header=False,index=False)
        test_data.to_csv('test.txt',header=False,index=False)
    train_x = np.asarray(pd.read_csv('train.txt', header=None))[:, 0:23]
    test_x = np.asarray(pd.read_csv('test.txt', header=None))[:, 0:23]
    train_y = np.asarray(pd.read_csv('train.txt', header=None))[:, 24]
    test_y = np.asarray(pd.read_csv('test.txt', header=None))[:, 24]
    return train_x,train_y,test_x,test_y

def f_measure(y_hat,y):
    TP = 0
    FP = 0
    FN = 0
    for i in range(np.size(y)):
        if np.allclose(y[i], 0., atol=1e-3)&np.allclose(y_hat[i], 0.,atol=1e-3):TP +=1
        elif np.allclose(y[i], 0., atol=1e-3) & np.allclose(y_hat[i], 1., atol=1e-3): FP += 1
        elif np.allclose(y[i], 1., atol=1e-3) & np.allclose(y_hat[i], 0., atol=1e-3): FN += 1
    PRE = TP/(TP+FP)
    REC = TP/(TP+FN)
    f_measure = (2*PRE*REC)/(PRE+REC)
    return f_measure

file = 'chronic_kidney_disease_full.arff'
train_x,train_y,test_x,test_y = load_file_save(file)
from sklearn.svm import SVC
clf = SVC(gamma='auto')
clf.fit(train_x, train_y)
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
test_predict = clf.predict(test_x).reshape(np.shape(test_y))
print(test_predict)
print(test_y)
f_measure(test_predict,test_y)