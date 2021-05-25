# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 19:50:33 2021

@author:rohika
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

salary_train=pd.read_csv('C:/Users/rohika/OneDrive/Desktop/360digiTMG assignment/Naive bayes/SalaryData_Train.csv')
salary_test=pd.read_csv('C:/Users/rohika/OneDrive/Desktop/360digiTMG assignment/Naive bayes/SalaryData_Test.csv')
salary_train.columns
salary_test.columns
string_columns=['workclass','education','maritalstatus','occupation','relationship','race','sex','native']

from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()
for i in string_columns:
    salary_train[i]=label_encoder.fit_transform(salary_train[i])
    salary_test[i]=label_encoder.fit_transform(salary_test[i])

col_names=list(salary_train.columns)
train_X=salary_train[col_names[0:13]]
train_Y=salary_train[col_names[13]]
test_x=salary_test[col_names[0:13]]
test_y=salary_test[col_names[13]]

######### Naive Bayes ##############
#Multinomial Naive Bayes

from sklearn.naive_bayes import MultinomialNB
Mmodel=MultinomialNB()
train_pred_multi=Mmodel.fit(train_X,train_Y).predict(train_X)
test_pred_multi=Mmodel.fit(train_X,train_Y).predict(test_x)

train_acc_multi=np.mean(train_pred_multi==train_Y)
test_acc_multi=np.mean(test_pred_multi==test_y)
train_acc_multi
test_acc_multi
