# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 19:53:52 2021

@author: rohika
"""
import pandas as pd
import numpy as np
car= pd.read_csv("C:\\Users\\rohika\\OneDrive\\Desktop\\360digiTMG assignment\\Naive bayes\\NB_Car_Ad.csv")
car =car.iloc[:,[3,4]]


# splitting data into train and test data sets 
from sklearn.model_selection import train_test_split

car_train, car_test = train_test_split(car, test_size = 0.2)
 
# Preparing a naive bayes model on training data set 

from sklearn.naive_bayes import MultinomialNB as MB

# Multinomial Naive Bayes
classifier_mb = MB()
classifier_mb.fit(car_train, car_train.Purchased)

# Evaluation on Test Data
test_pred_m = classifier_mb.predict(car_test)
accuracy_test_m = np.mean(test_pred_m == car_test.Purchased)
accuracy_test_m

from sklearn.metrics import accuracy_score
accuracy_score(test_pred_m, car_test.Purchased) 

pd.crosstab(test_pred_m, car_test.Purchased)

# Training Data accuracy
train_pred_m = classifier_mb.predict(car_train)
accuracy_train_m = np.mean(train_pred_m == car_train.Purchased)
accuracy_train_m

# Multinomial Naive Bayes changing default alpha for laplace smoothing
# if alpha = 0 then no smoothing is applied and the default alpha parameter is 1
# the smoothing process mainly solves the emergence of zero probability problem in the dataset.

classifier_mb_lap = MB(alpha = 3)
classifier_mb_lap.fit(car_train, car_train.Purchased)

# Evaluation on Test Data after applying laplace
test_pred_lap = classifier_mb_lap.predict(car_test)
accuracy_test_lap = np.mean(test_pred_lap == car_test.Purchased)
accuracy_test_lap

from sklearn.metrics import accuracy_score
accuracy_score(test_pred_lap, car_test.Purchased) 

pd.crosstab(test_pred_lap, car_test.Purchased)

# Training Data accuracy
train_pred_lap = classifier_mb_lap.predict(car_train)
accuracy_train_lap = np.mean(train_pred_lap == car_train.Purchased)
accuracy_train_lap

