# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 17:15:35 2019

@author: Shashank
"""

# Importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Importing the Dataset
dataset = pd.read_csv('parkinsons.data')
dataset.head()

# Exploring the Dataset
#Getting the Features and Labels
features = dataset.loc[:, dataset.columns != 'status'].values[:, 1:]
labels = dataset.loc[:, 'status'].values
# Getting the count of each label in Labels
print(labels[labels == 1].shape[0], labels[labels == 0].shape[0])

# Normalizing the Features
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler((-1, 1))
x = scaler.fit_transform(features)
y = labels

# Splitting the dataset into Training Set and Test Set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Fitting the XG Boost Classifier to the training set
from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(x_train, y_train)

# Predicting the Test Set Results
y_pred = xgb.predict(x_test)

# Creating the Confusion Matrix and Calculating the Accuracy Score
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
print('Accuracy Score = ', acc * 100)
print(cm)
