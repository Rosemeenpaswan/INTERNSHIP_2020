# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 19:05:42 2020

@author: Rosemeen paswan
"""

import numpy as np
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
#from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


data=pd.read_csv(r"C:\Users\Rosemeen paswan\Desktop\INTERN_2020\Max_2017.csv")
labels = np.array(data['ANNUAL'])
features= data.drop(['YEAR','ANNUAL','JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'],axis=1)
#features = data.drop('ANNUAL', axis = 1)
feature_list = list(features.columns)
features = np.array(features)
#print(features)
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf.fit(train_features,train_labels)
predictions = rf.predict(test_features)
errors = abs(predictions - test_labels)

#print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
mean_abs_error = 100 * (errors / test_labels)
# Calculate and display accuracy
accuracy = 100 - np.mean(mean_abs_error)
#print('Accuracy:', round(accuracy, 2), '%.')
fn=features
cn=labels
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=240)
tree.plot_tree(rf.estimators_[0],
               feature_names = fn, 
               class_names=cn,
               filled = True);
plt.show()
#plt.savefig('rf_individualtree.png')

plt.plot(predictions, 'ro', label = 'prediction')
plt.plot(test_labels, '-b', label = 'actual')
plt.xticks(rotation = '60');
plt.xlabel('year'); plt.ylabel('Maximum Rainfall (cm)'); plt.title('Actual and Predicted Values'); 
plt.legend()
               
               
             
