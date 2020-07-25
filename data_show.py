# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 20:41:02 2020

@author: Rosemeen paswan
"""

import pandas as pd
import numpy as np
#import math
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
#from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
import matplotlib.pyplot as plt



#DATA PREPROCESSING
data=pd.read_csv(r"C:\Users\Rosemeen paswan\Desktop\INTERN_2020\Max_2017.csv")
print("Data heads:")
print(data.head())
print("Null values in the dataset before preprocessing:")
print(data.isnull().sum())
print("Filling null values with mean of that particular column")
data=data.fillna(np.mean(data))
print("Mean of data:")
print(np.mean(data))
print("Null values in the dataset after preprocessing:")
print(data.isnull().sum())
data = data.replace('NA', data) 
data.to_csv('FINALDATA_2.csv') 
print("\n\nShape: ",data.shape)

print("Info:")
print(data.info())



#print("Co-Variance =",data.cov())
#print("Co-Relation =",data.corr())

#corr_cols=data.corr()['ANNUAL'].sort_values()[::-1]
#print("Index of correlation columns:",corr_cols.index)
#
print("Scatter plot of Year and january attributes")
plt.subplots(figsize=(12,5))

plt.plot(data.YEAR,data.FEB,label='FEB')
plt.plot(data.YEAR,data.MAR,'.-y',label='MAR')
#plt.plot(data.YEAR,data.APR,'.-g',label='APR')
#plt.plot(data.YEAR,data.MAY,'.-b',label='MAY')
#plt.plot(data.YEAR,data.JUN,'.-r',label='JUN')
#plt.plot(data.YEAR,data.JUL,'.-m',label='JUL')
#plt.plot(data.YEAR,data.AUG,'.-c',label='AUG')
#plt.plot(data.YEAR,data.SEP,'.-k',label='SEP')
#plt.plot(data.YEAR,data.OCT,'.-g',label='OCT')
#plt.plot(data.YEAR,data.NOV,'.-y',label='NOV')
#plt.plot(data.YEAR,data.DEC,'.-k',label='DEC')
#plt.title("Rainfall of months in various years")
#
plt.legend()
#plt.savefig('MONTHS')


#print("Histograms showing the data from attributes (JANUARY to DECEMBER) of the years 1901-2015:")
#data['JAN'].hist(bins=10, label = 'JAN')
#data['FEB'].hist(bins=20, label = 'FEB')
#data['MAR'].hist(bins=20, label = 'MAR')
#data['APR'].hist(bins=20, label = 'APR')
#data['MAY'].hist(bins=20, label = 'MAY')
#data['JUN'].hist(bins=20, label = 'JUN')
#data['JUL'].hist(bins=20, label = 'JUL')
#data['AUG'].hist(bins=20, label = 'AUG')
#data['SEP'].hist(bins=20, label = 'SEP')
#data['OCT'].hist(bins=20, label = 'OCT')
#data['NOV'].hist(bins=20, label = 'NOV')
#data['DEC'].hist(bins=20, label = 'DEC')
#plt.xlabel("year")
#plt.ylabel("rainfall in various momths")
#plt.legend()

#print("Histogram showing the annual rainfall of the all states:")
#data['ANNUAL'].hist(bins=20)


#DATA_VISUALISATION
d2=data.drop(['YEAR','ANNUAL','JAN-FEB','MAR-MAY','JUN-SEP','OCT-DEC'],axis=1)
k=((d2.head().sum()))
month=list(d2.head())
print("Months are: ",month)
print(k)
s=0
for i in d2.sum():
 s=s+i
print("Total recorded rainfall in these 12 months",s)
probability=list(k/s)
print(probability)
max_rainfall=max(probability)
for i in range(len(month)):
    if probability[i]==max_rainfall:
        print("Maximum Rainfall will be in the month of",month[i])


min_rainfall=min(probability)
for i in range(len(month)):
    if probability[i]==min_rainfall:
        print("Minimum Rainfall will be in the month of",month[i])
        
        

labels = np.array(data['ANNUAL'])
#features = data.drop('ANNUAL', axis = 1)
features= data.drop(['YEAR','ANNUAL','JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'],axis=1)
print(features)
print(labels)
# Saving feature names for later use
feature_list = list(features.columns)
# Convert to numpy array
features = np.array(features)


# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)



#training the model
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf.fit(train_features,train_labels)




# Use the forest's predict method on the test data
predictions = rf.predict(test_features)
# Calculate the absolute errors
errors = abs(predictions - test_labels)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')


# Calculate mean absolute percentage error (MAPE)
mean_abs_error = 100 * (errors / test_labels)
# Calculate and display accuracy
accuracy = 100 - np.mean(mean_abs_error)
print('Accuracy:', round(accuracy, 2), '%.')




#fig.savefig('rf_individualtree.png')

