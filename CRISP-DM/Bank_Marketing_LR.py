# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 23:41:44 2024

@author: hp
"""


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report



### Load data and remove any rows with "unknown" values
df = pd.read_csv("bank-additional-full.csv", delimiter = ';', header=0)

dataset = df[~df.apply(lambda row: 'unknown' in row.values, axis=1)]

# Rows with NaN values 
dataset.isnull().sum()

###Count target variable
sns.set(style="darkgrid")
plt.figure(figsize=(8, 6))

# Create count plot with count values displayed on the bars
ax = sns.countplot(x='y', data=dataset, palette='viridis')

# Display count values on the bars
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')

plt.show()


# Instantiate LabelEncoder
label_encoder = LabelEncoder()
dataset.job = label_encoder.fit_transform(dataset.job)
dataset.marital = label_encoder.fit_transform(dataset.marital)
dataset.default = label_encoder.fit_transform(dataset.default)
dataset.contact = label_encoder.fit_transform(dataset.contact)
dataset.month = label_encoder.fit_transform(dataset.month)
dataset.day_of_week = label_encoder.fit_transform(dataset.day_of_week)
dataset.education = label_encoder.fit_transform(dataset.education)
dataset.housing = label_encoder.fit_transform(dataset.housing)
dataset.loan = label_encoder.fit_transform(dataset.loan)
dataset.poutcome = label_encoder.fit_transform(dataset.poutcome)
dataset.y = label_encoder.fit_transform(dataset.y)

dataset.head()

###Define dependent and independant variables
y = dataset.values[:,20]
Y = dataset.iloc[:,0:20].values
Y = StandardScaler().fit_transform(Y)

###Perform PCA
pca1 = PCA(n_components = 20)
X = pca1.fit(Y).transform(Y)
print('explained variance ratio (20 components): %s' 
      % str(pca1.explained_variance_ratio_)) 

###Plot Explained varance ratio
plt.figure(figsize=(10,10))
plt.plot(np.cumsum(pca1.explained_variance_ratio_), 'ro-')
plt.grid()

### Choose n_components by analysing the plot
pca = PCA(n_components = 13)
X = pca.fit(Y).transform(Y)
print('explained variance ratio (12 components): %s' 
      % str(pca.explained_variance_ratio_)) 

###Logistic Regression
logreg=LogisticRegression(penalty='l2', max_iter=1000, class_weight='balanced')

###Kfold
kf = KFold(n_splits=20, shuffle = True, random_state = 99)
k=0
sm=0
for train_index, test_index in kf.split(X, y):
 print('k=',k)
 logreg.fit(X[train_index],y[train_index])
 score_test = logreg.score(X[test_index], y[test_index])
 print (score_test)
 if sm < score_test :
  sm=score_test
 train_minindex = train_index
 test_minindex = test_index

 k+=1
 print
logreg.fit(X[train_minindex],y[train_minindex])
pred_xtest =logreg.predict(X[test_minindex])
print('SCORE on test set: ',logreg.score(X[test_minindex],
 y[test_minindex]))
print('SCORE on train set: ',logreg.score(X[train_minindex],
 y[train_minindex]))

scTrain=logreg.score(X[train_minindex],y[train_minindex])
scTest=logreg.score(X[test_minindex],y[test_minindex])
print('Error: ', abs(scTrain-scTest) )


X_test = X[test_minindex]
y_test = y[test_minindex]
y_pred= logreg.predict(X_test)

###Confusion Matrix and Classification Report
print ('\n confusion matrix for the test set :\n',confusion_matrix(y_test,
 y_pred))
print ('\n classification report for the test set :\n',
 classification_report(y_test, y_pred))



