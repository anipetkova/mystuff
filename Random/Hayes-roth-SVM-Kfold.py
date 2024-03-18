# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 14:18:31 2023

@author: hp
"""

import pandas as pd
import math

dataset = pd.read_csv("hayes-roth.dat", delimiter = ', ', header=None, skiprows = 9)

dataset[4].value_counts()

X = dataset.values[:,0:4]
X = X.astype(float)
y = dataset.values[:,4]
y = y.astype(float)

print("SVM")
from sklearn.svm import SVC
logreg = SVC( C=10000,kernel='rbf', gamma='auto',class_weight='balanced')

from sklearn.model_selection import KFold
kf = KFold(n_splits=10, shuffle = True, random_state = 22) 
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
from sklearn.metrics import confusion_matrix
print ('\n confusion matrix for the test set :\n',confusion_matrix(y_test,
 y_pred))
from sklearn.metrics import classification_report
print ('\n classification report for the test set :\n',
 classification_report(y_test, y_pred)) 

##mGM
cm=confusion_matrix(y_test,y_pred)
r1=cm[0,0]/(cm[0,0]+cm[0,1]+cm[0,2])
r2=cm[1,1]/(cm[1,0]+cm[1,1]+cm[1,2])
r3=cm[2,2]/(cm[2,0]+cm[2,1]+cm[2,2])
r=r1*r2*r3
mGM=math.pow(r,1/3)
print('mGM=',mGM*100)


##CBA
cm=confusion_matrix(y_test,y_pred)
CBA = 0
for i in range(3):
    a = cm[i,i]
    sum1 = cm[i,0] + cm[i,1] + cm[i,2]
    sum2 = cm[0,i] + cm[1,i] + cm[2,i]
    CBA += ((a / max(sum1,sum2))/3)
print('CBA= ',CBA*100)
