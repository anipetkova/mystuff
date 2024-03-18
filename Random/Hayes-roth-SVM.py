# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 13:20:34 2023

@author: hp
"""

import pandas as pd

dataset = pd.read_csv("hayes-roth.dat", delimiter = ', ', header=None, skiprows = 9)

dataset[4].value_counts()

X = dataset.values[:,0:4]
X = X.astype(float)
y = dataset.values[:,4]
y = y.astype(float)

print("SVM")
from sklearn.svm import SVC
logreg = SVC( C=1000,kernel='rbf', gamma='auto',class_weight='balanced' )

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7693)
logreg.fit(X_train,y_train)

print('test set score :',logreg.score(X_test, y_test))
print('train set score :',logreg.score(X_train, y_train))
test_score = logreg.score(X_test, y_test)
train_score = logreg.score(X_train, y_train)
print('Error: ', abs(train_score-test_score))

#Confusiton matrix and clasification report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
y_pred = logreg.predict(X_test)
print ('\n confussion matrix for the test set :\n',confusion_matrix(y_test,
                                                                    y_pred))
print ('\n classification report for the test set :\n',
       classification_report(y_test, y_pred))

##mGM
import math
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
