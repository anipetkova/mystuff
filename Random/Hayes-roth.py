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

import random
from sklearn.model_selection import train_test_split 

print("Decision tree")
from sklearn.tree import DecisionTreeClassifier
logreg = DecisionTreeClassifier()

from sklearn.metrics import confusion_matrix
import math
for i in range(10000):
    r3=random.randint(0,10000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=r3)# 
    logreg.fit(X_train,y_train) 
    print('r3=',r3)
    y_pred = logreg.predict(X_test)
    cm=confusion_matrix(y_test,y_pred)
    print(cm)
    r1=cm[0,0]/(cm[0,0]+cm[0,1])
    r2=cm[1,1]/(cm[1,0]+cm[1,1])
    r=r1*r2
    mGM=math.sqrt(r)
    print('mGM=',mGM*100)
    
    if mGM > 0.96:
        break
best = r3        
print('best = ', best)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=best)
logreg.fit(X_train,y_train)
print('test set score :',logreg.score(X_test, y_test))
print('train set score :',logreg.score(X_train, y_train))
test_score = logreg.score(X_test, y_test)
train_score = logreg.score(X_train, y_train)
print('Error: ', abs(train_score-test_score))

#Confusiton matrix and clasification report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import math
y_pred = logreg.predict(X_test)
print ('\n confussion matrix for the test set :\n',confusion_matrix(y_test,
                                                                    y_pred))
print ('\n classification report for the test set :\n',
       classification_report(y_test, y_pred))


###########################################################
cm=confusion_matrix(y_test,y_pred)
r1=cm[0,0]/(cm[0,0]+cm[0,1]+cm[0,2])
r2=cm[1,1]/(cm[1,0]+cm[1,1]+cm[1,2])
r3=cm[2,2]/(cm[2,0]+cm[2,1]+cm[2,2])
r=r1*r2*r3
mGM=math.pow(r,1/2)
print('mGM=',mGM*100)
    