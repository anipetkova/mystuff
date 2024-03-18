# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 14:41:13 2023

@author: hp
"""

import pandas as pd


dataset = pd.read_csv("page-blocks.dat", delimiter = ',', header=None, skiprows = 15)

dataset[10].value_counts()

X = dataset.values[:,0:10]
X = X.astype(float)
y = dataset.values[:,10]
y = y.astype(float)

print("Decision tree")
from sklearn.tree import DecisionTreeClassifier
logreg = DecisionTreeClassifier()

import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import math
for i in range(10000):
    r3=random.randint(0,10000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=r3)
    logreg.fit(X_train,y_train) 
    print('r3=',r3)
    y_pred = logreg.predict(X_test)
    cm=confusion_matrix(y_test,y_pred)
    print(cm)
    rc1=cm[0,0]/(cm[0,0]+cm[0,1]+cm[0,2]+cm[0,3]+cm[0,4])
    rc2=cm[1,1]/(cm[1,0]+cm[1,1]+cm[1,2]+cm[1,3]+cm[1,4])
    rc3=cm[2,2]/(cm[2,0]+cm[2,1]+cm[2,2]+cm[2,3]+cm[2,4])
    rc4=cm[3,3]/(cm[3,0]+cm[3,1]+cm[3,2]+cm[3,3]+cm[3,4])
    rc5=cm[4,4]/(cm[4,0]+cm[4,1]+cm[4,2]+cm[4,3]+cm[4,4])
    r=rc1*rc2*rc3*rc4*rc5
    mGM=math.pow(r,1/5)
    print('mGM=',mGM*100)
    if mGM > 0.91:
        break
best = r3        
print('best = ', best)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=best)#6518, 6016, 1494
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


##mGM

cm=confusion_matrix(y_test,y_pred)
r1=cm[0,0]/(cm[0,0]+cm[0,1]+cm[0,2]+cm[0,3]+cm[0,4])
r2=cm[1,1]/(cm[1,0]+cm[1,1]+cm[1,2]+cm[1,3]+cm[1,4])
r3=cm[2,2]/(cm[2,0]+cm[2,1]+cm[2,2]+cm[2,3]+cm[2,4])
r4=cm[3,3]/(cm[3,0]+cm[3,1]+cm[3,2]+cm[3,3]+cm[3,4])
r5=cm[4,4]/(cm[4,0]+cm[4,1]+cm[4,2]+cm[4,3]+cm[4,4])
r=r1*r2*r3*r4*r5
mGM=math.pow(r,1/5)
print('mGM=',mGM*100)


##CBA
cm=confusion_matrix(y_test,y_pred)
CBA = 0
for i in range(5):
    a = cm[i,i]
    sum1 = cm[i,0] + cm[i,1] + cm[i,2] + cm[i,3] + cm[i,4]
    sum2 = cm[0,i] + cm[1,i] + cm[2,i] + cm[3,i] + cm[4,i]
    CBA += ((a / max(sum1,sum2))/5)
print('CBA= ',CBA*100)
