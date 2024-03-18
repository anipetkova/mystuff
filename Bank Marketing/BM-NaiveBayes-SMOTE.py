# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 23:41:44 2024

@author: hp
"""

import time
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

#Record the start time
start_time = time.time()

#Load data and remove any rows with "unknown" values
dataset = pd.read_csv("bank-additional-full.csv", delimiter = ';', header=0)

dataset.replace('unknown', np.nan, inplace=True)
dataset.dropna(inplace=True)

#Rows with NaN values 
dataset.isnull().sum()

#Count target variable
sns.set(style="darkgrid")
plt.figure(figsize=(8, 6))

#Create count plot with count values displayed on the bars
ax = sns.countplot(x='y', data=dataset, palette='viridis')

#Display count values on the bars
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')

plt.show()


#Instantiate LabelEncoder
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

#Define dependent and independant variables
y = dataset.values[:,20]
X = dataset.iloc[:,0:20].values


from imblearn.over_sampling import SMOTE
#Apply SMOTE to balance the data
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

#Convert back to DataFrame if needed
resampled_df = pd.DataFrame(data=X_resampled, columns=dataset.columns[:-1])
resampled_df['y'] = y_resampled


#Count target variable
sns.set(style="darkgrid")
plt.figure(figsize=(8, 6))

#Create count plot with count values displayed on the bars
ax = sns.countplot(x='y', data=resampled_df, palette='viridis')

#Display count values on the bars
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')

plt.show()

#Naive Bayes
from sklearn.naive_bayes import GaussianNB
logreg = GaussianNB()

#Kfold
kf = KFold(n_splits=20, shuffle = True, random_state =70)
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

#AUC ROC
#Fit the model on the entire resampled dataset
logreg.fit(X_resampled, y_resampled)

#Predict probabilities for the positive class
y_probs = logreg.predict_proba(X_test)[:, 1]

#Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

#Calculate AUC
auc_score = roc_auc_score(y_test, y_probs)
print('AUC Score:', auc_score)

#Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc_score)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")

#Display the AUC score on the plot
plt.text(0.6, 0.2, 'AUC = %.2f' % auc_score, bbox=dict(facecolor='white', alpha=0.5))

plt.show()


#Confusion Matrix and Classification Report
print ('\n confusion matrix for the test set :\n',confusion_matrix(y_test,
 y_pred))
print ('\n classification report for the test set :\n',
 classification_report(y_test, y_pred))

#Record the end time
end_time = time.time()

#Calculate the elapsed time
elapsed_time = end_time - start_time
print(f"Elapsed Time: {elapsed_time} seconds")


