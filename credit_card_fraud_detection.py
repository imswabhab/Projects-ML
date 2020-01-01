import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings 
warnings.filterwarnings('ignore')

raw_data = pd.read_csv('creditcard.csv')
data = raw_data.copy()

data.head()

#Separating Feature and Target matrices
X = data.drop(['Class'], axis=1)
y=data['Class']

from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
X = scale.fit_transform(X)

 #Split the data into Training set and Testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size =0.2,random_state=42)


from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=0, ratio = 1.0)
X_train,y_train = sm.fit_sample(X_train,y_train)

#using Support Vector Classifier
from sklearn.svm import SVC
svc_classifier = SVC()
svc_classifier.fit(X_train,y_train)

#predicting y_test
y_pred = svc_classifier.predict(X_test)

#building the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

""" By using Svc i have achived an accuracy of 81%"""

#using XGboost Classifier

from xgboost import XGBClassifier
model  = XGBClassifier()
model.fit(X_train,y_train)

#predicting y_test
y_pred_xgb = model.predict(X_test)

#building confusion matrix for xgboost
cm_xgb = confusion_matrix(y_test,y_pred_xgb)
 
""" by using Xgboost Clasiifier i have achived an accuracy of 83.5%"""


#using naive bayes 

from sklearn.naive_bayes import GaussianNB
GNB = GaussianNB()
GNB.fit(X_train,y_train)

#predicting x_test
y_pred_GNB = GNB.predict(X_test)

#building confusion matrix for xgboost
cm_NB = confusion_matrix(y_test,y_pred_GNB)
 
""" by using Xgboost Clasiifier i have achived an accuracy of 80%"""


#using Decision Tree
from sklearn.tree import DecisionTreeClassifier
D_classifier = DecisionTreeClassifier()
D_classifier.fit(X_train,y_train)

#predicting x_test
y_pred_decision_tree = D_classifier.predict(X_test)

#building confusion matrix for xgboost
cm_decision_tree = confusion_matrix(y_test,y_pred_decision_tree)
 
""" by using Decisioin tree Clasiifier i have achived an accuracy of 81.3%"""


#using random forest classifier 
from sklearn.ensemble import RandomForestClassifier
rfc_classifier = RandomForestClassifier()
rfc_classifier.fit(X_train,y_train)

y_pred_rf = rfc_classifier.predict(X_test)

#building Confusion MATRIX
cm_rf = confusion_matrix(y_test,y_pred_rf)
 
""" by usuing Random forest CLassifier I have acheived an acuuraccy of 87%"


