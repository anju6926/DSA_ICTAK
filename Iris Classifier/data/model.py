import numpy as np
import pandas as pd
import pickle

data = pd.read_excel('iris .xls')


X = data.drop(['Classification'],axis=1)
y = data['Classification']
from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.2,random_state=42)

#Logistic Regression

from sklearn.linear_model import LogisticRegression
logr = LogisticRegression()
logr_model= logr.fit(X_train,y_train)

#Saving the model to disk
pickle.dump(logr_model,open('model.pkl','wb'))

from sklearn.tree import DecisionTreeClassifier
mod_dt = DecisionTreeClassifier(max_depth = 3, random_state = 1)
mod_dt.fit(X_train,y_train)
pickle.dump(mod_dt,open('dt_model_iris.pkl','wb'))

from sklearn.neighbors import KNeighborsClassifier
mod_nn=KNeighborsClassifier(n_neighbors=5)
mod_nn.fit(X_train,y_train)
pickle.dump(mod_nn, open('nn_model_iris.pkl','wb'))

from sklearn.svm import SVC
svc = SVC(kernel='linear').fit(X_train, y_train)
pickle.dump(svc,open ('svc_model_iris.pkl','wb'))

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
model=lr.fit(X_train,y_train)
pickle.dump(lr, open ('linr_model_iris.pkl','wb'))

from sklearn.linear_model import LogisticRegression
mod_lr = LogisticRegression(solver = 'newton-cg').fit(X_train, y_train)
pickle.dump(mod_lr,open ('logr_model_iris.pkl','wb'))
