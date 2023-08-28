import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# Loading data
bookings = pd.read_csv('hotel_bookings_clean.csv')
data = bookings.dropna()

x = data.iloc[:, 1:]
y = data.iloc[:, 0]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(x, y, test_size=.25, random_state=123)

# Forecasting using KNN
knn_clf = KNeighborsClassifier(n_neighbors=3)
knn_clf.fit(X_train, y_train)
predsKnn = knn_clf.predict(X_test)
accuracy_score(y_test, preds)


# Forecasting using Dtree
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
predDtree = dtree.predict(X_test)
accuracy_score(y_test, predDtree)


# Forecasting using XGBoost
xgb_clf = xgb.XGBClassifier(random_state=123, n_estimators=50)
xgb_clf.fit(X_train, y_train)
predXGB = xgb_clf.predict(X_test)
accuracy_score(y_test, predXGB)


# Forecasting using SVM
svm_clf = SVC(random_state=3432, C=0.5)
svm_clf.fit(X_train, y_train)
predSVM = svm_clf.predict(X_test)
accuracy_score(y_test, predSVM)






