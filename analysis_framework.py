# modeled off sklearn diabetes data set
'''
Linear regression
'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import datasets, linear_model
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
#stats models.api as SM, SM OLS for metrics

diabetes = datasets.load_diabetes()
# diabetes is a dictionary of arrays, call X and y with keys

X = diabetes['data']
y = diabetes['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

model = linear_model.LinearRegression()
model.fit(X_train, y_train)
# coefficients
print('Coefficients: {}'.format(model.coef_))
# Mean Square Error
print("Residual sum of squares: {}".format(\
	np.mean((model.predict(X_test) - y_test) ** 2)))
# Explained Variance (1 is perfect prediction)
print('Variance score: {}'.format(model.score(X_test, y_test)))
'''
Coefficients: [  -7.85951708 -245.05253542  575.11667591  323.85372717 -519.77447335
  250.61132753    0.96367294  180.50891964  614.75959394   52.10619986]
Residual sum of squares: 2903.10000132
Variance score: 0.443974132651
  '''

plt.scatter(X_test[:,0], y_test)