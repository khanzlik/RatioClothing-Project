import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import datasets, linear_model
from sklearn.cross_validation import train_test_split,\
cross_val_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier,\
GradientBoostingClassifier

#dataset = datasets.load_diabetes()
def linear_regression(data):
	dataset = data
	X = dataset['data']
	y = dataset['target']
	X_train, X_test, y_train, y_test = train_test_split(X, y,\
		random_state=1)
	model = linear_model.LinearRegression()
	model.fit(X_train, y_train)
	print('Coefficients: {}'.format(model.coef_))
	print("Residual sum of squares: {}".format(\
	np.mean((model.predict(X_test) - y_test) ** 2)))
	print('Variance score: {}'.format(model.score(X_test, y_test)))
