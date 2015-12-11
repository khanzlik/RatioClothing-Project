from functions import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import linear_model
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error as mse
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

def load_clean_data(file_loc, columns=None, target=None):
	df = pd.read_csv(file_loc)
	if columns is not None and target is not None:
		df = df[columns]
		df.dropna(inplace=True)
		y = df.pop(target).values 
		X = df.values
		X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=.1)
	  	return df, X_train, X_test, y_train, y_test
	else:
		return df

def model_evaluation(model, features, target, y_test, X_test, survey=None):
	model.fit(features, target)
	intercept = model.intercept_
	coef = np.hstack([intercept, model.coef_])
	if survey is not None:
		prediction = model.predict(survey)
		mean_squared_error = mse(y_test, model.predict(X_test))
		variance = model.score(X_test, y_test)
		return coef, prediction, mean_squared_error, variance
	else:
		return coef

if __name__ == '__main__':

	file_loc = 'data/cleaned_ratio_data'

	'''
	Neck
	'''
	columns = ['t_shirt_size', 'height_inches', 'weight_pounds', 'neck']
	df, X_train, X_test, y_train, y_test = load_clean_data(file_loc, columns, 'neck')
	linear = linear_model.LinearRegression()
	survey = np.array([[9, 70, 210]])
	ratio_eval_neck = model_evaluation(linear, X_train, y_train, y_test, X_test, survey)
	# mse =  0.31745101549213006; R2 = 71746544559006742

	standardizer = preprocessing.StandardScaler().fit(X_train)
  	X_train = standardizer.transform(X_train)
  	X_test = standardizer.transform(X_test)

  	alphas = np.logspace(-3, 1)
	ridge = RidgeCV(alphas=alphas)
	ridge_eval = model_evaluation(ridge, X_train, y_train, y_test, X_test, survey)
	lasso = LassoCV()
	lasso_eval = model_evaluation(lasso, X_train, y_train, y_test, X_test, survey)

	'''
	Sleeve
	'''
	columns = ['t_shirt_size', 'height_inches', 'weight_pounds', 'Fit',
	'Full', 'Muscular', 'jacket_size', 'jacket_length', 'pant_inseam_inches', 'sleeve']
	df, X_train, X_test, y_train, y_test = load_clean_data(file_loc, columns, 'sleeve')
	survey = np.array([[6, 76, 185, 1, 0, 0, 41, 1, 33]])
	ratio_eval_sleeve = model_evaluation(linear, X_train, y_train, y_test, X_test, survey)

	standardizer = preprocessing.StandardScaler().fit(X_train)
  	X_train = standardizer.transform(X_train)
  	X_test = standardizer.transform(X_test)

	ridge_eval_2 = model_evaluation(ridge, X_train, y_train, y_test, X_test, survey)
	lasso_eval_2 = model_evaluation(lasso, X_train, y_train, y_test, X_test, survey)

	columns = ['t_shirt_size', 'height_inches', 'weight_pounds', 'sleeve']
	df, X_train, X_test, y_train, y_test = load_clean_data(file_loc, columns, 'sleeve')
	survey = np.array([[6, 76, 185]])
	ratio_eval_sleeve2 = model_evaluation(linear, X_train, y_train, y_test, X_test, survey)



# required fields: age, height, weight, tshirt size
# 12 target variables

# if statements for when they supply different information:
	# if they supply shirt_neck_inches, we don't predict neck
# incoporating measurements from store
# build=None=Average/suitlength=None=0/tuck=None=0/fit=None=Slim/Tailored
