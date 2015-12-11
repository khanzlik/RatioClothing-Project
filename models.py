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

def model_evaluation(model, features, target, y_test, X_test):
	model.fit(features, target)
	intercept = model.intercept_
	coef = np.hstack([intercept, model.coef_])
	if X_test is not None:
		prediction = model.predict(X_test)
		mean_squared_error = mse(y_test, model.predict(X_test))
		variance = model.score(X_test, y_test)
		return coef, prediction, mean_squared_error, variance
	else:
		return coef

def ridge_lasso(model, X_train, y_train, y_test, X_test):
	standardizer = preprocessing.StandardScaler().fit(X_train)
  	X_train = standardizer.transform(X_train)
  	X_test = standardizer.transform(X_test)
  	if model == ridge:
  		return model_evaluation(ridge, X_train, y_train, y_test, X_test)
  	elif model == lasso:
  		return model_evaluation(lasso, X_train, y_train, y_test, X_test)

if __name__ == '__main__':

	file_loc = 'data/cleaned_ratio_data'

	'''
	Neck
	'''
	columns = ['t_shirt_size', 'height_inches', 'weight_pounds', 'shirt_neck_inches']
	mask_train = pd.notnull(df['shirt_neck_inches'])
	X_train = df[columns][mask_train]
	y_train = X_train.pop('shirt_neck_inches')
	mask_test = pd.notnull(df['neck_inches'])
	X_test = df[columns][mask_test]
	y_test = X_test.pop('neck_inches')
	linear = linear_model.LinearRegression()
	ratio_eval_neck = model_evaluation(linear, X_train, y_train, y_test, X_test)
	# mse =  0.31745101549213006; R2 = 71746544559006742

  	alphas = np.logspace(-3, 1)
	ridge = RidgeCV(alphas=alphas)
	ridge_eval_neck = ridge_lasso(ridge, X_train, y_train, y_test, X_test)
	lasso = LassoCV()
	lasso_eval_neck = ridge_lasso(lasso, X_train, y_train, y_test, X_test)

	'''
	Sleeve
	'''
	columns = ['t_shirt_size', 'height_inches', 'weight_pounds', 'Fit',
	'Full', 'Muscular', 'jacket_size', 'jacket_length', 'pant_inseam_inches', 'shirt_sleeve_inches']
	df, X_train, X_test, y_train, y_test = load_clean_data(file_loc, columns, 'shirt_sleeve_inches')
	ratio_eval_sleeve = model_evaluation(linear, X_train, y_train, y_test, X_test)

	ridge_eval_sleeve = ridge_lasso(ridge, X_train, y_train, y_test, X_test)
	lasso_eval_sleeve = ridge_lasso(lasso, X_train, y_train, y_test, X_test)

	columns = ['t_shirt_size', 'height_inches', 'weight_pounds', 'shirt_sleeve_inches']
	df, X_train, X_test, y_train, y_test = load_clean_data(file_loc, columns, 'shirt_sleeve_inches')
	ratio_eval_sleeve2 = model_evaluation(linear, X_train, y_train, y_test, X_test)

	ridge_eval_sleeve2 = ridge_lasso(ridge, X_train, y_train, y_test, X_test)
	lasso_eval_sleeve2 = ridge_lasso(lasso, X_train, y_train, y_test, X_test)

	'''
	Chest
	'''
	columns = ['jacket_size', 'Slim', 'Very Slim', 'chest']
	df, X_train, X_test, y_train, y_test = load_clean_data(file_loc, columns, 'chest')
	ratio_eval_chest = model_evaluation(linear, X_train, y_train, y_test, X_test)

  	ridge_eval_chest = ridge_lasso(ridge, X_train, y_train, y_test, X_test)
	lasso_eval_chest = ridge_lasso(lasso, X_train, y_train, y_test, X_test)

	columns = ['t_shirt_size', 'height_inches', 'weight_pounds', 'age_years', 'Fit', 'Full', 'Muscular', 'Slim', 'Very Slim', 'chest']
	df, X_train, X_test, y_train, y_test = load_clean_data(file_loc, columns, 'chest')
	ratio_eval_chest2 = model_evaluation(linear, X_train, y_train, y_test, X_test)

	ridge_eval_chest2 = ridge_lasso(ridge, X_train, y_train, y_test, X_test)
	lasso_eval_chest2 = ridge_lasso(lasso, X_train, y_train, y_test, X_test)

	'''
	Waist
	'''
	columns = ['Fit', 'Full', 'Muscular', 'waist']
	df, X_train, X_test, y_train, y_test = load_clean_data(file_loc, columns, 'waist')
	ratio_eval_waist = model_evaluation(linear, X_train, y_train, y_test, X_test)

	ridge_eval_chest = ridge_lasso(ridge, X_train, y_train, y_test, X_test)
	lasso_eval_chest = ridge_lasso(lasso, X_train, y_train, y_test, X_test)


# required fields: age, height, weight, tshirt size
# 12 target variables-->concentrate on neck, sleeve, chest, waist

# build table of MSE for 3 different models and for 2 different response cats.
# sklearn.utlis resample  ((ix, iy) for i in xrange(1000))

#what if i did something like just include all the relevant columns and drop columns that aren't relevant?
# if statements for when they supply different information?
# incorporating measurements from store?
