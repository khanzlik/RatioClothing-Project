import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn import preprocessing
from sklearn.linear_model import RidgeCV, LassoCV

'''
I wrote this model as a pythonic represenation of the Excel model Ratio Clothing provided me with. The Ratio algorithm used a subset of the features (based on what features the customer supplied) to try and predict the customer's shirt size. My file 'new_models.py' provides the model I created to replace the model shown here.
'''

def load_clean_data(file_loc, columns=None, target=None):
	'''
	INPUT: path to file, column names as list of strings (input values), target value as string
	OUTPUT: cleaned dataframe, train-test for X and y based on target specified
	'''
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

def make_model(file_loc, target, columns):
	'''
	Function passes data frame and target variable to the 3 different linear models we are fitting and returns the MSE and r2 as a measure of performance
	'''
	df, X_train, X_test, y_train, y_test = load_clean_data(file_loc, columns, target)
	linear = linear_model.LinearRegression()
	return linear(linear, X_train, y_train, y_test, X_test),  ridge_lasso('ridge', X_train, y_train, y_test, X_test), ridge_lasso('lasso', X_train, y_train, y_test, X_test)

def linear(model, features, target, y_test, X_test):
	model.fit(features, target)
	intercept = model.intercept_
	coef = np.hstack([intercept, model.coef_])
	if X_test is not None:
		prediction = model.predict(X_test)
		mean_squared_error = mse(y_test, model.predict(X_test))
		r2 = model.score(X_test, y_test)
		return mean_squared_error, r2
	else:
		return coef

def ridge_lasso(model, X_train, y_train, y_test, X_test):
	standardizer = preprocessing.StandardScaler().fit(X_train)
  	X_train = standardizer.transform(X_train)
  	X_test = standardizer.transform(X_test)
  	if model == 'ridge':
		alphas = np.logspace(-3, 1)
		ridge = RidgeCV(alphas=alphas)
		return linear(ridge, X_train, y_train, y_test, X_test)
  	elif model == 'lasso':
		lasso = LassoCV()
		return linear(lasso, X_train, y_train, y_test, X_test)

def return_table(file_loc, target_name, target, columns):
	'''
	Makes a table of the MSE and r2 of all of the models for a given target
	'''
	ratio_eval, ridge_eval, lasso_eval = make_model(file_loc, target, columns)
	model_scores = [ratio_eval, ridge_eval, lasso_eval ]
	models = ['linear', 'ridge', 'lasso']
	mses = []
	r2s = []
	for i in model_scores:
		mses.append(i[0])
		r2s.append(i[1])
	result = '{} Model\t|\tMSE\t\t|\tR2\n'.format(target_name)
	for i, model in enumerate(models):
		if len(model) > 7:
			result += '{0}\t|\t{1:.4f}\t\t|\t{2:.4f}\n'.format(model, mses[i], r2s[i])
		else:
			result += '{0}\t\t|\t{1:.4f}\t\t|\t{2:.4f}\n'.format(model, mses[i], r2s[i])
	return result

if __name__ == '__main__':

	file_loc = 'data/cleaned_ratio_data'

	columns_neck = ['t_shirt_size', 'height_inches', 'weight_pounds', 'neck']
	neck_table = return_table(file_loc, 'Neck', 'neck', columns_neck)

	columns_sleeve = ['t_shirt_size', 'height_inches', 'weight_pounds', 'Fit',
	'Full', 'Muscular', 'jacket_size', 'jacket_length', 'pant_inseam_inches', 'sleeve']	
	sleeve_table = return_table(file_loc, 'Sleeve', 'sleeve', columns_sleeve)
	columns_sleeve2 = ['t_shirt_size', 'height_inches', 'weight_pounds', 'shirt_sleeve_inches', 'sleeve']
	sleeve_table2 = return_table(file_loc, 'Sleeve 2', 'sleeve', columns_sleeve2)

	columns_chest = ['jacket_size', 'Slim', 'Very Slim', 'chest']
	chest_table = return_table(file_loc, 'Chest', 'chest', columns_chest)
	columns_chest2 = ['t_shirt_size', 'height_inches', 'weight_pounds', 'age_years', 'Fit', 'Full', 'Muscular', 'Slim', 'Very Slim', 'chest']
	chest_table2 = return_table(file_loc, 'Chest 2', 'chest', columns_chest2)

	columns_waist = ['Fit', 'Full', 'Muscular', 'waist']
	waist_table = return_table(file_loc, 'Waist', 'waist', columns_waist)

