import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn import preprocessing
from sklearn.linear_model import RidgeCV, LassoCV

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

def make_model(file_loc, target, columns):
	df, X_train, X_test, y_train, y_test = load_clean_data(file_loc, columns, target)
	linear = linear_model.LinearRegression()
	return model_evaluation(linear, X_train, y_train, y_test, X_test),  ridge_lasso('ridge', X_train, y_train, y_test, X_test), ridge_lasso('lasso', X_train, y_train, y_test, X_test)

def model_evaluation(model, features, target, y_test, X_test):
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
		return model_evaluation(ridge, X_train, y_train, y_test, X_test)
  	elif model == 'lasso':
		lasso = LassoCV()
		return model_evaluation(lasso, X_train, y_train, y_test, X_test)

def print_table(file_loc, model_scores, target_name):
	models = ['linear', 'ridge', 'lasso']
	mses = []
	r2s = []
	for i in model_scores:
		mses.append(i[0])
		r2s.append(i[1])
	print '{} Model\t|\tMSE\t\t|\tR2'.format(target_name)
	for i, model in enumerate(models):
		if len(model) > 7:
			print '{0}\t|\t{1:.4f}\t\t|\t{2:.4f}'.format(model, mses[i], r2s[i])
		else:
			print '{0}\t\t|\t{1:.4f}\t\t|\t{2:.4f}'.format(model, mses[i], r2s[i])

if __name__ == '__main__':

	file_loc = 'data/cleaned_ratio_data'

	'''
	Neck
	'''
	columns_neck = ['t_shirt_size', 'height_inches', 'weight_pounds', 'neck']

	ratio_eval_neck, ridge_eval_neck, lasso_eval_neck = make_model(file_loc, 'neck', columns_neck)
	model_scores_neck = [ratio_eval_neck, ridge_eval_neck, lasso_eval_neck]
	print_table(file_loc, model_scores_neck, 'Neck')

	'''
	Sleeve
	'''
	columns_sleeve = ['t_shirt_size', 'height_inches', 'weight_pounds', 'Fit',
	'Full', 'Muscular', 'jacket_size', 'jacket_length', 'pant_inseam_inches', 'sleeve']	

	ratio_eval_sleeve, ridge_eval_sleeve, lasso_eval_sleeve = make_model(file_loc, 'sleeve', columns_sleeve)
	model_scores_sleeve = [ratio_eval_sleeve, ridge_eval_sleeve, lasso_eval_sleeve]
	print_table(file_loc, model_scores_sleeve, 'Sleeve')

	columns_sleeve2 = ['t_shirt_size', 'height_inches', 'weight_pounds', 'shirt_sleeve_inches', 'sleeve']

	ratio_eval_sleeve2, ridge_eval_sleeve2, lasso_eval_sleeve2 = make_model(file_loc, 'sleeve', columns_sleeve2)
	model_scores_sleeve2 = [ratio_eval_sleeve2, ridge_eval_sleeve2, lasso_eval_sleeve2]
	print_table(file_loc, model_scores_sleeve2, 'Sleeve 2')

	'''
	Chest
	'''
	columns_chest = ['jacket_size', 'Slim', 'Very Slim', 'chest']

	ratio_eval_chest, ridge_eval_chest, lasso_eval_chest = make_model(file_loc, 'chest', columns_chest)

	model_scores_chest = [ratio_eval_chest, ridge_eval_chest, lasso_eval_chest]
	print_table(file_loc, model_scores_chest, 'Chest')

	columns_chest2 = ['t_shirt_size', 'height_inches', 'weight_pounds', 'age_years', 'Fit', 'Full', 'Muscular', 'Slim', 'Very Slim', 'chest']

	ratio_eval_chest2, ridge_eval_chest2, lasso_eval_chest2 = make_model(file_loc, 'chest', columns_chest2)

	model_scores_chest2 = [ratio_eval_chest2, ridge_eval_chest2, lasso_eval_chest2]
	print_table(file_loc, model_scores_chest2, 'Chest 2')

	'''
	Waist
	'''
	columns_waist = ['Fit', 'Full', 'Muscular', 'waist']

	ratio_eval_waist, ridge_eval_waist, lasso_eval_waist = make_model(file_loc, 'waist', columns_waist)

	model_scores_waist = [ratio_eval_waist, ridge_eval_waist, lasso_eval_waist]
	print_table(file_loc, model_scores_waist, 'Waist')