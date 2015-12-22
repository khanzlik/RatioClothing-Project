import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle 
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn import preprocessing
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.grid_search import GridSearchCV

def load_clean_data(file_loc, target=None):
	'''
	INPUT: path to file for dataframe, target value as string (dependent variable)
	OUTPUT: cleaned dataframe, train-test for X and y based on target specified
	'''
	df = pd.read_csv(file_loc)
	if target is not None:
		df = df.fillna(-1)
		y = df.pop(target).values 
		X = df.values
		X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=.1)
	  	return df, X_train, X_test, y_train, y_test
	else:
		return df

def make_model(file_loc, target):
	'''
	INPUT:  path to file and  target variable
	OUTPUT: linear regression, ridge, lasso, random forest, and gradient boosting models (mean squared error and r2) for the target variable

	Function passes data frame and target variable train-test to the 5 different models we are fitting and returns the MSE and r2 as a measure of performance
	'''
	df, X_train, X_test, y_train, y_test = load_clean_data(file_loc, target)
	linear = linear_model.LinearRegression()
	return linear_regression(linear, X_train, y_train, y_test, X_test),  ridge_lasso('ridge', X_train, y_train, y_test, X_test), ridge_lasso('lasso', X_train, y_train, y_test, X_test), random_forest(X_train, y_train, y_test, X_test), gradient(X_train, y_train, y_test, X_test, file_loc, target)

def linear_regression(model, features, target, y_test, X_test):
	model.fit(features, target)
	intercept = model.intercept_
	coef = np.hstack([intercept, model.coef_])
	prediction = model.predict(X_test)
	mean_squared_error = mse(y_test, model.predict(X_test))
	r2 = model.score(X_test, y_test)
	return (mean_squared_error, r2)

def ridge_lasso(model, X_train, y_train, y_test, X_test):
	standardizer = preprocessing.StandardScaler().fit(X_train)
  	X_train = standardizer.transform(X_train)
  	X_test = standardizer.transform(X_test)
  	if model == 'ridge':
  		alphas = np.logspace(-3, 1)
  		ridge = RidgeCV(alphas=alphas)
  		return linear_regression(ridge, X_train, y_train, y_test, X_test)
  	elif model == 'lasso':
  		lasso = LassoCV()
  		return linear_regression(lasso, X_train, y_train, y_test, X_test)

def random_forest(X_train, y_train, y_test, X_test, num_trees=100):
	model = RandomForestRegressor(n_estimators=num_trees, oob_score=True)
	model.fit(X_train, y_train)
	prediction = model.predict(X_test)
	mean_squared_error = mse(y_test, model.predict(X_test))
	r2 = model.score(X_test, y_test)
	return (mean_squared_error, r2)

def gradient(X_train, y_train, y_test, X_test, file_loc, target):
	'''
	Passes to grid search function within this function to pick the best parameters for each gradient boosted model depending on the target variable we are trying to predict
	'''
	grid = grid_search(file_loc, target)
	best_params = grid.best_params_
	learn_rate = best_params['learning_rate']
	n_estimators = best_params['n_estimators']
	max_feat = best_params['max_features']
	model = GradientBoostingRegressor(learning_rate=learn_rate,  n_estimators=n_estimators, max_features=max_feat)
	model.fit(X_train, y_train)
	prediction = model.predict(X_test)
	mean_squared_error = mse(y_test, model.predict(X_test))
	r2 = model.score(X_test, y_test)
	return (mean_squared_error, r2)

def plot_feature_importance(file_loc, target, model, max_col=10):
	'''
	INPUT: path to location of df, target variable, model as string (either 'random_forest' or 'gradient'), and max columns set to 10 to show top 10 most important features
	OUTPUT: plots the  most important features for the specified model
	'''
	df, X_train, X_test, y_train, y_test = load_clean_data(file_loc, target)
	columns = list(df.columns)
	if model == 'random_forest':
		model = RandomForestRegressor(n_estimators=100, oob_score=True)
	elif model == 'gradient':
		model = GradientBoostingRegressor(learning_rate = 0.05)
	model.fit(X_train, y_train)
	result = pd.Series(model.feature_importances_, index=columns).sort_values(ascending=False)
	if max_col:
		result = result[:max_col]
	result.plot(kind='bar')
	plt.title('Feature Importances: {}, {}'.format(model.__class__.__name__, target))
	plt.show()

def grid_search(file_loc, target):
	'''
	Grid search for best paramaters for Gradient Boosted model
	'''
	df = pd.read_csv(file_loc)
	df = df.fillna(-1)
	y = df.pop(target)
	X = df
	params = {'learning_rate': [0.05, 0.1, 0.15], 'n_estimators': [50, 100, 200], 'max_features': [None, 'auto']}
	search = GridSearchCV(GradientBoostingRegressor(), params, n_jobs = -1)
	return search.fit(X, y)

def return_table(file_loc, target):
	'''
	Makes a table of the MSE and r2 of all of the models for a given target
	'''
	models = ['linear', 'ridge', 'lasso', 'random forest', 'gradient']
	ratio_eval, ridge_eval, lasso_eval, rf, gb = make_model(file_loc, target)
	mses = [ratio_eval[0], ridge_eval[0], lasso_eval[0], rf[0], gb[0]]
	r2s = [ratio_eval[1], ridge_eval[1], lasso_eval[1], rf[1], gb[1]]
	result = '{} Model\t|\tMSE\t\t|\tR2\n'.format(target)
	for i, model in enumerate(models):
		if len(model) > 7:
			result += '{0}\t|\t{1:.4f}\t\t|\t{2:.4f}\n'.format(model, mses[i], r2s[i])
		else:
			result += '{0}\t\t|\t{1:.4f}\t\t|\t{2:.4f}\n'.format(model, mses[i], r2s[i])
	return result

def best_models(file_loc, model_info):
	'''
	INPUT: path to file, parameters of models as dictionary {'target': model}
	OUTPUT: pickled models
	'''
	df = pd.read_csv(file_loc)
	drop_columns = ['Unnamed: 0', 'Unnamed: 0.1', 'id_x', 'user_id', 'id_y', 'neck_inches', 'shoulder_inches', 'chest_inches', 'overarm_inches', 'thigh_inches', 'belly_inches', 'bicep_inches', 'waist_inches', 'wrist_inches', 'seat_inches', 'armpit_inches', 'torso_length_inches', 'leg_length_inches', 'knee_inches', 'center_back_arm_left_inches', 'center_back_arm_right_inches', 'arm_left_inches', 'arm_right_inches', 'seat', 'yoke', 'length', 'bicep', 'cuff', 'asym_cuff', 'asym_sleeve', 'forearm_alteration', 'short_sleeve_length', 'short_sleeve_opening', 'Slim', 'Fitted', 'Tucked', 'Untucked', 'Slim.1', 'Very Slim']
	df.drop(drop_columns, axis=1, inplace=True)
	for shirt_part in model_info:
		if shirt_part == 'neck':
			drop_columns = ['sleeve', 'chest', 'waist']
			df_copy = df.drop(drop_columns, axis=1)
		elif shirt_part == 'chest':
			drop_columns = ['neck', 'waist', 'sleeve']
			df_copy = df.drop(drop_columns, axis=1)
		elif shirt_part == 'waist':
			drop_columns = ['neck', 'chest', 'sleeve']
			df_copy = df.drop(drop_columns, axis=1)
		else:
			drop_columns = ['neck', 'chest', 'waist']
			df_copy = df.drop(drop_columns, axis=1)
		df_copy = df_copy.fillna(-1)
		y = df_copy.pop(shirt_part).values 
		X = df_copy.values
		X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=.1)
		model = model_info[shirt_part]
		model.fit(X_train, y_train)
		with open('app/models/{}.pkl'.format(shirt_part), 'wb') as file:
			pickle.dump(model, file)

if __name__ == '__main__':

	file_loc = 'data/cleaned_ratio_data'

	neck_eval = return_table(file_loc, 'neck')
	sleeve_eval = return_table(file_loc, 'sleeve')
	chest_eval = return_table(file_loc, 'chest')
	waist_eval = return_table(file_loc, 'waist')

