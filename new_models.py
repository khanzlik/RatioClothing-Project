import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn import preprocessing
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.grid_search import GridSearchCV

def load_clean_data(file_loc, target=None):
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
	df, X_train, X_test, y_train, y_test = load_clean_data(file_loc, target)
	linear = linear_model.LinearRegression()
	return linear_regression(linear, X_train, y_train, y_test, X_test), ridge_lasso('ridge', X_train, y_train, y_test, X_test), ridge_lasso('lasso', X_train, y_train, y_test, X_test), random_forest(X_train, y_train, y_test, X_test), gradient(X_train, y_train, y_test, X_test)

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

def gradient(X_train, y_train, y_test, X_test):
	model = GradientBoostingRegressor()
	model.fit(X_train, y_train)
	prediction = model.predict(X_test)
	mean_squared_error = mse(y_test, model.predict(X_test))
	r2 = model.score(X_test, y_test)
	return (mean_squared_error, r2)

def plot_feature_importance(model, columns, max_col=10):
	result = pd.Series(model.feature_importances_, index=columns).sort_values(ascending=False)
	if max_col:
		result = result[:max_col]
	result.plot(kind='bar')
	plt.title('Feature Importances:  {}'.format(model.__class__.__name__))
	plt.show()

if __name__ == '__main__':

	file_loc = 'data/cleaned_ratio_data'

	ratio_eval_neck, ridge_eval_neck, lasso_eval_neck, rf_neck, gb_neck = make_model(file_loc, 'neck')

	ratio_eval_sleeve, ridge_eval_sleeve, lasso_eval_sleeve, rf_sleeve, gb_sleeve = make_model(file_loc, 'sleeve')

	ratio_eval_chest, ridge_eval_chest, lasso_eval_chest, rf_chest, gb_chest = make_model(file_loc, 'chest')

	ratio_eval_chest, ridge_eval_chest, lasso_eval_chest, rf_chest, gb_chest = make_model(file_loc, 'waist')


	'''
	Neck
	'''
	# df, X_train, X_test, y_train, y_test = load_clean_data(file_loc, 'neck')
	# columns = list(df.columns)
	# # Linear Regresion: mse =  0.31885454402716518, r2 = 0.77279074986611529
	# linear = linear_model.LinearRegression()
	# ratio_eval_neck = linear_regression(linear, X_train, y_train, y_test, X_test)
	# # Ridge: mse =  0.328163151931468; r2 = 0.76615762557372069
	# alphas = np.logspace(-3, 1)
	# ridge = RidgeCV(alphas=alphas)
	# ridge_eval_neck = ridge_lasso(ridge, X_train, y_train, y_test, X_test)
	# # Lasso: mse = 0.32470144401621842; r2 = 0.76862436808794798
	# lasso = LassoCV()
	# lasso_eval_neck = ridge_lasso(lasso, X_train, y_train, y_test, X_test)
	# # Random Forrest: mse = 0.18862087264150951, r2 = 0.92705939013792449
	# rf_neck = random_forest(X_train, y_train, y_test, X_test)
	# forest = RandomForestRegressor(n_estimators=100, oob_score=True)
	# forest.fit(X_train, y_train)
	# plot_feature_importance(forest, columns)
	# # Gradient Boosting: mse 0.13569632383074304; r2 0.94752557085732669
	# gb_neck = gradient(X_train, y_train, y_test, X_test)
	# gb = GradientBoostingRegressor()
	# gb.fit(X_train, y_train)
	# plot_feature_importance(gb, columns)
	# '''
	# Sleeve
	# '''
	# df, X_train, X_test, y_train, y_test = load_clean_data(file_loc, 'sleeve')

	# linear = linear_model.LinearRegression()
	# ratio_eval_sleeve = linear_regression(linear, X_train, y_train, y_test, X_test)
	# ridge_eval_sleeve = ridge_lasso(ridge, X_train, y_train, y_test, X_test)
	# lasso_eval_sleeve = ridge_lasso(lasso, X_train, y_train, y_test, X_test)
	# rf_neck = random_forest(X_train, y_train, y_test, X_test)
	# gb_neck = gradient(X_train, y_train, y_test, X_test)
	# '''
	# Chest
	# '''
	# df, X_train, X_test, y_train, y_test = load_clean_data(file_loc, 'chest')

	# linear = linear_model.LinearRegression()
	# ratio_eval_chest = linear_regression(linear, X_train, y_train, y_test, X_test)
	# ridge_eval_chest = ridge_lasso(ridge, X_train, y_train, y_test, X_test)
	# lasso_eval_chest = ridge_lasso(lasso, X_train, y_train, y_test, X_test)
	# rf_chest = random_forest(X_train, y_train, y_test, X_test)
	# gb_chest = gradient(X_train, y_train, y_test, X_test)
	# '''
	# Waist
	# '''
	# df, X_train, X_test, y_train, y_test = load_clean_data(file_loc, 'waist')

	# linear = linear_model.LinearRegression()
	# ratio_eval_waist = linear_regression(linear, X_train, y_train, y_test, X_test)
	# ridge_eval_waist = ridge_lasso(ridge, X_train, y_train, y_test, X_test)
	# lasso_eval_waist = ridge_lasso(lasso, X_train, y_train, y_test, X_test)
	# rf_waist = random_forest(X_train, y_train, y_test, X_test)
	# gb_waist = gradient(X_train, y_train, y_test, X_test)


