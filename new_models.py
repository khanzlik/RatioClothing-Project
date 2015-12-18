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
	df = pd.read_csv(file_loc)
	df = df.fillna(-1)
	y = df.pop(target)
	X = df
	params = {'learning_rate': [0.05, 0.1, 0.15], 'n_estimators': [50, 100, 200], 'max_features': [None, 'auto']}
	search = GridSearchCV(GradientBoostingRegressor(), params, n_jobs = -1)
	return search.fit(X, y)

def return_table(file_loc, target):
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

def make_models(file_loc, model_info):
	for shirt_part in model_info:
		df, X_train, X_test, y_train, y_test = load_clean_data(file_loc, shirt_part)
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


	rfr = RandomForestRegressor(n_estimators=100, oob_score=True)
	gbr = GradientBoostingRegressor(learning_rate=0.1, max_features='auto', n_estimators=200)
	model_info = {'neck': rfr, 'sleeve': gbr, 'waist': rfr, 'chest': rfr}


