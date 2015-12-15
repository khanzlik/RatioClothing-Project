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

def model_evaluation(model, features, target, y_test, X_test):
	model.fit(features, target)
	intercept = model.intercept_
	coef = np.hstack([intercept, model.coef_])
	if X_test is not None:
		prediction = model.predict(X_test)
		mean_squared_error = mse(y_test, model.predict(X_test))
		r2 = model.score(X_test, y_test)
		return coef, prediction, mean_squared_error, r2
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
	columns = ['t_shirt_size', 'height_inches', 'weight_pounds', 'neck']
	df, X_train, X_test, y_train, y_test = load_clean_data(file_loc, columns, 'neck')

	linear = linear_model.LinearRegression()
	ratio_eval_neck = model_evaluation(linear, X_train, y_train, y_test, X_test)
	# mse =  0.47750327483826049; r2 = 0.67984086429868151 

  	alphas = np.logspace(-3, 1)
	ridge = RidgeCV(alphas=alphas)
	ridge_eval_neck = ridge_lasso(ridge, X_train, y_train, y_test, X_test)
	# mse = 0.47721527750811232; r2 = 0.68003396240117264
	lasso = LassoCV()
	lasso_eval_neck = ridge_lasso(lasso, X_train, y_train, y_test, X_test)
	# mse = 0.47722398030101915; r2 = 0.6800281273026465

	'''
	Sleeve
	'''
	columns = ['t_shirt_size', 'height_inches', 'weight_pounds', 'Fit',
	'Full', 'Muscular', 'jacket_size', 'jacket_length', 'pant_inseam_inches', 'sleeve']	
	df, X_train, X_test, y_train, y_test = load_clean_data(file_loc, columns, 'sleeve')

	ratio_eval_sleeve = model_evaluation(linear, X_train, y_train, y_test, X_test)
	# mse = 0.39771933086395506; r2 = 0.83666254075411528
	ridge_eval_sleeve = ridge_lasso(ridge, X_train, y_train, y_test, X_test)
	# mse = 0.40528999341433258; r2 = 0.83355338138009027
	lasso_eval_sleeve = ridge_lasso(lasso, X_train, y_train, y_test, X_test)
	# mse = 0.39687011534069044; r2 = 0.83701130103594745

	columns = ['t_shirt_size', 'height_inches', 'weight_pounds', 'shirt_sleeve_inches']
	df, X_train, X_test, y_train, y_test = load_clean_data(file_loc, columns, 'shirt_sleeve_inches')
	ratio_eval_sleeve2 = model_evaluation(linear, X_train, y_train, y_test, X_test)
	# mse = 1.1247710512623568; r2 = 0.71683849830813262

	ridge_eval_sleeve2 = ridge_lasso(ridge, X_train, y_train, y_test, X_test)
	# mse = 1.1267313982382454; r2 = 0.71634498027803506
	lasso_eval_sleeve2 = ridge_lasso(lasso, X_train, y_train, y_test, X_test)
	# mse = 1.1251534027213046; r2 = 0.71674224119592456

	'''
	Chest
	'''
	columns = ['jacket_size', 'Slim', 'Very Slim', 'chest']
	df, X_train, X_test, y_train, y_test = load_clean_data(file_loc, columns, 'chest')
	ratio_eval_chest = model_evaluation(linear, X_train, y_train, y_test, X_test)
	# mse =  0.54706151455779206; r2 = 0.86078110314750333

  	ridge_eval_chest = ridge_lasso(ridge, X_train, y_train, y_test, X_test)
  	# mse = 4.2048059326703857; r2 = 0.18922634965511664
	lasso_eval_chest = ridge_lasso(lasso, X_train, y_train, y_test, X_test)
	# mse =  4.2097740338073146; r2 = 0.18826839688427088

	columns = ['t_shirt_size', 'height_inches', 'weight_pounds', 'age_years', 'Fit', 'Full', 'Muscular', 'Slim', 'Very Slim', 'chest']
	df, X_train, X_test, y_train, y_test = load_clean_data(file_loc, columns, 'chest')
	ratio_eval_chest2 = model_evaluation(linear, X_train, y_train, y_test, X_test)
	# mse =  0.76175726435387658; r2 = 0.84407444361854078

	ridge_eval_chest2 = ridge_lasso(ridge, X_train, y_train, y_test, X_test)
	# mse =  0.76465039887538788; r2 = 0.84348224236091873
	lasso_eval_chest2 = ridge_lasso(lasso, X_train, y_train, y_test, X_test)
	# mse =  0.77741931289295585 r2 = 0.84086854884495987

	'''
	Waist
	'''
	columns = ['Fit', 'Full', 'Muscular', 'waist']
	df, X_train, X_test, y_train, y_test = load_clean_data(file_loc, columns, 'waist')
	ratio_eval_waist = model_evaluation(linear, X_train, y_train, y_test, X_test)
	# mse = 4.2104438126990829; r2 = 0.18813924964520723

	ridge_eval_chest = ridge_lasso(ridge, X_train, y_train, y_test, X_test)
	# mse =  4.2048059326703857; r2 = 0.18922634965511664
	lasso_eval_chest = ridge_lasso(lasso, X_train, y_train, y_test, X_test)
	# mse = 4.2097740338073146; r2 = 0.18826839688427088