from functions import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import linear_model
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error as mse

df = pd.read_csv('data/cleaned_ratio_data')

'''
Ratio's Current Model
'''
def model_evaluation(model, features, target, y_test, x_test, test_x=None):
	model.fit(features, target)
	intercept = model.intercept_
	coef = np.hstack([intercept, model.coef_])
	if test_x is not None:
		prediction = model.predict(test_x)
		performance = mse(y_test, model.predict(X_test))
		return coef, prediction, performance
	else:
		return coef
 

if __name__ == '__main__':

	columns = df[['t_shirt_size', 'height_inches', 'weight_pounds', 'neck']]
	columns.dropna(inplace=True)
	y = columns.pop('neck').values 
	X = columns.values
	X_train, X_test, y_train, y_test = train_test_split(X, y,\
  	random_state=1, test_size=.1)
	model = linear_model.LinearRegression() 
	test_x = np.array([[9, 70, 210]])
	ratio = model_evaluation(model, X_train, y_train, y_test, X_test, test_x)
	# (array([ 16.11502377,   0.187813  ,  -0.06965918,   0.02083254]),\
		# array([ 17.30403131]), mse =  0.31745101549213006)

# lm = linear_model.LinearRegression()
# neckfeatures = df[['t_shirt_size', 'height_inches', 'weight_pounds']]
# test_x = np.array([[9, 70, 210]])
# model_evaluation(lm, neckfeatures, df['neck'], test_x)


# 12 target variables
# if they supply shirt_neck_inches, we don't predict neck
# changing build to 1/2/3/4?, fit preference?
# build=None=Average/suitlength=None=0/tuck=None=0
# incoporating measurements from store
