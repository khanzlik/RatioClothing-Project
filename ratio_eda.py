import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cross_validation import train_test_split,\
cross_val_score
from functions.py import *
from sklearn import datasets, linear_model
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor,\
GradientBoostingRegressor
from sklearn.metrics import mean_squared_error as mse

def dummies(df, column_list):
	df = pd.concat([df, pd.get_dummies(df[column_list])], axis=1)
	df.drop(column_list, inplace=True, axis=1)
	return df

def plot_corr(df):
	sns.heatmap(df.corr())
	plt.title('Correlation Plot')
	plt.xticks(rotation=70)
	plt.yticks(rotation=0)
	plt.show()

def histogram(df, column_list, bins=10):
	df[column_list].hist(bins=bins)
	plt.show()


#correlation matrix, heat maps
#skewed distribution (histograms):loop for columns to do subplots
	#df.hist() (look at pandas)

if __name__=='__main__':

	df = pd.read_csv('data/ratio_sizing_data_and_user_profiles.csv')
	drop_columns = ['created_at', 'updated_at', 'source', 'birthday_month',\
	'posture', 'watch_wrist', 'watch_size', 'rise_inches', 'pocket_size',\
	'lastorderdate', 'shoulder_left', 'shoulder_right']
	df.drop(drop_columns, axis=1, inplace=True)

	df = dummies(df, 'build')
	df = dummies(df, 'shoulder_slope')
	df = dummies(df, 'back_pleats')
	df = dummies(df, 'sleeve_fit')
	df = dummies(df, 'tuck')
	df = dummies(df, 'fit')
	drop_dummies = ['Average', 'Normal', 'Standard', 'No Pleats', 'Both'\
	'Traditional']
	df.drop('drop_dummies', inplace=True, axis=1)

	# df.describe()
	df = df[df['height_inches']<100.0]
	df = df[df['age_years']>2]

	column_list = ['height_inches', 'age_years', 'weight_pounds']
	histogram(df, column_list, bins=50)
	# df.hist(column='height_inches', bins=50)
	# df.hist(column='age_years')
	# df.hist(column='weight_pounds', bins=50)
	# df.hist(column='jacket_size')

	y = df.pop('jacket_size')
	X = df
	X_train, X_test, y_train, y_test = train_test_split(X, y,\
	random_state=1, test_size=.1)



