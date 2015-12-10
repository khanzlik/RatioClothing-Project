import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cross_validation import train_test_split,\
cross_val_score
from functions import *
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

if __name__=='__main__':

	df = pd.read_csv('data/ratio_sizing_data.csv')
	drop_columns = ['created_at', 'updated_at', 'source', 'birthday_month',\
	'posture', 'watch_wrist', 'watch_size', 'rise_inches', 'pocket_size',\
	'lastorderdate', 'shoulder_left', 'shoulder_right', 'shoulder_slope',\
	'estimated_birth_year', 'button_count', 'button_stance']
	df.drop(drop_columns, axis=1, inplace=True)

	df = dummies(df, 'build')
	df = dummies(df, 'back_pleats')
	df = dummies(df, 'sleeve_fit')
	df = dummies(df, 'tuck')
	df = dummies(df, 'fit')
	drop_dummies = ['Average', 'Standard', 'No Pleats', 'Both', 'Traditional']
	df.drop(drop_dummies, inplace=True, axis=1)

	# df.describe()
	df = df[df['height_inches']<100.0]
	df = df[df['age_years']>2]

	column_list = ['height_inches', 'age_years', 'weight_pounds']
	histogram(df, column_list, bins=50)

