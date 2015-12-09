import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import datasets, linear_model
from sklearn.cross_validation import train_test_split,\
cross_val_score
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor,\
GradientBoostingRegressor
from sklearn.metrics import mean_squared_error as mse

def dummies(df, column):
	df = pd.concat([df, pd.get_dummies(df[column])], axis=1)
	df.drop(column, inplace=True, axis=1)
	return df

def plot_corr(df):
	sns.heatmap(df.corr())
	plt.title('Correlation Plot')
	plt.xticks(rotation=70)
	plt.yticks(rotation=0)
	plt.show()

def pair_plot:
	sns.set()
	sns.pairplot(df.iloc[:,1:8], hue="mission")
	plt.show()


#correlation matrix, heat maps
#skewed distribution (histograms):loop for columns to do subplots
	#df.hist() (look at pandas)

if __name__=='__main__':

	df = pd.read_csv('data/ratio_sizing_data_and_user_profiles.csv')
	drop_columns = ['created_at', 'updated_at', 'source', 'birthday_month',\
	'posture', 'watch_wrist', 'watch_size', 'rise_inches', 'pocket_size',\
	'lastorderdate']
	df.drop(drop_columns, axis=1, inplace=True)

	df = dummies(df, 'build')
	df = dummies(df, 'shoulder_slope')
	df = dummies(df, 'back_pleats')
	df = dummies(df, 'sleeve_fit')
	df = dummies(df, 'tuck')
	df = dummies(df, 'fit')
	df.drop('Average', inplace=True, axis=1)
	df.drop('Normal', inplace=True, axis=1)
	df.drop('Standard', inplace=True, axis=1)
	df.drop('No Pleats', inplace=True, axis=1)
	df.drop('Both', inplace=True, axis=1)
	df.drop('Traditional', inplace=True, axis=1)

	df.describe()


