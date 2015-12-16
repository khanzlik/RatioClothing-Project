import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def clean_data(file_loc):
	df = pd.read_csv(file_loc)
	drop_columns = ['created_at', 'updated_at', 'source', 'birthday_month', 'posture', 'watch_wrist', 'watch_size', 'rise_inches', 'pocket_size','lastorderdate', 'shoulder_left', 'shoulder_right', 'shoulder_slope', 'estimated_birth_year', 'button_count', 'button_stance', 'posture_alteration', 'lastorderid', 'back_pleats']
	df.drop(drop_columns, axis=1, inplace=True)

	fill_columns = ['build', 'jacket_length', 'tuck', 'fit']
	df = fill_nans(df, fill_columns)

	df = dummies(df, 'build')
	df = dummies(df, 'sleeve_fit')
	df = dummies(df, 'tuck')
	df = dummies(df, 'fit')
	drop_dummies = ['Average', 'Standard', 'Both', 'Traditional']
	df.drop(drop_dummies, inplace=True, axis=1)

	return df

def dummies(df, column_list):
	df = pd.concat([df, pd.get_dummies(df[column_list])], axis=1)
	df.drop(column_list, inplace=True, axis=1)
	return df

def fill_nans(df, columns):
	for column in columns:
		if column == 'build':
			df['build'].fillna('Average', inplace=True)
		elif column == 'jacket_length':
			df['jacket_length'].fillna(0, inplace=True)
		elif column == 'tuck':
			df['tuck'].fillna('Both', inplace=True)
		elif column == 'fit':
			df['fit'].fillna('Slim', inplace=True)
	return df

def plot_corr(df):
	sns.heatmap(df.corr())
	plt.title('Correlation Plot')
	plt.xticks(rotation=90)
	plt.yticks(rotation=0)
	plt.show()

def histogram(df, column_list, bins=10):
	df[column_list].hist(bins=bins)
	plt.show()

if __name__=='__main__':

	id_userid = pd.read_csv('data/id_userid.csv')
	ids = pd.read_csv('data/id.csv')
	ratio = pd.read_csv('data/ratio_sizing_data.csv')

	orders  = pd.merge(id_userid, ids, 'left', 'id')
	total_orders = pd.merge(orders, ratio, 'inner', 'user_id')
	index = total_orders.duplicated('user_id').loc

	df_duplicates = total_orders.ix[total_orders.duplicated('user_id'), :]

	df_duplicates.to_csv('data/repeat_orders')
	file_loc = 'data/repeat_orders'
	drop_columns = ['created_at', 'updated_at', 'source', 'birthday_month', 'posture', 'watch_wrist', 'watch_size', 'rise_inches', 'pocket_size','lastorderdate', 'shoulder_left', 'shoulder_right', 'shoulder_slope', 'estimated_birth_year', 'button_count', 'button_stance', 'posture_alteration', 'lastorderid', 'back_pleats']
	df = clean_data(file_loc)
	df.to_csv('data/cleaned_ratio_data')

	# df.describe()
	column_list = ['height_inches', 'age_years', 'weight_pounds']
	histogram(df, column_list, bins=50)

	'''
	Retention
	'''
	total_orders.to_csv('data/total_orders')
	num_shirt = pd.DataFrame(total_orders.groupby('user_id').size())
	num_shirt.rename(columns={0: 'retention'}, inplace=True)
	num_shirt.reset_index(inplace=True)
	retention = pd.merge(total_orders, num_shirt, 'inner', 'user_id')
	retention.eval('churn = retention <= 1')
	retention.churn = retention.churn.astype(int)
	drop_columns = ['id_y', 'id_x', 'retention']
	retention.drop(drop_columns, axis=1, inplace=True)
	retention.drop_duplicates('user_id', inplace=True)
	retention.to_csv('data/churn')
	
	file_loc_churn = 'data/churn'
	df_retention = clean_data(file_loc_churn)
	df_retention.to_csv('data/cleaned_churn')


