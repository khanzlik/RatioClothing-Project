import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def make_ratio_df(file_id_userid, file_id, file_sizing_data):
	'''
	INPUT: csv files of userid's for id, orders, and sizing data
	OUTPUT: new data frame of only repeated orders and the dataframe of the total orders
	'''
	id_userid = pd.read_csv(file_id_userid)
	ids = pd.read_csv(file_id)
	ratio = pd.read_csv(file_sizing_data)

	orders  = pd.merge(id_userid, ids, 'left', 'id')
	total_orders = pd.merge(orders, ratio, 'inner', 'user_id')
	index = total_orders.duplicated('user_id').loc

	df_duplicates = total_orders.ix[total_orders.duplicated('user_id'), :]

	df_duplicates.to_csv('data/repeat_orders')
	file_loc = 'data/repeat_orders'
	df = clean_data(file_loc)
	df.to_csv('data/cleaned_ratio_data')

	return total_orders, df

def make_retention_df(total_orders):
	'''
	INPUT: dataframe of total orders made
	OUTPUT: dataframe with a churn column for if customers were retained
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
	return df_retention

def clean_data(file_loc):
	'''
	INPUT: path to file location
	OUTPUT: dataframe with insignificant columns dropped, dummies, and filled in default values
	'''
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

def descriptions(df):
	df.describe()
	column_list = ['height_inches', 'age_years', 'weight_pounds']
	hist = histogram(df, column_list, bins=50)
	return df.describe(), hist

def plot_corr(df):
	'''
	Heat map of correlations
	'''
	plt.clf()
	drop_columns = ['Unnamed: 0', 'id_x', 'id_y', 'armpit_inches', 'torso_length_inches', 'leg_length_inches', 'knee_inches', 'arm_left_inches', 'arm_right_inches', 'thigh_inches']
	df.drop(drop_columns, inplace=True, axis=1)
	sns.heatmap(df.corr())
	plt.title('Correlation Plot')
	plt.xticks(rotation=90)
	plt.yticks(rotation=0)
	plt.savefig('images/heat_map.png', dpi=300)

def histogram(df, column_list, bins=10):
	df[column_list].hist(bins=bins)
	plt.show()

if __name__=='__main__':

	file_id_userid = 'data/id_userid.csv'
	file_id = 'data/id.csv'	
	file_sizing_data = 'data/ratio_sizing_data.csv'

	total_orders, df = make_ratio_df(file_id_userid, file_id, file_sizing_data)

	make_retention_df(total_orders)

