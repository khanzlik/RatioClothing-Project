import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_curve

def load_clean_data(file_loc, target):
	'''
	INPUT: path to file for dataframe, target value as string (dependent variable)
	OUTPUT: cleaned dataframe, X, y, nad train-test for X and y based on target specified
	'''
	df = pd.read_csv(file_loc)
	drop_columns = ['Unnamed: 0', 'Unnamed: 0.1', 'user_id']
	df.drop(drop_columns, axis=1, inplace=True)
	df = df.fillna(-1)
	y = df.pop(target)
	X = df.values
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
	return df, X, y, X_train, X_test, y_train, y_test

def logistic_model(X, y):
	model = linear_model.LogisticRegression()
	model.fit(X, y)
	return model

def random_forest(X, y, num_trees=10):
	model = RandomForestClassifier(n_estimators=num_trees, oob_score=True)
	model.fit(X, y)
	return model

def gradient(X, y):
	model = GradientBoostingClassifier()
	model.fit(X, y)
	return model

def models_labels(X_train, y_train):
	models, labels = [], []
	models.append(logistic(X_train, y_train))
	labels.append('Logistic Regression')
	models.append(random_forest(X_train, y_train, 100))
	labels.append('Random Forest')
	models.append(gradient(X_train, y_train))
	labels.append('Gradient Boosting')
	predictions = [model.predict_proba(X_test)[:,1] for model in models]
	return models, labels, predictions

def plot_roc_curve(y_test, predictions, labels):
	'''
	prints ROC curve of logistic, random forest, and gradient bossted models
	'''
	for y_predic, label in zip(predictions, labels):
		fpr, tpr, thresholds = roc_curve(y_true, y_predic)
		plt.plot(fpr, tpr, label=label)
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic')
	plt.legend(loc="best")
	plt.show()

def feature_importance(df, model, X, y):
	'''
	INPUT: dataframe, model that performed best from ROC curve, X, y
	OUTPUT: feature importances in descdending order
	'''
	model.fit(X, y)
	columns = df.columns
	feature_importance = pd.Series(model.feature_importances_, index=columns).sort_values(ascending=False)
	return feature_importance

if __name__ == '__main__':

	file_loc = 'data/cleaned_churn'
	df, X, y, X_train, X_test, y_train, y_test = load_clean_data(file_loc, 'churn')

	# Chose model based on ROC curve
	model = RandomForestClassifier(n_estimators=100, oob_score=True)
	feat_importances = feature_importance(df, model, X, y)

