import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn import preprocessing
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.grid_search import GridSearchCV

def random_forest(X, y, num_trees=10):
	model = RandomForestClassifier(n_estimators=num_trees, oob_score=True)
	model.fit(X, y)
	return model

def logistic(X, y):
	model = LogisticRegression()
	model.fit(X, y)
	return model

def gradient(X, y):
	model = GradientBoostingClassifier()
	model.fit(X, y)
	return model

def plot_roc_curve(y_true, predictions, labels):
	for y_predic, label in zip(predictions, labels):
		fpr, tpr, thresholds = roc_curve(y_true, y_predic)
		plt.plot(fpr, tpr, label=label)
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic')
	plt.legend(loc="best")
	plt.show()


if __name__ == '__main__':

	df = pd.read_csv('data/cleaned_total_orders')

	y = df.pop('retained')
	X = df.values
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

	models, labels = [], []
	models.append(random_forest(X_train, y_train, 100))
	labels.append('Random Forest with 100 Trees')
	models.append(logistic(X_train, y_train))
	labels.append('Logistic Regression')
	models.append(gradient(X_train, y_train))
	labels.append('Gradient Boosting with Defaults')

	predictions = [model.predict_proba(X_test)[:,1] for model in models]
	plot_roc_curve(y_test, predictions, labels)

