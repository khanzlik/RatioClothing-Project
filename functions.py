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
from sklearn.ensemble import RandomForestClassifier,\
GradientBoostingClassifier

def linear_regression(X, y):
	X_train, X_test, y_train, y_test = train_test_split(X, y,\
		random_state=1)
	model = linear_model.LinearRegression()
	model.fit(X_train, y_train)
	print('Coefficients: {}'.format(model.coef_))
	print("Residual sum of squares: {}".format(\
	np.mean((model.predict(X_test) - y_test) ** 2)))
	print('Variance score: {}'.format(model.score(X_test, y_test)))

# Standardizing Data 
def standardize(X):
	for i in X:
		X = (i - np.mean(X))/np.std(X)
		return X

# PCA

# data = diabetes.data
# def pca(data):
#     X = StandardScaler().fit_transform(data)
#     pca = PCA()
#     pca.fit(X)
#     pca.explained_variance_ratio_
#     X_transform = pca.transform(X)
#     return X_transform, pca

def scree_plot(pca, title=None):
    num_components = pca.n_components_
    ind = np.arange(num_components)
    vals = pca.explained_variance_ratio_
    ax = plt.subplot(111)
    ax.bar(ind, vals, 0.35, 
           color=[(0.949, 0.718, 0.004),
                  (0.898, 0.49, 0.016),
                  (0.863, 0, 0.188),
                  (0.694, 0, 0.345),
                  (0.486, 0.216, 0.541),
                  (0.204, 0.396, 0.667),
                  (0.035, 0.635, 0.459),
                  (0.486, 0.722, 0.329),
                 ])
    for i in xrange(num_components):
        ax.annotate(r"%s%%" % ((str(vals[i]*100)[:4])),\
        	(ind[i]+0.2, vals[i]), va="bottom", ha="center",\
        	fontsize=12)
    
    ax.set_xticklabels(ind, 
                       fontsize=12)
    ax.set_ylim(0, max(vals)+0.05)
    ax.set_xlim(0-0.45, 8+0.45)

    ax.xaxis.set_tick_params(width=0)
    ax.yaxis.set_tick_params(width=2, length=12)

    ax.set_xlabel("Principal Component", fontsize=12)
    ax.set_ylabel("Variance Explained (%)", fontsize=12)

    if title is not None:
        plt.title(title, fontsize=16)

def ridge(X_transform, y):
  dummy_arr = []
  for i,a in enumerate(alphas):
    fit = Ridge(alpha=a, normalize=True).fit(X_transform, y)
    params[i] = fit.coef_
    dummy_arr.append(((fit.predict(X_transform) - y)**2)\
      .mean())
  for param in params.T:
    plt.plot(alphas, param)
    plt.show()

def lasso(X_transform, y):
  alphas, _, coefs = linear_model.lars_path(X, y,\
    method='lasso', verbose=True)


if __name__=='__main__':
  diabetes = datasets.load_diabetes()
  X = diabetes['data']
  y = diabetes['target']

  linear_regression(X, y)
  # X = standardize(X)

  X = preprocessing.StandardScaler().fit_transform(diabetes.data)
  pca = PCA()
  pca.fit(X)
  pca.explained_variance_ratio_

  scree_plot(pca)
  # plt.show()
  X_transform = pca.transform(X)
  # linear_regression(X_transform, y)

  k = X_transform.shape[1]
  alphas = np.logspace(-5, 5)
  params = np.zeros((len(alphas), k))
  train_MSE = np.zeros((len(alphas)))
  test_MSE = np.zeros((len(alphas)))

  ridge(X_transform, y)
  # fit.score(X_transform, y)
  plt.plot(np.log10(alphas), train_MSE, color = 'b', label = "Training Ridge Set")


# Standardizing right? Or should I do a function?
# Do I use X_train/y_train in PCA/ridge/lasso?
