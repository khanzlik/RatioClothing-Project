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

def linear_regression(X, y):
  model = linear_model.LinearRegression()
  model.fit(X, y)
  print('Coefficients: {}'.format(model.coef_))
  print("Residual sum of squares: {}".format(\
  np.mean((model.predict(X) - y) ** 2)))
  print('Variance score: {}'.format(model.score(X, y)))
  return model

# PCA Plot
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

def ridge(X, y, alphas):
  scores = []
  for alpha in alphas:
    model = Ridge(alpha=alpha, normalize=True)
    fit = model.fit(X, y)
    scores.append(cross_val_score(model, X, y))
  a = alphas[np.argmax(scores)]
  model = Ridge(alpha=a)
  model.fit(X, y)
  return model

def lasso(X, y, alphas):
  scores = []
  for alpha in alphas:
    model = Lasso(alpha=alpha, normalize=True)
    fit = model.fit(X, y)
    scores.append(cross_val_score(model, X, y))
  a = alphas[np.argmax(scores)]
  model = Lasso(alpha=a)
  model.fit(X, y)
  return model

def random_forest(X_train, y_train, num_trees=10):
  model = RandomForestRegressor(n_estimators=num_trees,\
    oob_score=True)
  model.fit(X_transform, y)
  return model

def gradient(X_train, y_train):
  model = GradientBoostingRegressor()
  model.fit(X_transform, y)
  return model


if __name__=='__main__':

  diabetes = datasets.load_diabetes()
  X = diabetes['data']
  y = diabetes['target']

  X_train, X_test, y_train, y_test = train_test_split(X, y,\
  random_state=1, test_size=.1)

  OLS = linear_regression(X_train, y_train)
  OLS_perf = mse(y_test, OLS.predict(X_test))

  standardizer = preprocessing.StandardScaler().fit(X_train)
  X_train = standardizer.transform(X_train)
  X_test = standardizer.transform(X_test)
  pca = PCA()
  pca.fit(X_train)
  pca.explained_variance_ratio_

  scree_plot(pca)
  X_transform = pca.transform(X_train)
  X_transform_test = pca.transform(X_test)

  PCR = linear_regression(X_transform, y_train) # different coefs, same MSE
  PCR_perf = mse(y_test, PCR.predict(X_transform_test))

  alphas = np.logspace(-2, 2)
  ridge_model = ridge(X_train, y_train, alphas)
  lasso_model = lasso(X_train, y_train, alphas)

  ridge_perf = mse(y_test, ridge_model.predict(X_test))
  lasso_per = mse(y_test, lasso_model.predict(X_test))

  # rf_model = random_forest(X_train, y_train)
  # gradient_model = gradient(X_train, y_train)

  # rf_perf = mse(X_test, rf_mdoel.predict(X_test))
  # gradient_model = mse(X_test, gradient_model.predict(X_test))

  # models, labels = [], []
  # models.append(linear_regression(X_transform, y))
  # labels.append('Linear Regression')
  # models.append(ridge(X_transform, y, alphas))
  # labels.append('Ridge Regression')
  # models.append(lasso(X_transform, y, alphas))
  # labels.append('Lasoo Regression')

