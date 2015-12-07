import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import datasets, linear_model
from sklearn.cross_validation import train_test_split,\
cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier,\
GradientBoostingClassifier

diabetes = datasets.load_diabetes()
X = diabetes['data']
y = diabetes['target']
def linear_regression(X, y):
	X_train, X_test, y_train, y_test = train_test_split(X, y,\
		random_state=1)
	model = linear_model.LinearRegression()
	model.fit(X_train, y_train)
	print('Coefficients: {}'.format(model.coef_))
	print("Residual sum of squares: {}".format(\
	np.mean((model.predict(X_test) - y_test) ** 2)))
	print('Variance score: {}'.format(model.score(X_test, y_test)))

# PCA

X = StandardScaler().fit_transform(diabetes.data)
pca = PCA()
pca.fit(X)
pca.explained_variance_ratio_

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
    # plt.show()

X_transform = pca.transform(X)
# linear_regression(X_transform, y)

