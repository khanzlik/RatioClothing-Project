# modeled off sklearn diabetes data set
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import datasets, linear_model
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
#stats models.api as SM, SM OLS for metrics

'''
Linear regression
'''
diabetes = datasets.load_diabetes()
# diabetes is a dictionary of arrays, call X and y with keys

X = diabetes['data']
y = diabetes['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

model = linear_model.LinearRegression()
model.fit(X_train, y_train)
# coefficients
print('Coefficients: {}'.format(model.coef_))
# Mean Square Error
print("Residual sum of squares: {}".format(\
	np.mean((model.predict(X_test) - y_test) ** 2)))
# Explained Variance (1 is perfect prediction)
print('Variance score: {}'.format(model.score(X_test, y_test)))
'''
Coefficients: [  -7.85951708 -245.05253542  575.11667591\
323.85372717 -519.77447335 250.61132753    0.96367294  180.50891964\
614.75959394   52.10619986]
Residual sum of squares: 2903.10000132
Variance score: 0.443974132651
  '''

plt.scatter(X_test[:,0], y_test
#plt.show()

'''
Principal Component Analysis
'''

X = StandardScaler().fit_transform(diabetes.data)
pca = PCA() #how many components
pca.fit(X)
pca.explained_variance_ratio_

def scree_plot(pca, title=None):
    num_components = pca.n_components_
    ind = np.arange(num_components)
    vals = pca.explained_variance_ratio_
    #plt.figure(figsize=(10, 6), dpi=250)
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
#plt.show()
#np.cumsum(pca.explained_variance_ratio_)

X_transform = pca.transform(X)
# apply this X_transform to linear regression now
# linear_regression(X_transform, y)
# Coefficients: [-21.46555352 -12.18646095  11.33141471  29.15276267  -0.43853856
#  10.41901241   5.63232952   3.947253     0.55600835  32.64848419]
# Residual sum of squares: 2903.10000132
# Variance score: 0.443974132651
# Same MSE/Variance, completely different coefficients? 
