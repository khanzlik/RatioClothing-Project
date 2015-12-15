df = pd.read_csv('data/cleaned_ratio_data')
df = df.fillna(-1)
y = df.pop('neck')
X = df

params = {'learning_rate': [0.05, 0.1, 0.15], 'n_estimators': [50, 100, 200], 'max_features': [None, 'auto']}

search = GridSearchCV(GradientBoostingRegressor(), params, n_jobs = -1)
search.fit(X, y)