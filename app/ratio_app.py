import os
import pickle
import pandas as pd
import numpy as np
from flask import Flask, render_template, request
app = Flask(__name__)

def make_predictions(models, form_info):
	predictions = []
	for model in models:
		predictions.append(model.predict(form_info))
	return predictions

@app.route('/')
def home():
	return render_template('submission_form.html')

@app.route('/predictions', methods=['POST'])
def shirt_predictions():
	names = ["height_inches", "weight_pounds", "age_years", "jacket_size", "jacket_length", "shirt_neck_inches", "shirt_sleeve_inches", "pant_waist_inches", "pant_inseam_inches", "t_shirt_size", "build"]
	form_info = [float(request.form[name]) for name in names]
	form_info = np.array(form_info).reshape(1, -1)
	df = pd.DataFrame(data=form_info, columns=names)

	builds = {1: 'Fit', 2:'Full', 3:'Muscular'}
	for build in builds.itervalues():
		df[build] = 0

	for row_idx in df.index.values:
		for k, build in builds.iteritems():
			if df.loc[row_idx, 'build'] == k:
				df.loc[row_idx, build] = 1
	df.drop('build', axis=1, inplace=True)

	neck, sleeve, chest, waist = load_pickle_obs(['models/neck.pkl', 'models/sleeve.pkl', 'models/chest.pkl', 'models/waist.pkl'])
	predictions = make_predictions([neck, sleeve, chest, waist], df.values)

	neck = 'Neck: {} '.format(str(round(predictions[0], 2)))
	sleeve = 'Sleeve: {} '.format(str(round(predictions[1], 2)))
	chest = 'Chest: {} '.format(str(round(predictions[2], 2)))
	waist = 'Waist: {} '.format(str(round(predictions[3], 2)))
	preds = [neck, sleeve, chest, waist]
	return render_template('results.html', preds=preds)

def load_pickle_obs(file_paths):
    objs = []
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            objs.append(pickle.load(f))
    return tuple(objs)


if __name__ == '__main__':
 	app.run(host='0.0.0.0', port=8080, debug=True)