import os
import pandas as pd
from flask import Flask, render_template
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('submission_form.html')

# @app.route('/predictions')
# def shirt_predictions():
# 	return render_template('results.html')

if __name__ == '__main__':
 	app.run(host='0.0.0.0', port=8080, debug=True)