from flask import Flask, redirect, url_for, render_template, request
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pickle
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata', methods=['POST', 'GET'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('print.html')
    else:
        data = CustomData(
                indus = float(request.form.get('indus')),
                nox = float(request.form.get('nox')),
                rm = float(request.form.get('rm')),
                age = float(request.form.get('age')),
                tax = float(request.form.get('tax'),
                ptratio = float(request.form.get('ptratio'))
            )

                
     



if __name__ == '__main__':
    app.run(debug=True)
