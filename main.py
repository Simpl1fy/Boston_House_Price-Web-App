from flask import Flask, redirect, url_for, render_template, request
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load('models/model.pkl', 'rb')

@app.route('/')
def form():
    return render_template('index.html')


@app.route('/submit', methods=['POST', 'GET'])
def submit():
    data = [float(x) for x in request.form.values()]
    x = [np.array(data)]
    x_scaled = MinMaxScaler.fit_transform(x)
    prediction = model.predict(x_scaled)
    
    return render_template('print.html', price=prediction)



if __name__ == '__main__':
    app.run(debug=True)