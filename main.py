from flask import Flask, redirect, url_for, render_template, request
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open('models/model.pkl', 'rb'))

@app.route('/')
def form():
    return render_template('index.html')


@app.route('/submit', methods=['POST', 'GET'])
def submit():
    data = [float(x) for x in request.form.values()]
    x = np.array(data)
    x = x.reshape(1, -1)
    # min_max_scaler = MinMaxScaler(feature_range=(0, 1))
    # x_scaled = min_max_scaler.fit_transform(x.reshape(8, -1))
    # x_scaled = x_scaled.reshape(-1, 8)
    prediction = model.predict(x)
    
    return render_template('print.html', price=prediction)



if __name__ == '__main__':
    app.run(debug=True)