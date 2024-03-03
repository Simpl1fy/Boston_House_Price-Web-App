from flask import Flask, redirect, url_for, render_template, request
from sklearn.preprocessing import MinMaxScaler
import numpy as np

app = Flask(__name__)

@app.route('/')
def form():
    return render_template('index.html')


@app.route('/submit', methods=['POST', 'GET'])
def submit():
    input = [float(x) for x in request.values()]
    array = [np.array(input)]


@app.route('/print/<input>')
def print(input):
    return render_template('print.html', data=input)





if __name__ == '__main__':
    app.run(debug=True)