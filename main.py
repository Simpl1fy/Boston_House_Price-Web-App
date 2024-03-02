from flask import Flask, redirect, url_for, render_template, request
from sklearn.preprocessing import MinMaxScaler
import numpy as np

app = Flask(__name__)

@app.route('/')
def form():
    return render_template('index.html')


@app.route('/submit', methods=['POST', 'GET'])
def submit():
    if request.method == 'POST':
        indus = float(request.form['indus'])
        nox = float(request.form['nox'])
        rm = float(request.form['rm'])
        age = float(request.form['age'])
        dis = float(request.form['dis'])
        tax = float(request.form['tax'])
        ptratio = float(request.form['ptratio'])
        lstat = float(request.form['lstat'])
    # array = np.array([indus, nox, rm, age, dis, tax, ptratio, lstat])
    data = {'indus': indus, 'nox': nox, 'rm': rm, 'age': age, 'dis': dis, 'tax': tax, 'ptratio': ptratio, 'lstat': lstat}
    return redirect(url_for('print', input= data))


@app.route('/print/<input>')
def print(input):
    return render_template('print.html', data=input)





if __name__ == '__main__':
    app.run(debug=True)