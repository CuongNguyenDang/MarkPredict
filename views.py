"""
Routes and views for the flask application.
"""

from datetime import datetime
from flask import render_template, request
from MarkPredict import app
import Process

@app.route('/')
@app.route('/home')
def home():
    """Renders the home page."""
    return render_template(
        'index.html',
        title='Home Page',
        year=datetime.now().year,
        toan = 0.00, van = 0.00, li = 0.00, hoa = 0.00,
        sinh = 0.00, su = 0.00, dia = 0.00, gdcd = 0.00, anh = 0.00,
    )

@app.route('/contact')
def contact():
    """Renders the contact page."""
    return render_template(
        'contact.html',
        title='Contact',
        year=datetime.now().year,
        message='Your contact page.'
    )

@app.route('/about')
def about():
    """Renders the about page."""
    return render_template(
        'about.html',
        title='About',
        year=datetime.now().year,
        message='Your application description page.'
    )

@app.route('/get-text', methods=['POST'])
def foo():
    # name = request.form['test']
    toan = float(request.form['toan'])
    van = float(request.form['van'])
    li = float(request.form['li'])
    hoa = float(request.form['hoa'])
    sinh = float(request.form['sinh'])
    su = float(request.form['su'])
    dia = float(request.form['dia'])
    gdcd = float(request.form['gdcd'])
    anh = float(request.form['anh'])
    Process.Prediction(toan, van, li, hoa, sinh, su, dia, gdcd, anh)
    return render_template(
        'result.html',
        title='Home Page',
        year=datetime.now().year,
        toan = float(request.form['toan']),
        van = float(request.form['van']),
        li = float(request.form['li']),
        hoa = float(request.form['hoa']),
        sinh = float(request.form['sinh']),
        su = float(request.form['su']),
        dia = float(request.form['dia']),
        gdcd = float(request.form['gdcd']),
        anh = float(request.form['anh'])
    )
