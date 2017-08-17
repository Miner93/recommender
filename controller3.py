# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 20:00:21 2017

@author: Spider
"""

from model3 import InputForm
from flask import *
from compute3 import recommend_mv, recommend_bx

app = Flask(__name__)
app.secret_key = 'some_secret'

@app.route('/', methods=['GET', 'POST'])
def start():
    return render_template('start.html')


@app.route('/movielens', methods=['GET', 'POST'])
def movielens():
    form = InputForm(request.form)
    if request.method == 'POST':    
        if form.validate() and form.user_id.data > 0 and form.user_id.data < 944 and form.top_cnt.data > 0 and form.top_cnt.data < 1683:
            result = recommend_mv(form.user_id.data, form.top_cnt.data)
            flash('Top %d' % form.top_cnt.data + ' rekomendacji dla użytkownika %d' % form.user_id.data)
            return render_template('movielens.html', form=form, result=result.to_html(classes='skin'))
        else:
            flash('Podaj poprawny ID użytkownika i ilość rekomendacji')
    
    return render_template('movielens.html', form=form)

    

@app.route('/bookcrossing', methods=['GET', 'POST'])
def bookcrossing():
    form = InputForm(request.form)
    if request.method == 'POST': 
        
        if form.validate() and form.user_id.data > 0 and form.user_id.data < 501 and form.top_cnt.data > 0 and form.top_cnt.data < 68736:
            result = recommend_bx(form.user_id.data, form.top_cnt.data)
            flash('Top %d' % form.top_cnt.data + ' rekomendacji dla użytkownika %d' % form.user_id.data)
            return render_template('bookcrossing.html', form=form, result=result.to_html(classes='skin'))
        else:
            flash('Podaj poprawny ID użytkownika i ilość rekomendacji')
            
    return render_template('bookcrossing.html', form=form)

if __name__ == '__main__':
    app.run(debug=True)
