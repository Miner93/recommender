# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 19:55:35 2017

@author: Spider
"""

from wtforms import Form, IntegerField, validators

class InputForm(Form):
    user_id = IntegerField(
              label='ID użytkownika:',
              validators=[validators.InputRequired()])
    top_cnt = IntegerField(
              label='Ilość rekomendacji:',
              validators=[validators.InputRequired()])