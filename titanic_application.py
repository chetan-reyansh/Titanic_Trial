#!/usr/bin/env python

from flask import Flask,request,render_template
import pandas as pd
import numpy as np
import joblib

app=Flask(__name__)

model = joblib.load('titanic.pkl')  

cols=['pclass','age','fare','adult_male','alone','sex_female','sex_male','alive_no','alive_yes']

@app.route("/")
def getModel():
    return render_template("form.html")


# In[10]:


@app.route("/predict",methods=["POST"])
def predict():
    input_data=[]
    
    for col in cols:
        input_data.append(float(request.form[col]))


    pred=model.predict([input_data])

    if pred == 1:
        return 'Survived'
    else:
        return 'Didnt Survive'

if __name__=='__main__':
    app.run(host='127.0.0.1')

