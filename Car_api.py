#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app=Flask(__name__)
model=pickle.load(open('model_pkl','rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    final_features=[np.array(int_features)]
    prediction=model.predict(final_features)
    
    output=round(prediction[0],3)
    
    return render_template('index.html', prediction_text='Resale price is {}'.format(output))

if __name__=='__main__':
    app.run(debug=True) 


# In[ ]:




