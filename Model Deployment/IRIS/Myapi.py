from flask import Flask, request
import numpy as np
import pickle
import pandas as pd
import flasgger
from flasgger import Swagger

app=Flask(__name__)
Swagger(app)

pickle_in = open("MyClassifier.pkl","rb")
classifier=pickle.load(pickle_in)

@app.route('/')
def welcome():
    return "Welcome All"

@app.route('/predict',methods=["Get"])
def iris():
    
    """Let's Predict the type of Iris
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: SepalLengthCm
        in: query
        type: number
        required: true
      - name: SepalWidthCm
        in: query
        type: number
        required: true
      - name: PetalLengthCm
        in: query
        type: number
        required: true
      - name: PetalWidthCm
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
        
    """
    SepalLengthCm=request.args.get("SepalLengthCm")
    SepalWidthCm=request.args.get("SepalLengthCm")
    PetalLengthCm=request.args.get("PetalLengthCm")
    PetalWidthCm=request.args.get("PetalWidthCm")
    prediction=classifier.predict([[SepalLengthCm,SepalLengthCm,PetalLengthCm,PetalWidthCm]])
    print(prediction)
    if prediction == 0:
        return 'Iris-Setosa'
    elif prediction == 1:
        return 'Iris-Versicolor'
    elif prediction == 2:
        return 'Iris-Virginica'
        
    
@app.route('/predict_file',methods=["POST"])
def iris_file():
    """Let's Predict the type of Iris
    This is using docstrings for specifications.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
      
    responses:
        200:
            description: The output values
        
    """
    df_test=pd.read_csv(request.files.get("file"))
    print(df_test.head())
    prediction=classifier.predict(df_test)
    
    return str(list(prediction))

if __name__=='__main__':
    app.run()
    
    
