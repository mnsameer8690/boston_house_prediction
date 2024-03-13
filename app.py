import pickle
from flask import Flask, request, jsonify, render_template, url_for,app
import numpy as np
import pandas as pd
app=Flask(__name__)
regmodel=pickle.load(open('regression.pkl','rb'))
scaler=pickle.load(open('scaler.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
# def predict_api():
#     data=request.get_json()
#     data_unseen=pd.DataFrame([data])
#     print(data_unseen,"lol")
#     return data_unseen
#     # data_unseen=scaler.transform(data_unseen)
#     # prediction=regmodel.predict(data_unseen)
#     # output=prediction[0]
#     # return jsonify(output)
def predict_api():
    data = request.get_json()['data']
    new_data = scaler.transform(np.array(list(data.values())).reshape(1, -1))
    output = regmodel.predict(new_data)
    return jsonify(output[0])
@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    newdata=scaler.transform(np.array(data).reshape(1,-1))
    output=regmodel.predict(newdata)[0]
    return render_template("home.html",prediction_text="The predicted House price is {}".format(output))
if(__name__=='__main__'):
    app.run(debug=True)

