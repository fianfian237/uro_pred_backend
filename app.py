from flask import Flask, Response
from model import Model
from requests import request
import pandas as pd
import jsonify


app = Flask(__name__)
model_grade = Model('Modeles/model_grade.joblib')
model_stade = Model('Modeles/model_stade.joblib')


@app.route('/predict_grade', methods=["POST"])
def predict_grade():
    input = pd.DataFrame(request.body)
    prediction = model_grade.predict(input)
    result = dict({
    'prediction': dict(prediction.iloc[0])
    })
    return jsonify(result)

@app.route('/predict_stade', methods=["POST"])
def predict_stade():
    input = pd.DataFrame(request.body)
    prediction = model_stade.predict(input)
    result = dict({
        'prediction': dict(prediction.iloc[0])
    })
    return jsonify(result)

@app.route('/health')
def health_check():
    return Response("", status = 200)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=4000)