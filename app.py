from flask import Flask, Response
from model import Model
from flask import request
import pandas as pd
from flask import jsonify
import json


app = Flask(__name__)
model_grade = Model('Modeles/model_grade.joblib')
model_stade = Model('Modeles/model_stade.joblib')


@app.route('/predict_grade', methods=["POST"])
def predict_grade():
    input_json = {k: [request.args.get(k)] for k in model_stade.meta_data["required_input"]}
    input = pd.DataFrame(input_json)
    app.logger.info(input)
    prediction = model_grade.predict(input)
    result = {
        'prediction': prediction[0]
    }
    return result

@app.route('/predict_stade', methods=["POST"])
def predict_stade():
    input_json = {k: [request.args.get(k)] for k in model_stade.meta_data["required_input"]}
    input = pd.DataFrame(input_json)
    app.logger.info(input)
    prediction = model_stade.predict(input)
    result = {
        'prediction': prediction[0]
    }
    return result

@app.route('/health')
def health_check():
    return Response("", status = 200)


if __name__ == '__main__':
    app.run(debug=True)
    # app.run(debug=True, host='0.0.0.0', port=4000)