# app/app.py
from flask import Flask, request, render_template, jsonify
from model import Model

app = Flask(__name__)
model = Model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']
    rubrics = data['rubrics']
    prediction = round(model.predict(text, rubrics))
    
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
