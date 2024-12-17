# app/model.py
import joblib
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer

class Model:
    def __init__(self):
        self.model = joblib.load('xgb_model.pkl')
        self.vectorizer = joblib.load('vectorizer.pkl')
        self.mlb = joblib.load('mlb.pkl')

    def predict(self, text, rubrics):
        rubrics_vector = self.mlb.transform([rubrics])[0]
        text_vector = self.vectorizer.transform([text]).toarray()[0]
        input_vector = list(text_vector) + list(rubrics_vector)
        return self.model.predict([input_vector])[0]
