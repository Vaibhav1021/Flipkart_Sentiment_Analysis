from flask import Flask, request, jsonify, render_template
from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import string

app = Flask(__name__)

# Load the model and vectorizer
model = load('/home/ubuntu/webapp/logistic_regression_model.joblib')
tfidf_vectorizer = load('/home/ubuntu/webapp/tfidf_vectorizer.joblib') 

# Define preprocess_text function

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review_text = request.form['review']
    processed_review = preprocess_text(review_text)
    tfidf_features = tfidf_vectorizer.transform([processed_review])
    prediction = model.predict(tfidf_features)
    sentiment = 'positive' if prediction[0] == 1 else 'negative'
    return render_template('index.html', sentiment=sentiment, review=review_text)



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
