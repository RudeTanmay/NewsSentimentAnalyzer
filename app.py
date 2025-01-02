from flask import Flask, render_template, jsonify, request
import requests
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle

app = Flask(__name__)

# Load the pre-trained model
model = load_model('deep_lstm_model.h5')

# Initialize tokenizer (you'll need to fit this with your training data)
tokenizer = Tokenizer(num_words=10000)

# Load the fitted tokenizer
try:
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
except:
    print("Warning: tokenizer.pickle not found. Make sure to fit and save the tokenizer first.")

def predict_sentiment(news_text: str):
    # Tokenize and pad the input text
    sequences = tokenizer.texts_to_sequences([news_text])
    padded_sequence = pad_sequences(sequences, maxlen=256)
    
    # Make prediction
    prediction = model.predict(padded_sequence)
    return "Positive" if prediction[0][0] > 1 else "Negative"

def fetch_news():
    url = "https://newsapi.org/v2/top-headlines?country=us&apiKey=971092d9e809431a8d031b7eec52b21d"
    response = requests.get(url)
    return response.json() if response.status_code == 200 else None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/news')
def get_news():
    news_data = fetch_news()
    if not news_data:
        return jsonify({'error': 'Failed to fetch news'}), 500
    
    articles = []
    for article in news_data['articles']:
        articles.append({
            'title': article['title'],
            'description': article.get('description', 'No description available'),
            'url': article['url'],
            'publishedAt': article['publishedAt'],
            'urlToImage': article.get('urlToImage')
        })
    
    return jsonify({
        'articles': articles
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_text():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    text = data['text']
    if not text.strip():
        return jsonify({'error': 'Text is empty'}), 400
        
    sentiment = predict_sentiment(text)
    return jsonify({
        'sentiment': sentiment
    })

if __name__ == '__main__':
    app.run(debug=True)