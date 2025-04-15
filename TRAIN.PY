import pandas as pd
import numpy as np
import yfinance as yf
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import requests

NEWS_API_KEY = "a1c2677ee19a4a10a6d2ebba8773f3ea"

MODEL_NAME = "ProsusAI/finbert"
tokenizer_bert = BertTokenizer.from_pretrained(MODEL_NAME)
model_bert = BertForSequenceClassification.from_pretrained(MODEL_NAME)

def fetch_stock_data(company, period="1y"):
    stock = yf.Ticker(company)
    df = stock.history(period=period)
    return df[['Close']]

def fetch_news(company):
    url = f"https://newsapi.org/v2/everything?q={company}&language=en&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        articles = response.json()["articles"]
        return [article["title"] for article in articles if article["title"]]
    return []

def analyze_sentiment(headlines):
    sentiments = []
    for text in headlines:
        inputs = tokenizer_bert(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model_bert(**inputs)
        probabilities = softmax(outputs.logits, dim=1)[0]
        sentiment_score = probabilities[2].item() - probabilities[0].item()  # Positive - Negative
        sentiments.append(sentiment_score)
    return np.mean(sentiments) if sentiments else 0

def prepare_data(df):
    df['Return'] = df['Close'].pct_change()
    df.dropna(inplace=True)
    X, y = [], []
    sequence_length = 10
    for i in range(len(df) - sequence_length):
        X.append(df['Return'].iloc[i:i+sequence_length].values)
        y.append(df['Return'].iloc[i+sequence_length])
    return np.array(X), np.array(y)

def train_model(company):
    df = fetch_stock_data(company)
    news_headlines = fetch_news(company)
    sentiment_score = analyze_sentiment(news_headlines)
    X, y = prepare_data(df)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    model.fit(X, y, epochs=20, batch_size=16, verbose=1)
    model.save("stock_model.h5")
    print("Model trained and saved as stock_model.h5")

if __name__ == "__main__":
    train_model("AAPL")  
