import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from tensorflow.keras.models import load_model
from torch.nn.functional import softmax
import matplotlib.pyplot as plt

model = load_model("stock_model.h5", compile=False)

MODEL_NAME = "ProsusAI/finbert"
tokenizer_bert = BertTokenizer.from_pretrained(MODEL_NAME)
model_bert = BertForSequenceClassification.from_pretrained(MODEL_NAME)

NEWS_API_KEY = "a1c2677ee19a4a10a6d2ebba8773f3ea"

def fetch_stock_data(company, period="1mo"):
    try:
        stock = yf.Ticker(company)
        df = stock.history(period=period)
        return df[['Close']]
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return None

def fetch_news(company):
    url = f"https://newsapi.org/v2/everything?q={company}&language=en&apiKey={NEWS_API_KEY}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            articles = response.json().get("articles", [])
            return [article["title"] for article in articles if "title" in article]
    except Exception as e:
        st.error(f"Error fetching news: {e}")
    return []

def analyze_sentiment(headlines):
    if not headlines:
        return 0  
    sentiments = []
    for text in headlines:
        inputs = tokenizer_bert(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model_bert(**inputs)
        probabilities = softmax(outputs.logits, dim=1)[0]
        sentiment_score = probabilities[2].item() - probabilities[0].item()  # Positive - Negative
        sentiments.append(sentiment_score)
    return np.mean(sentiments) if sentiments else 0

def predict_stock(company):
    df = fetch_stock_data(company)
    if df is None or df.empty:
        st.error("Stock data not found. Check the symbol.")
        return None, None
    news_headlines = fetch_news(company)
    sentiment_score = analyze_sentiment(news_headlines)
    X = df['Close'].pct_change().dropna().values[-10:]
    if len(X) < 10:
        st.error("Not enough stock data for prediction.")
        return None, None
    X = np.reshape(X, (1, 10, 1))
    try:
        predicted_return = model.predict(X)[0][0]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None
    last_close_price = df['Close'].iloc[-1]
    predicted_price = last_close_price * (1 + predicted_return + sentiment_score * 0.01)
    return predicted_price, df

st.sidebar.title("Stock Market Predictor")
page = st.sidebar.radio("Navigation", ["Home", "Predictions", "About Us"])

if page == "Home":
    st.title("Stock Market Prediction App")
    st.write("This app predicts future stock prices based on past trends and real-time news sentiment.")

elif page == "Predictions":
    st.title("Predict Stock Price")
    company = st.text_input("Enter Stock Symbol (e.g., AAPL for Apple):")
    if st.button("Predict"):
        predicted_price, df = predict_stock(company)
        if predicted_price is not None:
            st.subheader(f"Predicted Stock Price for {company}: ${predicted_price:.2f}")
            # Plot historical stock prices
            st.subheader("Stock Price Trend")
            plt.figure(figsize=(10, 5))
            plt.plot(df.index, df["Close"], label="Actual Price", marker="o")
            plt.axhline(y=predicted_price, color="r", linestyle="--", label="Predicted Price")
            plt.xlabel("Date")
            plt.ylabel("Stock Price")
            plt.legend()
            st.pyplot(plt)

elif page == "About Us":
    st.title("Developers")
    st.write("""
    **Developer:**
    - VAIBHAV GUPTA (22BCE0042)
""")
