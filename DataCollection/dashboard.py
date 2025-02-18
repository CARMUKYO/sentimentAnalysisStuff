import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from prophet import Prophet
from sklearn.linear_model import LinearRegression

# Load trained models
nb_model = joblib.load("naive_bayes_model.pkl")
svm_model = joblib.load("svm_model.pkl")
lr_model = joblib.load("logistic_regression_model.pkl")

# Load sample data
sentiment_data = pd.read_csv("sentiment_trends.csv")

# Initialize VADER
analyzer = SentimentIntensityAnalyzer()

# UI Layout
st.title("📊 Sentiment Analysis & Election Prediction Dashboard")

# Section 1: Sentiment Classification
st.header("🔍 Sentiment Analysis")

uploaded_file = st.file_uploader("Upload a CSV file for sentiment classification", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['sentiment'] = df['text'].apply(lambda x: "pos" if analyzer.polarity_scores(x)['compound'] > 0 else "neg" if analyzer.polarity_scores(x)['compound'] < 0 else "neutral")
    
    st.write("Classified Sentiments:")
    st.dataframe(df.head())

    # Save results
    df.to_csv("classified_sentiments.csv", index=False)

# Section 2: Sentiment Trends
st.header("📈 Sentiment Trends Over Time")
sentiment_data['timestamp'] = pd.to_datetime(sentiment_data['timestamp'])

fig, ax = plt.subplots(figsize=(10, 5))
sns.lineplot(x='timestamp', y='sentiment_score', data=sentiment_data, ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)

# Section 3: Election Outcome Prediction
st.header("🔮 Predict Election Outcomes")

if st.button("Predict Future Vote Share"):
    X = sentiment_data[['positive_sentiment', 'negative_sentiment', 'neutral_sentiment']]
    y = sentiment_data['vote_percentage']
    
    model = LinearRegression()
    model.fit(X, y)
    
    future_predictions = model.predict(X)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sentiment_data['timestamp'], y, label="Actual Votes", linestyle="--")
    ax.plot(sentiment_data['timestamp'], future_predictions, label="Predicted Votes", color='red')
    plt.legend()
    plt.xticks(rotation=45)
    plt.title("Election Outcome Prediction")
    st.pyplot(fig)

st.write("This model predicts election outcomes based on sentiment trends.")
