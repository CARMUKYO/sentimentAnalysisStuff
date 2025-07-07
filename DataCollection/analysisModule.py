import joblib
import asyncio
import os
import csv
import pandas as pd
import re
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import configparser
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn import metrics
from random import randint
from twikit import Client, TooManyRequests
from nltk.corpus import stopwords
from itertools import combinations
from collections import Counter

# Ensure VADER is downloaded
import nltk
nltk.download("vader_lexicon")
# Initialize VADER
vaderAnalyzer = SentimentIntensityAnalyzer()

# Load trained models
def loadModel(modelPath):
    if os.path.exists(modelPath):
        return joblib.load(modelPath)
    else:
        return None

# Load all models
nbModel = loadModel("naiveBayesModel.pkl")
svmModel = loadModel("svmModel.pkl")
lrModel = loadModel("logisticRegressionModel.pkl")

# Function to predict sentiment using ML models
def predictSentimentMl(model, textData):
    if model:
        return model.predict(textData)
    else:
        return ["N/A"] * len(textData)  # Return "N/A" if the model is missing

# Function to predict sentiment using VADER
def predictSentimentVader(textData):
    predictions = []
    for text in textData:
        score = vaderAnalyzer.polarity_scores(text)["compound"]
        if score >= 0.05:
            predictions.append("positive")
        elif score <= -0.05:
            predictions.append("negative")
        else:
            predictions.append("neutral")
    return predictions

# Process CSV File
def processCsv(inputCsv, outputCsv="processed_sentiment.csv"):
    df = pd.read_csv(inputCsv)
    if "text" not in df.columns:
        print("Error: 'text' column not found in CSV!")
        return
    textData = df["text"].astype(str)
    
    # Get predictions from all models
    df["vaderSentiment"] = predictSentimentVader(textData)
    df["naiveBayesSentiment"] = predictSentimentMl(nbModel, textData)
    df["svmSentiment"] = predictSentimentMl(svmModel, textData)
    df["logisticRegressionSentiment"] = predictSentimentMl(lrModel, textData)
    
    # Calculate metrics for each model
    if "true_sentiment" in df.columns:  # Ensure you have ground truth labels
        metrics_vader = calculateMetrics(df["true_sentiment"], df["vaderSentiment"])
        metrics_nb = calculateMetrics(df["true_sentiment"], df["naiveBayesSentiment"])
        metrics_svm = calculateMetrics(df["true_sentiment"], df["svmSentiment"])
        metrics_lr = calculateMetrics(df["true_sentiment"], df["logisticRegressionSentiment"])
        
        # Store metrics in a dictionary for visualization
        all_metrics = {
            "VADER": metrics_vader,
            "Naive Bayes": metrics_nb,
            "SVM": metrics_svm,
            "Logistic Regression": metrics_lr,
        }
        
        print(all_metrics)
    
    # Save processed CSV
    df.to_csv(outputCsv, index=False)
    print(f"Sentiment analysis completed! Results saved to {outputCsv}")

# Single Text Sentiment Analysis
def analyzeText(text):
    results = {
        "vaderSentiment": predictSentimentVader([text])[0],
        "naiveBayesSentiment": predictSentimentMl(nbModel, [text])[0],
        "svmSentiment": predictSentimentMl(svmModel, [text])[0],
        "logisticRegressionSentiment": predictSentimentMl(lrModel, [text])[0],
    }
    return results

#Metric Calculation
def calculateMetrics(yTrue, yPred):
    classes = ['positive', 'negative', 'neutral']
    metrics_dict = {'accuracy': metrics.accuracy_score(yTrue, yPred)}  # Overall accuracy

    for cls in classes:
        tp = sum((yt == yp == cls) for yt, yp in zip(yTrue, yPred))
        fp = sum((yt != cls and yp == cls) for yt, yp in zip(yTrue, yPred))
        fn = sum((yt == cls and yp != cls) for yt, yp in zip(yTrue, yPred))
        tn = sum((yt != cls and yp != cls) for yt, yp in zip(yTrue, yPred))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Store per-class metrics
        metrics_dict[cls] = {
            'precision': precision,
            'recall': recall,
            'f1-score': f1,
            'confusion_matrix': {
                'TP': tp,
                'TN': tn,
                'FP': fp,
                'FN': fn
            }
        }

    return metrics_dict

config = configparser.ConfigParser()
config.read("config.ini")

userAgent = config.get('Twitter', 'UserAgent', fallback='')
cookies = {
    "authToken": config.get('Twitter', 'AuthToken', fallback=''),
    "ct0": config.get('Twitter', 'CT0', fallback='')
}

# Load Cookie
async def createClientWithCookies():
    return Client(
        user_agent=userAgent,
        language='en-US',
        headers={
            "Authorization": f"Bearer {cookies['authToken']}",
            "x-csrf-token": cookies["ct0"],
            "cookie": f"auth_token={cookies['authToken']}; ct0={cookies['ct0']};"
        }
    )

# Function to create a word network graph
def create_word_network(texts, num_words=75, min_edge_weight=2):
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    
    stop_words = set(stopwords.words('english'))
    custom_stopwords = {"na", "ng", "https", "www", "lang", "mga", "ka", "ang", "may", 
                        "yung", "yan", "nga", "com", "kung", "pag"}
    stop_words.update(custom_stopwords)
    
    # Extract words from all texts and count their frequency
    all_words = []
    for text in texts:
        if isinstance(text, str):
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            filtered_words = [word for word in words if word not in stop_words]
            all_words.extend(filtered_words)
    
    word_counts = Counter(all_words)
    top_words = [word for word, count in word_counts.most_common(num_words)]
    
    G = nx.Graph()
    
    # Process each text to find co-occurrences between top words
    edge_weights = {}
    for text in texts:
        if not isinstance(text, str):
            continue
        
        text_words = [word.lower() for word in re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
                     if word.lower() in top_words]
        
        # Count co-occurrences (pairs of words that appear in the same text)
        for word1, word2 in combinations(set(text_words), 2):
            if word1 != word2:
                edge_key = tuple(sorted([word1, word2]))
                edge_weights[edge_key] = edge_weights.get(edge_key, 0) + 1
    
    # Add nodes with their frequency as a property
    for word in top_words:
        G.add_node(word, size=word_counts[word])
    
    for (word1, word2), weight in edge_weights.items():
        if weight >= min_edge_weight:
            G.add_edge(word1, word2, weight=weight)
    
    return G, word_counts

#Function to Scrape
async def scrapeTweets(query, limit=100, csvFile="ScrapeData.csv"):
    client = await createClientWithCookies()
    tweetCount = 0
    tweet = None

    # Open the file in write mode to overwrite existing data
    with open(csvFile, 'w', newline='', encoding='utf-8') as file: 
        writer = csv.DictWriter(file, fieldnames=['text', 'createdAt', 'likeCount'])
        writer.writeheader()  # Always write the header

        while tweetCount < limit:
            if tweet is None:
                tweet = await client.search_tweet(query, "top")
            else:
                await asyncio.sleep(randint(5, 15))
                tweet = await tweet.next()

            if not tweet:
                break

            for t in tweet:
                tweetCount += 1
                tweetData = {
                    'text': t.text.replace('\n', ' '),
                    'createdAt': t.created_at_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                    'likeCount': t.favorite_count,
                }
                writer.writerow(tweetData)
                print(f"Scraped Tweet #{tweetCount}: {tweetData}")

                if tweetCount >= limit:
                    return

            if isinstance(tweet, TooManyRequests):
                await asyncio.sleep(tweet.wait_time)

def runScraper(query, limit):
    asyncio.run(scrapeTweets(query, limit, csvFile="ScrapeData.csv"))