import joblib
import asyncio
import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn import metrics
from random import randint
from twikit import Client, TooManyRequests

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
        print(f"Warning: {modelPath} not found! Make sure the model is trained and available.")
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
            predictions.append("pos")
        elif score <= -0.05:
            predictions.append("neg")
        else:
            predictions.append("neut")
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
    classes = ['pos', 'neg', 'neut']
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

userAgent = ""
cookies = {
    "authToken": "",
    "ct0": ""
}

#Cookie load
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