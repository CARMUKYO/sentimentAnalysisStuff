import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.pipeline import Pipeline
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pickle
from collections import Counter

nltk.download('vader_lexicon')

def clean_text(text):
    if pd.isna(text):
        return ""
    return str(text).lower()

def standardize_label(label):
    return str(label).lower().strip()

def loadDataset(fileData):
    data = pd.read_csv(fileData)
    data['text'] = data['text'].apply(clean_text)
    data['trueSentiment'] = data['trueSentiment'].apply(standardize_label)
    return data['text'].tolist(), data['trueSentiment'].tolist()

def vaderPredict(texts, sentimentAnalyzer):
    predictions = []
    for text in texts:
        text = clean_text(text)
        if not text.strip():
            predictions.append('neutral')
            continue
            
        scores = sentimentAnalyzer.polarity_scores(text)
        compound = scores['compound']
        if compound > 0.05:
            predictions.append('positive')
        elif compound < -0.05:
            predictions.append('negative')
        else:
            predictions.append('neutral')
    return predictions

if __name__ == "__main__":
    # Load dataset
    fileData = "labeled_combined_output.csv"
    texts, labels = loadDataset(fileData)

    # Check class distribution
    label_counts = Counter(labels)
    for cls, count in label_counts.items():
        print(f"{cls}: {count} samples")

    # Split dataset
    if all(count >= 2 for count in Counter(labels).values()):
        xTrain, xTest, yTrain, yTest = train_test_split(texts, labels, test_size=0.3, random_state=42, stratify=labels)
    else:
        xTrain, xTest, yTrain, yTest = train_test_split(texts, labels, test_size=0.3, random_state=42)

    # VADER Sentiment Analysis
    sentimentAnalyzer = SentimentIntensityAnalyzer()
    vaderPreds = vaderPredict(xTest, sentimentAnalyzer)

    # Naive Bayes Model
    nbPipeline = Pipeline([
        ('vectorizer', CountVectorizer(lowercase=True)),
        ('classifier', MultinomialNB())
    ])
    nbPipeline.fit(xTrain, yTrain)
    with open('naiveBayesModel.pkl', 'wb') as f:
        pickle.dump(nbPipeline, f)

    # SVM Model
    svmPipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(lowercase=True)),
        ('classifier', svm.SVC(kernel="linear"))
    ])
    svmPipeline.fit(xTrain, yTrain)
    with open('svmModel.pkl', 'wb') as f:
        pickle.dump(svmPipeline, f)

    # Logistic Regression Model
    lrPipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(lowercase=True)),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    lrPipeline.fit(xTrain, yTrain)
    with open("logisticRegressionModel.pkl", "wb") as f:
        pickle.dump(lrPipeline, f)