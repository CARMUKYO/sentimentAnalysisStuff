import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_predict, KFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm, metrics
from sklearn.pipeline import Pipeline
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('punkt')
nltk.download('vader_lexicon')

def clean_text(text):
    if pd.isna(text):
        return ""
    return str(text)

def calculateMetrics(yTrue, yPred):
    classes = ['pos', 'neg', 'neut']
    metrics_dict = {'accuracy': metrics.accuracy_score(yTrue, yPred)}
    
    # Calculate per-class metrics
    for cls in classes:
        tp = sum((yt == yp == cls) for yt, yp in zip(yTrue, yPred))
        fp = sum((yt != cls and yp == cls) for yt, yp in zip(yTrue, yPred))
        fn = sum((yt == cls and yp != cls) for yt, yp in zip(yTrue, yPred))
        tn = sum((yt != cls and yp != cls) for yt, yp in zip(yTrue, yPred))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        metrics_dict[cls] = {
            'precision': precision,
            'recall': recall,
            'f1-score': f1,
            'specificity': specificity,
            'confusion_matrix': {
                'TP': tp,
                'TN': tn,
                'FP': fp,
                'FN': fn
            }
        }
        
        print(f"\n{cls.upper()} Metrics:")
        print(f"True Positives: {tp}")
        print(f"True Negatives: {tn}")
        print(f"False Positives: {fp}")
        print(f"False Negatives: {fn}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"Specificity: {specificity:.3f}")
        print(f"F1-Score: {f1:.3f}")
    
    print(f"\nOverall Accuracy: {metrics_dict['accuracy']:.3f}")
    return metrics_dict

# Load dataset
def loadDataset(fileData):
    data = pd.read_csv(fileData)
    data['text'] = data['text'].apply(clean_text)
    return data['text'].tolist(), data['label'].tolist()

fileData = "labeledprocessedtestingLang.csv"
texts, labels = loadDataset(fileData)

# Split dataset into training and testing sets
xTrain, xTest, yTrain, yTest = train_test_split(texts, labels, test_size=0.3, random_state=42, stratify=labels)

# VADER Lexicon
print("\nPerforming VADER Sentiment Analysis...")
sentimentAnalyzer = SentimentIntensityAnalyzer()

def vaderPredict(texts):
    predictions = []
    for text in texts:
        text = clean_text(text)
        if not text.strip():
            predictions.append('neut')
            continue
            
        scores = sentimentAnalyzer.polarity_scores(text)
        compound = scores['compound']
        if compound > 0.05:
            predictions.append('pos')
        elif compound < -0.05:
            predictions.append('neg')
        else:
            predictions.append('neut')
    return predictions

vaderPreds = vaderPredict(xTest)
print("VADER Results:")
vaderReport = calculateMetrics(yTest, vaderPreds)

# NVM
print("\nTraining Naive Bayes Model...")
nbPipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])
nbPipeline.fit(xTrain, yTrain)
nbPreds = nbPipeline.predict(xTest)
print("Naive Bayes Results:")
nbReport = calculateMetrics(yTest, nbPreds)

# SVM
print("\nTraining SVM Model...")
svmPipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', svm.SVC(kernel="linear"))
])
svmPipeline.fit(xTrain, yTrain)
svmPreds = svmPipeline.predict(xTest)
print("SVM Results:")
svmReport = calculateMetrics(yTest, svmPreds)

# LGR
print("\nTraining Logistic Regression Model...")
lrPipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])
lrPipeline.fit(xTrain, yTrain)
lrPreds = lrPipeline.predict(xTest)
print("Logistic Regression Results:")
lrReport = calculateMetrics(yTest, lrPreds)

# Save LRG
import pickle
with open("logisticRegressionModel.pkl", "wb") as f:
    pickle.dump(lrPipeline, f)
print("\nLogistic Regression Model saved as 'logisticRegressionModel.pkl'.")

predictionsDf = pd.DataFrame({
    'text': xTest,
    'trueSentiment': yTest,
    'vaderPrediction': vaderPreds,
    'nbPrediction': nbPreds,
    'svmPrediction': svmPreds,
    'lrPrediction': lrPreds
})
predictionsFile = "predictions" + fileData
predictionsDf.to_csv(predictionsFile, index=False)

# Bar graph 
def plotModelMetrics(nbReport, svmReport, lrReport, vaderReport):
    metricsLabels = ['accuracy']
    classes = ['pos', 'neg', 'neut']
    metrics_types = ['precision', 'recall', 'f1-score']
    
    for cls in classes:
        for metric in metrics_types:
            metricsLabels.append(f'{cls}_{metric}')
    
    models = ['Naive Bayes', 'SVM', 'Logistic Regression', 'VADER']
    
    def getMetricsList(report):
        metrics_list = [report['accuracy']]
        for cls in classes:
            for metric in metrics_types:
                metrics_list.append(report[cls][metric])
        return metrics_list
    
    nbMetrics = getMetricsList(nbReport)
    svmMetrics = getMetricsList(svmReport)
    lrMetrics = getMetricsList(lrReport)
    vaderMetrics = getMetricsList(vaderReport)
    
    data = np.array([nbMetrics, svmMetrics, lrMetrics, vaderMetrics])
    
    x = np.arange(len(metricsLabels))
    width = 0.15
    
    fig, ax = plt.subplots(figsize=(15, 7))
    
    ax.bar(x - 1.5*width, data[0], width, label='Naive Bayes')
    ax.bar(x - 0.5*width, data[1], width, label='SVM')
    ax.bar(x + 0.5*width, data[2], width, label='Logistic Regression')
    ax.bar(x + 1.5*width, data[3], width, label='VADER')
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Scores')
    ax.set_title('Model Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metricsLabels, rotation=45, ha='right')
    ax.legend()
    
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('modelComparison.png', dpi=300, bbox_inches='tight')
    plt.show()

plotModelMetrics(nbReport, svmReport, lrReport, vaderReport)