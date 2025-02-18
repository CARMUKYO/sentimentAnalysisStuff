import pandas as pd
import re
from langdetect import detect, DetectorFactory
import nltk
from tqdm import tqdm
import os
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import logging

# Set seed for langdetect consistency
DetectorFactory.seed = 42

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def downloadNltkData():
    resources = ['vader_lexicon']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            logging.error(f"Failed to download {resource}: {e}")

downloadNltkData()

def createWordcloud(data, column, title, outputFile):
    allText = " ".join(data[column].dropna().astype(str))
    
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='viridis',
        max_words=200,
        random_state=42
    ).generate(allText)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=16)
    plt.savefig(outputFile)
    
    return wordcloud, plt.gcf()

def displayWordclouds(fig1, fig2):
    plt.show()

class TextPreprocessor:
    def __init__(self):
        pass
    
    # Remove URLs, etc., but keep emojis
    def replaceTags(self, text):
        if not isinstance(text, str):
            return ""
        text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
        text = re.sub(r'@\w+|#\w+', '', text)  
        text = re.sub(r'[^\w\s.,!?\'"🙂🙁😂😭😍😡]', '', text)  # Keep punctuation & emojis
        text = ' '.join(text.split())  # Remove extra whitespace
        return text
    
    # Normalize elongated characters
    def normalizeElongatedChars(self, text):
        if not isinstance(text, str):
            return ""
        return re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    def expandContractions(self, text):
        contractions = {
            "can't": "cannot", "won't": "will not", "n't": " not",
            "i'm": "i am", "he's": "he is", "she's": "she is", "it's": "it is",
            "that's": "that is", "there's": "there is", "they're": "they are",
            "i've": "i have", "you've": "you have", "we've": "we have", "they've": "they have",
            "isn't": "is not", "aren't": "are not", "wasn't": "was not", "weren't": "were not",
            "don't": "do not", "doesn't": "does not", "didn't": "did not", "haven't": "have not",
            "hasn't": "has not", "hadn't": "had not", "wouldn't": "would not", "shouldn't": "should not",
            "couldn't": "could not", "mightn't": "might not", "mustn't": "must not"
        }
        for contraction, full_form in contractions.items():
            text = re.sub(contraction, full_form, text, flags=re.IGNORECASE)
        return text
    
    def isRelevant(self, text):
        """Detect if text is in Filipino or English and filter out ads."""
        ad_keywords = ["buy now", "limited offer", "discount", "promo", "subscribe", "order now",
                       "vote for", "campaign", "election", "candidate", "poll", "ballot", "rally"]
        
        if not isinstance(text, str) or not text.strip():
            return False
        
        try:
            lang = detect(text)
            if lang not in ['en', 'tl']:
                return False
        except:
            return False
        
        if any(keyword in text.lower() for keyword in ad_keywords):
            return False
        
        return True
    
    def preprocessText(self, text):
        if not isinstance(text, str):
            return ""
        
        # 1: Expand contractions
        textExpanded = self.expandContractions(text)
        # 2: Replace tags (remove URLs, mentions, etc.)
        textNoTags = self.replaceTags(textExpanded)
        # 3: Normalize elongated characters
        normalizedText = self.normalizeElongatedChars(textNoTags)
        # 4: Case transformation
        lowercaseText = normalizedText.lower()
        
        return lowercaseText.strip()

def processCsv(inputFile, outputFile, textColumn='text'):
    logging.info(f"Reading data from {inputFile}...")
    df = pd.read_csv(inputFile)

    if textColumn not in df.columns:
        raise ValueError(f"Column '{textColumn}' not found in CSV file.")

    baseFilename = os.path.splitext(inputFile)[0]
    originalWordcloudFile = f'{baseFilename}OriginalWordcloud.png'
    
    _, figOriginal = createWordcloud(
        df, 
        textColumn, 
        f'Word Cloud - Original {baseFilename} Data', 
        originalWordcloudFile
    )
    
    preprocessor = TextPreprocessor()

    logging.info("Filtering and processing text...")
    dfProcessed = df.copy()
    dfProcessed[textColumn] = df[textColumn].astype(str)
    
    dfProcessed = dfProcessed[dfProcessed[textColumn].apply(preprocessor.isRelevant)]
    dfProcessed.drop_duplicates(subset=[textColumn], inplace=True)
    dfProcessed[textColumn] = dfProcessed[textColumn].apply(preprocessor.preprocessText)
    dfProcessed = dfProcessed[dfProcessed[textColumn] != ""]  # Remove empty rows after preprocessing

    processedWordcloudFile = f'{baseFilename}PreprocessedWordcloud.png'
    _, figProcessed = createWordcloud(
        dfProcessed, 
        textColumn, 
        f'Word Cloud - Processed {baseFilename} Data', 
        processedWordcloudFile
    )

    logging.info(f"Saving processed data to {outputFile}...")
    dfProcessed.to_csv(outputFile, index=False)
    logging.info("Processing complete!")
    
    displayWordclouds(figOriginal, figProcessed)

if __name__ == "__main__":
    inputFile = input("Enter the path of the input CSV file: ").strip()
    outputFile = "processed2_" + inputFile
    textColumn = "text"
    
    processCsv(inputFile, outputFile, textColumn)