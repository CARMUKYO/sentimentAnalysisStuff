import pandas as pd
from transformers import pipeline

def label_sentiment(input_csv, text_column, output_csv):
    """
    Labels a CSV file with sentiment analysis using a ternary sentiment model.

    Parameters:
        input_csv (str): Path to the input CSV file.
        text_column (str): Name of the column containing the text to analyze.
        output_csv (str): Path to save the output CSV with sentiment labels.
    """
    # Load the CSV file
    df = pd.read_csv(input_csv)
    
    # Check if the specified text column exists
    if text_column not in df.columns:
        raise ValueError(f"The specified column '{text_column}' does not exist in the input CSV.")
    
    # Load the ternary sentiment analysis model
    print("Loading sentiment analysis model...")
    sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
    
    # Define a mapping for sentiment labels
    sentiment_map = {
        "LABEL_0": "neg",   # Negative sentiment
        "LABEL_1": "neut",  # Neutral sentiment
        "LABEL_2": "pos"    # Positive sentiment
    }
    
    # Analyze sentiment and map labels
    print("Analyzing sentiment...")
    def analyze_sentiment(text):
        try:
            result = sentiment_pipeline(text[:512])[0]  # Truncate text to 512 tokens
            label = result['label']
            return sentiment_map.get(label, "neut")
        except Exception as e:
            print(f"Error processing text: {text[:50]}... - {e}")
            return "neut"
    
    df['Sentiment'] = df[text_column].astype(str).apply(analyze_sentiment)
    
    # Save the labeled CSV
    df.to_csv(output_csv, index=False)
    print(f"Labeled data saved to {output_csv}")

# Example usage
if __name__ == "__main__":
    # Replace these with your file paths and column name
    input_file = "processedtestingLang.csv"
    output_file = "labeled" + input_file 
    text_field = "text"  # Column name with text data
    
    label_sentiment(input_file, text_field, output_file)