import pandas as pd
from transformers import pipeline

def label_sentiment(input_csv, text_column, output_csv):
    """
    Labels a CSV file with sentiment analysis using a ternary sentiment model.
    Skips rows where the sentiment is already labeled.

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
    
    # If the sentiment column doesn't exist, create it with empty values
    if 'sentiment' not in df.columns:
        df['sentiment'] = ""

    # Load the sentiment analysis model
    print("Loading sentiment analysis model...")
    sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
    
    # Define a mapping for sentiment labels
    sentiment_map = {
        "LABEL_0": "negative",
        "LABEL_1": "neutral",
        "LABEL_2": "positive"
    }
    
    # Function to analyze sentiment only if not already labeled
    def analyze_sentiment_if_needed(row):
        if pd.isna(row['sentiment']) or str(row['sentiment']).strip() == "":
            try:
                result = sentiment_pipeline(row[text_column][:512])[0]
                label = result['label']
                return sentiment_map.get(label, "neutral")
            except Exception as e:
                print(f"Error processing text: {row[text_column][:50]}... - {e}")
                return "neutral"
        else:
            return row['sentiment']  # Keep existing sentiment

    print("Analyzing sentiment where missing...")
    df['sentiment'] = df.apply(analyze_sentiment_if_needed, axis=1)
    
    # Save the updated CSV
    df.to_csv(output_csv, index=False)
    print(f"Labeled data saved to {output_csv}")

# Example usage
if __name__ == "__main__":
    input_file = "data annotate - processed2_FbData.csv"  # Changed from "combined_output.csv"
    output_file = "labeled_" + input_file
    text_field = "text"
    
    label_sentiment(input_file, text_field, output_file)