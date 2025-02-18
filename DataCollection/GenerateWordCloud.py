import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

csv_file = 'processed_redditData.csv'
data = pd.read_csv(csv_file)

if 'text' not in data.columns:
    raise ValueError("Column 'text' not found in the CSV file. Please update the script with the correct column name.")

# Combine all comments into a single string
all_text = " ".join(data['text'].dropna().astype(str))

stopwords = set(STOPWORDS)
stopwords.update(['https', 'www', 'com', 'philippines', 'twitter', 'reddit', 'na', 'ang', 'ni', 'ako', 'election', 'sa', 'pa', 'ay', 'may', 'mga', 'yung']) 

# Create the WordCloud object
wordcloud = WordCloud(
    width=800,
    height=400,
    background_color='white',
    stopwords=stopwords,
    colormap='viridis',  
    max_words=200,  
    random_state=42  
).generate(all_text)

# Plot the word cloud
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  
plt.title('Word Cloud Reddit After Preprocessing', fontsize=16)
plt.show()

wordcloud.to_file('redditProcessedCloud.png')
