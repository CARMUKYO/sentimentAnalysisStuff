import praw
import datetime
import csv
import pandas as pd
import os

# API credentials
reddit = praw.Reddit(
    client_id="CubUGYFSzyDt4IsCvp4P2g",
    client_secret="hAz7c4Mn--iFPiF4hv4VpGbsgp8mHA",
    user_agent="ElectionSentimentScraper"
)

# submission ID
subreddit = reddit.subreddit('Philippines')
post = reddit.submission(url='https://www.reddit.com/r/worldnews/comments/ulsmfq/philippines_marcos_maintains_huge_lead_in/')

post.comments.replace_more(limit=None)

# Store the comment data
comment_data = []

# Function to recursively gather comment data
def gather_comment_data(comment):
    comment_data.append({
        'Comment': comment.body,
        'Timestamp': datetime.datetime.fromtimestamp(comment.created_utc, datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
        'Ups': comment.ups,
        'Downs': comment.downs
    })
    for reply in comment.replies:
        gather_comment_data(reply)

# Iterate through all comments
for comment in post.comments.list():
    gather_comment_data(comment)

# Check if there are comments
if len(comment_data) > 0:
    # Create a DataFrame from the comment data
    df = pd.DataFrame(comment_data)
    
    # Check if the if file exist
    if os.path.exists('./redditData.csv'):
        # Append to CSV File
        df.to_csv('./redditData.csv', mode='a', header=False, index=False)
    else:
        # If the file doesn't exist, create it with headers
        df.to_csv('./redditData.csv', index=False)
    
    print(f"Added {len(comment_data)} comments to redditData.csv")
else:
    print("No comments found.")