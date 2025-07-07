import praw
import datetime
import csv
import pandas as pd
import os
import configparser

# Read configuration from config.ini
config = configparser.ConfigParser()
config.read('config.ini')

# API credentials
reddit = praw.Reddit(
    client_id=config.get('Reddit', 'ClientID', fallback=''),
    client_secret=config.get('Reddit', 'ClientSecret', fallback=''),
    user_agent=config.get('Reddit', 'UserAgent', fallback='')
)

subreddit = reddit.subreddit(config.get('Reddit', 'Subreddit', fallback='Philippines'))
post = reddit.submission(url=config.get('Post', 'URL', fallback=''))

post.comments.replace_more(limit=None)

commentData = []

# Function to recursively gather comment data
def collectComment(comment):
    commentData.append({
        'Comment': comment.body,
        'Timestamp': datetime.datetime.fromtimestamp(comment.created_utc, datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
        'Ups': comment.ups,
        'Downs': comment.downs
    })
    for reply in comment.replies:
        collectComment(reply)

# Iterate through all comments
for comment in post.comments.list():
    collectComment(comment)

if len(commentData) > 0:
    df = pd.DataFrame(commentData)
    
    output_file = config.get('Output', 'CSVFile', fallback='./redditDatad.csv')
    if os.path.exists(output_file):
        df.to_csv(output_file, mode='a', header=False, index=False)
    else:
        # If the file doesn't exist, create it with headers
        df.to_csv(output_file, index=False)
    
    print(f"Added {len(commentData)} comments to {output_file}")
else:
    print("No comments found.")