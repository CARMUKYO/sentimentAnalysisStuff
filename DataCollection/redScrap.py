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

subreddit = reddit.subreddit('Philippines')
post = reddit.submission(url='https://www.reddit.com/r/worldnews/comments/ulsmfq/philippines_marcos_maintains_huge_lead_in/')

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
    
    if os.path.exists('./redditDatad.csv'):
        df.to_csv('./redditDatad.csv', mode='a', header=False, index=False)
    else:
        # If the file doesn't exist, create it with headers
        df.to_csv('./redditDatad.csv', index=False)
    
    print(f"Added {len(commentData)} comments")
else:
    print("No comments found.")