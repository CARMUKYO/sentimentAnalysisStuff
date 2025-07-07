import facebook_scraper as fs
import pandas as pd
import emoji
import os
import time
import random
import configparser

# Read configuration from config.ini
config = configparser.ConfigParser()
config.read('config.ini')

# IMPORTANT: Replace with your actual cookies
cookies = {
    'c_user': config.get('Facebook', 'CUser', fallback=''),
    'xs': config.get('Facebook', 'XS', fallback=''),
    'fr': config.get('Facebook', 'FR', fallback='')
}

# post ID
postId = config.get('Post', 'ID', fallback='')
maxComments = config.getboolean('Post', 'MaxComments', fallback=True)

gen = fs.get_posts(
    post_urls=[postId],
    cookies=cookies,
    options={"comments": maxComments, "progress": True}
)

post = next(gen)
commentsData = post['comments_full']

comment_data = []

# Function to recursively gather comment data
def gather_comment_data(comment):
    comment_data.append({
        'Comment': emoji.emojize(comment['comment_text']),
        'Timestamp': comment['comment_time'],
        'Likes': comment['comment_reaction_count']
    })

    for reply in comment.get('replies', []):
        gather_comment_data(reply)

# Gather data for all comments and their replies
for comment in commentsData:
    gather_comment_data(comment)

# Check if there are comments
if len(comment_data) > 0:
    df = pd.DataFrame(comment_data)
    
    output_file = config.get('Output', 'CSVFile', fallback='./FbData.csv')
    if os.path.exists(output_file):
        df.to_csv(output_file, mode='a', header=False, index=False)
        print(f"Added {len(comment_data)} comments to {output_file}.")
    else:
        df.to_csv(output_file, index=False)
        print(f"Created {output_file} and added {len(comment_data)} comments.")
else:
    print("No comments found.")