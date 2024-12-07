import facebook_scraper as fs
import pandas as pd
import emoji
import os
import time
import random 

# IMPORTANT: Replace with your actual cookies
cookies = {
    'c_user': '',
    'xs': '',
    'fr': ''
}

# post ID
postId = "pfbid0dxHUENDCkrRwK91johoGZFwVua1ZbVwjsaPsD8Vmk4pFP56d9qMeuBghjtxPTVJPl"
maxComments = True

# Initialize scraper with cookies
gen = fs.get_posts(
    post_urls=[postId],
    cookies=cookies,
    options={"comments": maxComments, "progress": True}
)

# Extract the post and its comments
post = next(gen)
commentsData = post['comments_full']

comment_data = []

# Function to recursively gather comment data
def gather_comment_data(comment):
    """Flatten the comment and reply hierarchy into a single list."""
    comment_data.append({
        'Comment': emoji.emojize(comment['comment_text']),
        'Timestamp': comment['comment_time'],
        'Likes': comment['comment_reaction_count']
    })
    # Rate limit
    delay = random.uniform(1, 2)  # Random float between 1 and 2 seconds
    time.sleep(delay)

    for reply in comment.get('replies', []):
        gather_comment_data(reply)

# Gather data for all comments and their replies
for comment in commentsData:
    gather_comment_data(comment)

# Check if there are comments
if len(comment_data) > 0:
    # Convert the comment data to a DataFrame
    df = pd.DataFrame(comment_data)
    
    # Check if the file  exists
    if os.path.exists('./FbData.csv'):
        # Append to CSV file
        df.to_csv('./FbData.csv', mode='a', header=False, index=False)
        print(f"Added {len(comment_data)} comments to {'./FbData.csv'}.")
    else:
        # If the file doesn't exist, create it with headers
        df.to_csv('./FbData.csv', index=False)
        print(f"Created {'./FbData.csv'} and added {len(comment_data)} comments.")
else:
    print("No comments found.")