import facebook_scraper as fs
import pandas as pd
import emoji
import os
import time
import random 

# IMPORTANT: Replace with your actual cookies
cookies = {
    'c_user': '100010497515883',
    'xs': '9:3B7oXdgwFafyqg:2:1737153581:-1:8175',
    'fr': '1JWykjW6xcCH19iwK.AWWNdSjV2JRLbKTRri09utPiGOM.BnildJ..AAA.0.0.BniuYS.AWW--Y0cabU'
}

# post ID
postId = "pfbid0ExJimUfa4K4bjq1CE5KrEaNKvdGGS3yJJMhfagsTKT8RhxsJjoMgrGDDjW6zyyDtl"
maxComments = True

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
    """Flatten the comment and reply hierarchy into a single list."""
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
    
    if os.path.exists('./FbData.csv'):
        df.to_csv('./FbData.csv', mode='a', header=False, index=False)
        print(f"Added {len(comment_data)} comments to {'./FbData.csv'}.")
    else:
        df.to_csv('./FbData.csv', index=False)
        print(f"Created {'./FbData.csv'} and added {len(comment_data)} comments.")
else:
    print("No comments found.")