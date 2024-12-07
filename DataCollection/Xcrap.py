import asyncio
import os
import csv
from datetime import datetime
from random import randint
from twikit import Client, TooManyRequests

# IMPORTANT: Replace with your actual cookies
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64; rv:132.0) Gecko/20100101 Firefox/132.0"  # Replace with your user agent string
COOKIES = {
    "auth_token": "",  # Replace with your auth_token
    "ct0": ""  # Replace with your ct0 token fadfdaf
}

# Search query change leni to what keyword/phrase you want to search
QUERY = "@bongbongmarcos until:2022-02-27 since:2022-02-01 -filter:links"
TYPE = "top"  # "latest", "top"
LIMIT = 1000  # limit

# CSVfile 
CSV_FILE = 'twitterData.csv'

async def create_client_with_cookies():
    """Create a Twikit client using provided cookies"""
    try:
        client = Client(
            user_agent=USER_AGENT, 
            language='en-US', 
            headers={
                "Authorization": f"Bearer {COOKIES['auth_token']}", 
                "x-csrf-token": COOKIES["ct0"], 
                "cookie": f"auth_token={COOKIES['auth_token']}; ct0={COOKIES['ct0']};"
            }
        )
        print("Client created successfully with cookies!")
        return client
    except Exception as e:
        print(f"Failed to create client with cookies: {e}")
        raise

async def get_tweet(client, query, tweet_type, maximum):
    """Async function to scrape tweets with improved error handling"""
    tweet_count = 0
    tweet = None

    # Check if CSV file exists create if it doesn't
    file_exists = os.path.exists(CSV_FILE)
    
    # Open file in append mode
    with open(CSV_FILE, 'a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['text', 'created_at', 'like_count'])
        
        # Write if file is new
        if not file_exists:
            writer.writeheader()

    try:
        while tweet_count < maximum:
            try:
                # Search initial
                if tweet is None:
                    tweet = await client.search_tweet(query, tweet_type)
                else:
                    # random wait to for rate limits
                    wait_time = randint(5, 15)
                    await asyncio.sleep(wait_time)
                    tweet = await tweet.next()

                if not tweet:
                    break

                # Process each tweet
                for t in tweet:
                    tweet_count += 1
                    tweet_data = {
                        'text': t.text.replace('\n', ' '),  
                        'created_at': t.created_at_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                        'like_count': t.favorite_count,
                    }

                    # Append data to the CSV
                    with open(CSV_FILE, 'a', newline='', encoding='utf-8') as file:
                        writer = csv.DictWriter(file, fieldnames=['text', 'created_at', 'like_count'])
                        writer.writerow(tweet_data)
                    
                    print(f"Scraped Tweet #{tweet_count}: {tweet_data}")

                    if tweet_count >= maximum:
                        break

            except TooManyRequests as e:
                # Handle rate limiting
                wait_time = e.wait_time
                print(f"Rate limit reached. Sleeping for {wait_time} seconds.")
                await asyncio.sleep(wait_time)
                continue

    except Exception as e:
        print(f"An error occurred while scraping tweets: {e}")
    
    print(f"Total tweets scraped: {tweet_count}")

async def main():
    """Main async function to coordinate tweet scraping"""
    if not all([USER_AGENT, COOKIES.get('auth_token'), COOKIES.get('ct0')]):
        print("ERROR: Please provide User Agent and Cookies!")
        return

    try:
        client = await create_client_with_cookies()
        
        await get_tweet(client, QUERY, TYPE, LIMIT)
    
    except Exception as e:
        print(f"An error occurred in the main process: {e}")

if __name__ == '__main__':
    # Use asyncio.run with the main coroutine
    asyncio.run(main())