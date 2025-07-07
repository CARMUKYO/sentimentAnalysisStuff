import asyncio
import os
import csv
import configparser
from random import randint
from twikit import Client, TooManyRequests

config = configparser.ConfigParser()
config.read('config.ini')

# Get configuration values
userAgent = config.get('Twitter', 'UserAgent', fallback='')
cookies = {
    "authToken": config.get('Twitter', 'AuthToken', fallback=''),
    "ct0": config.get('Twitter', 'CT0', fallback='')
}
query = config.get('Search', 'Query', fallback='')
tweetType = config.get('Search', 'TweetType', fallback='top')
limit = config.getint('Search', 'Limit', fallback=100)
csvFile = config.get('Output', 'CSVFile', fallback='twitterData.csv')

async def createClientWithCookies():
    return Client(
        user_agent=userAgent,
        language='en-US',
        headers={
            "Authorization": f"Bearer {cookies['authToken']}",
            "x-csrf-token": cookies["ct0"],
            "cookie": f"auth_token={cookies['authToken']}; ct0={cookies['ct0']};"
        }
    )

#Get Tweets
async def getTweet(client, query, tweetType, maximum):
    tweetCount = 0
    tweet = None

    fileExists = os.path.exists(csvFile)

    with open(csvFile, 'a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['text', 'createdAt', 'likeCount'])
        if not fileExists:
            writer.writeheader()

        while tweetCount < maximum:
            if tweet is None:
                tweet = await client.search_tweet(query, tweetType)
            else:
                await asyncio.sleep(randint(5, 15))
                tweet = await tweet.next()

            if not tweet:
                break

            for t in tweet:
                tweetCount += 1
                tweetData = {
                    'text': t.text.replace('\n', ' '),
                    'createdAt': t.created_at_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                    'likeCount': t.favorite_count,
                }
                writer.writerow(tweetData)
                print(f"Scraped Tweet #{tweetCount}: {tweetData}")

                if tweetCount >= maximum:
                    return

            if isinstance(tweet, TooManyRequests):
                await asyncio.sleep(tweet.wait_time)

async def main():
    if not all([userAgent, cookies.get('authToken'), cookies.get('ct0')]):
        print("ERROR: Please provide User Agent and Cookies in config.ini!")
        return

    client = await createClientWithCookies()
    await getTweet(client, query, tweetType, limit)

if __name__ == '__main__':
    asyncio.run(main())