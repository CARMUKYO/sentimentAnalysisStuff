import asyncio
import os
import csv
from random import randint
from twikit import Client, TooManyRequests

# IMPORTANT: Replace with your actual cookies
userAgent = "Mozilla/5.0 (X11; Linux x86_64; rv:132.0) Gecko/20100101 Firefox/132.0"  # Replace with your user agent string
cookies = {
    "authToken": "d6753577a055836e42e4647286ec62efa6b6f00f",  # Replace with your auth_token
    "ct0": "833c47a3c0583d8a2331b0e28c4b946cf7d2825e2f05704756da7a24ad2f2e93abaf4821f9d0f3d1f7313a53896dd19725de32e9b6eaef1881cc4cd6ede0b01c0d2d2a115427e3a801ab16af516b5b61"  # Replace with your ct0 token
}
query = "bongbong marcos leni ph until:2022-05-09 since:2022-05-08 -filter:links"
tweetType = "top"
limit = 110
csvFile = 'twitterData.csv'

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

# Function to scrape tweets
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
        print("ERROR: Please provide User Agent and Cookies!")
        return

    client = await createClientWithCookies()
    await getTweet(client, query, tweetType, limit)

if __name__ == '__main__':
    asyncio.run(main())