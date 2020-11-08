import json
import sys

import pymongo
import tweepy

# Authorization tokens
consumer_key = 'v0xKMKsBMFN5h2WUmWTG1leh8'
consumer_secret = 'rsSy7BfKhXU61ktvbn7VF9SHbCcTNZJ65xcvYWcc8dLhzAEbuY'
access_key = '133859328-QITghlxAxmVaDJim41H7hxmDSzTUk2pusFVPc6sS'
access_secret = 'ZJUF0Enx27RYltuz2cB7ItFxhinBlZx38PinZqEvmae5T'

# Connect to mongodb
client = pymongo.MongoClient('localhost', 27017)
db = client['master_thesis_db']
collection = db['us_elections']


# StreamListener class inherits from tweepy.StreamListener and overrides on_status/on_error methods.
class StreamListener(tweepy.StreamListener):
    def on_data(self, data):
        print(data)
        tweet = json.loads(data)
        print(tweet)
        collection.insert(tweet)
        print('tweet inserted')

    def on_error(self, status_code):
        print("Encountered streaming error (", status_code, ")")
        sys.exit()

    def on_disconnect(self, notice):
        print("Disconnected with notice (", notice, ")")
        pass


if __name__ == "__main__":
    # complete authorization and initialize API endpoint
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)

    # initialize stream
    streamListener = StreamListener()
    stream = tweepy.Stream(auth=api.auth, listener=streamListener, tweet_mode='extended')
    tags = ['#USElections2020', "USElections", "Elections2020"]

    while True:
        try:
            stream.filter(languages=["en"], track=tags)
        except:
            pass
