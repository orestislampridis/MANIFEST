import json
import sys

import pymongo
import tweepy

import config as cfg

# Authorization tokens
consumer_key = cfg.consumer_key
consumer_secret = cfg.consumer_secret
access_key = cfg.access_key
access_secret = cfg.access_secret

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
