import json

import tweepy
from tweepy import TweepError

import config as cfg
from Twitter_API import TwitterAPI

# authorization tokens
consumer_key = cfg.consumer_key
consumer_secret = cfg.consumer_secret
access_key = cfg.access_key
access_secret = cfg.access_secret

if __name__ == "__main__":
    # authorization of consumer key and consumer secret
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    # set access to user's access key and access secret
    auth.set_access_token(access_key, access_secret)
    # calling the api
    api = tweepy.API(auth)

    twitter_api = TwitterAPI()

    with open('covid_filtered_replies_tweet_ids.json', 'r') as fp:
        data_tweet_ids = json.load(fp)

    # Remove tweets that have less than 10 replies
    for index, value in list(data_tweet_ids.items()):
        if len(value) < 10 or len(value) > 200:
            del data_tweet_ids[index]

    no_of_replies_list = list()
    for index, value in data_tweet_ids.items():
        no_of_replies_list.append(len(value))

    print(no_of_replies_list)
    print(sum(no_of_replies_list) / len(no_of_replies_list))
    print(len(data_tweet_ids))

    predicted_labels = list()
    true_labels = list()

    dictionary = {
        'results': []
    }

    for index, value in data_tweet_ids.items():
        initial_tweet_id = index

        try:
            initial_user_id, initial_tweet_text = twitter_api.get_tweet_text(initial_tweet_id)
        except TypeError:
            continue

        # catch exception when user id doesn't exist
        try:
            # fetching the statuses
            statuses = api.user_timeline(user_id=initial_user_id, count=100)
        except TweepError:
            continue

        initial_100_tweets = """"""
        # printing the statuses
        for status in statuses:
            initial_100_tweets += status.text

        dictionary_replies = {
            'replies': []
        }

        for tweet_id in value:
            try:
                user_id, tweet_text = twitter_api.get_tweet_text(tweet_id)
            except TypeError:
                continue

            # print(tweet_id)
            # print(user_id)
            # print(tweet_text)

            # catch exception when user id doesn't exist
            try:
                # fetching the statuses
                statuses = api.user_timeline(user_id=user_id, count=100)
            except TweepError:
                continue

            s = """"""
            for status in statuses:
                s += status.text

            dictionary_replies['replies'].append({'tweet_id': tweet_id,
                                                  'user_id': user_id,
                                                  'tweet_text': tweet_text,
                                                  'text': s})

        dictionary['results'].append({'tweet_id': initial_tweet_id,
                                      'user_id': initial_user_id,
                                      'tweet_text': initial_tweet_text,
                                      'text': initial_100_tweets,
                                      'replies': dictionary_replies['replies']})

        print(dictionary)

    with open('covid_detailed_super_dict.json', 'w') as fp:
        json.dump(dictionary, fp)
