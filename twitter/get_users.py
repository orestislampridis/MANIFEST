import json
import pickle
import re

import pandas as pd
import tweepy
from tweepy import TweepError

import config as cfg
import fake_news_spreader_feature_extraction as feature_extraction
from Twitter_API import TwitterAPI
from preprocessing import clean_text

# authorization tokens
consumer_key = cfg.consumer_key
consumer_secret = cfg.consumer_secret
access_key = cfg.access_key
access_secret = cfg.access_secret


def list2string(list):
    return ','.join(map(str, list))


def cleanPunc(sentence):
    cleaned = re.sub(r'[?|!|\'|"|#]', r'', sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]', r' ', cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n", " ")
    return cleaned


# function to clean relics of dataset
def clean_relics(text):
    text = re.sub(r"RT", "", text)
    text = re.sub(r"#USER#", "", text)
    text = re.sub(r"#HASHTAG#", "", text)
    text = re.sub(r"#URL#", "", text)
    return text


if __name__ == "__main__":
    # authorization of consumer key and consumer secret
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    # set access to user's access key and access secret
    auth.set_access_token(access_key, access_secret)
    # calling the api
    api = tweepy.API(auth)

    twitter_api = TwitterAPI()

    with open('replies_user_ids.json', 'r') as fp:
        data_user_ids = json.load(fp)
    with open('replies_tweet_ids.json', 'r') as fp:
        data_tweet_ids = json.load(fp)

    mydict = {
        'user_ids': data_user_ids,
        'tweet_ids': data_tweet_ids}

    filtered_dict = dict()

    for index, value in list(mydict.items()):
        for index_, ids in list(value.items()):
            # Keep only posts with at least n comments
            if len(ids) < 15:
                del mydict[index][index_]

    dictionary = {
        'results': []
    }

    for (user_index, user_ids), (tweet_index, tweet_ids) in zip(mydict['user_ids'].items(),
                                                                mydict['tweet_ids'].items()):
        dictionary['results'].append({'tweet_id': tweet_index,
                                      'user_id': user_index,
                                      'reply_tweet_ids': tweet_ids,
                                      'reply_user_ids': user_ids})

    for item in dictionary['results']:

        df = pd.DataFrame(columns=['user_id', 'text', 'tweet_text'])

        # print(item['tweet_id'])
        # print(item['user_id'])
        # print(item['reply_tweet_ids'])
        # print(item['reply_user_ids'])

        tweet_text = twitter_api.get_tweet_text(item['tweet_id'])

        # catch exception when user id doesn't exist
        try:
            # fetching the statuses
            statuses = api.user_timeline(user_id=item['user_id'], count=100)
        except TweepError:
            continue

        s = """"""
        # printing the statuses
        for status in statuses:
            s += status.text

        temp_df = pd.DataFrame({'user_id': item['user_id'],
                                'text': s,
                                'tweet_text': tweet_text}, index=[0])

        df = df.append(temp_df, ignore_index=True)

        for user_id, tweet_id in zip(item['reply_user_ids'], item['reply_tweet_ids']):

            tweet_text = twitter_api.get_tweet_text(tweet_id)

            # print(user_id)
            # print(tweet_id)
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

            temp_df = pd.DataFrame({'user_id': user_id,
                                    'text': s,
                                    'tweet_text': tweet_text}, index=[0])

            df = df.append(temp_df, ignore_index=True)
            df = df.drop_duplicates()

        data_readability = feature_extraction.get_readability_features(df)

        df['text'] = df.text.apply(clean_text)
        df['text'] = [list2string(list) for list in df['text']]

        data_tfidf = feature_extraction.get_tfidf_vectors_from_pickle(df[['user_id', 'text']])

        i = 0
        features = list()

        features.append([data_tfidf, data_readability])

        for feature_combination in features:
            features = pd.concat([i.set_index('user_id') for i in feature_combination], axis=1, join='outer')

            X = features

            # load gender classifier
            filename = 'XGBoost_final.sav'
            clf = pickle.load(open(filename, 'rb'))

            y = clf.predict(X)
            df['label'] = y

            final_df = (df[['tweet_text', 'label']])
            final_df.to_csv('tweets_replies_labels.csv', mode='a', header=True)
