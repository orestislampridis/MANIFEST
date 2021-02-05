import json
import pickle
import re

import numpy as np
import pandas as pd
import tweepy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import config as cfg
import fake_news_spreader_feature_extraction as feature_extraction
from Twitter_API import TwitterAPI
from preprocessing import clean_text


def upgrade_to_work_with_single_class(SklearnPredictor):
    class UpgradedPredictor(SklearnPredictor):
        def __init__(self, *args, **kwargs):
            self._single_class_label = None
            super().__init__(*args, **kwargs)

        @staticmethod
        def _has_only_one_class(y):
            return len(np.unique(y)) == 1

        def _fitted_on_single_class(self):
            return self._single_class_label is not None

        def fit(self, X, y=None):
            if self._has_only_one_class(y):
                self._single_class_label = y[0]
            else:
                super().fit(X, y)
            return self

        def predict(self, X):
            if self._fitted_on_single_class():
                return np.full(X.shape[0], self._single_class_label)
            else:
                return super().predict(X)

    return UpgradedPredictor


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

    with open('detailed_super_dict.json', 'r') as fp:
        data_tweet_ids = json.load(fp)

    """
    print(len(data_tweet_ids))
    seen = []

    for k, val in list(data_tweet_ids.items()):
        if val in seen:
            del data_tweet_ids[k]
        else:
            seen.append(val)

    for x in data_tweet_ids.items():
        print(x)

    """
    print(len(data_tweet_ids['results']))

    predicted_labels = list()
    true_labels = list()

    for item in data_tweet_ids['results']:
        features_df = pd.DataFrame(columns=['user_id', 'text', 'tweet_text'])

        tweet_id = item['tweet_id']
        user_id = item['user_id']
        text = item['text']
        tweet_text = item['tweet_text']

        print(user_id)
        print(text)
        print(tweet_text)

        temp_df = pd.DataFrame({'user_id': user_id,
                                'text': text,
                                'tweet_text': tweet_text}, index=[0])

        features_df = features_df.append(temp_df, ignore_index=True)

        for reply in item['replies']:
            tweet_id = reply['tweet_id']
            user_id = reply['user_id']
            text = reply['text']
            tweet_text = reply['tweet_text']

            print(user_id)
            print(text)
            print(tweet_text)

            temp_df = pd.DataFrame({'user_id': user_id,
                                    'text': text,
                                    'tweet_text': tweet_text}, index=[0])

            features_df = features_df.append(temp_df, ignore_index=True)
            features_df = features_df.drop_duplicates()

        print(features_df)

        # count various readability features
        data_readability = feature_extraction.get_readability_features(features_df)

        # get personality features
        data_personality = feature_extraction.get_personality_features(features_df)

        # convert to lower and remove punctuation or special characters
        features_df['text'] = features_df['text'].str.lower()
        features_df['text'] = features_df['text'].apply(cleanPunc)
        features_df['text'] = features_df['text'].apply(clean_relics)

        # get sentiment features from cleaned text
        data_sentiment = feature_extraction.get_sentiment_features(features_df)

        # get gender features from cleaned text
        data_gender = feature_extraction.get_gender_features(features_df)

        data_tfidf = feature_extraction.get_tf_idf_features(features_df[['user_id', 'text']])

        i = 0
        features = list()
        features.append(
            [data_tfidf, data_readability, data_sentiment, data_personality, data_gender])

        for feature_combination in features:
            features = pd.concat([i.set_index('user_id') for i in feature_combination], axis=1, join='outer')

            X = features

            # load gender classifier
            filename = 'XGBoost_final.sav'
            clf = pickle.load(open(filename, 'rb'))

            y = clf.predict(X)
            features_df['label'] = y

            final_df = (features_df[['tweet_text', 'label']])

            print(final_df)

            final_df['tweet_text'] = final_df.tweet_text.apply(clean_text)
            final_df['tweet_text'] = [list2string(list) for list in final_df['tweet_text']]

            instance_df = final_df.loc[[0]]
            final_df = final_df.iloc[1:]

            print(instance_df)
            print(final_df)

            vectorizer = TfidfVectorizer(stop_words='english', min_df=0.01, max_df=0.90,
                                         ngram_range=(1, 3))

            vectors = vectorizer.fit_transform(final_df['tweet_text'])

            # save sparse tfidf vectors to dataframe to use with other features
            vectors_pd = pd.DataFrame(vectors.toarray())

            X = vectors_pd
            y = final_df['label']

            print(X)
            print(y)

            # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

            if len(y.unique()) != 2:
                print("All labels in same class, assigning that to prediction...")
                true_label = instance_df['label'].loc[[0]].values[0]
                y_instance = final_df['label'].to_list()[0]
                print(y_instance)
                print(true_label)
                predicted_labels.append(y_instance)
                true_labels.append(true_label)
                continue

            clf = LogisticRegression()
            clf.fit(X, y)
            # y_pred = clf.predict(X_test)

            # print("Accuracy:", accuracy_score(y_test, y_pred))

            instance = instance_df['tweet_text'].loc[[0]].values[0]
            true_label = instance_df['label'].loc[[0]].values[0]

            instance = [instance]

            instance_vector = vectorizer.transform(instance)
            y_instance = clf.predict(instance_vector)[0]
            print(y_instance)
            print(true_label)

            predicted_labels.append(y_instance)
            true_labels.append(true_label)

            # final_df.to_csv('tweets_replies_labels.csv', mode='a', header=True)

    print(len(predicted_labels))
    print(predicted_labels)
    print(true_labels)

    print(accuracy_score(predicted_labels, true_labels))
