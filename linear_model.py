import csv
import warnings

from pandas.core.common import SettingWithCopyWarning

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import json
import pickle
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import fake_news_spreader_feature_extraction as feature_extraction
from preprocessing import clean_text


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


def create_csv(idx, tweet_text, predicted_labels, true_labels, replies):
    """
        Creates csv with results for further analysis
        :return:
        """
    with open('output/' + dataset_name + '_' + 'results' + '.csv', mode='a', encoding="utf-8") as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        if idx == 0:
            writer.writerow(["idx", "tweet_text", "lr_pred", "bb_pred", "replies"])

        print(tweet_text)
        writer.writerow([idx, tweet_text, predicted_labels, true_labels, replies])


if __name__ == "__main__":

    dataset_name = 'us_elections'
    with open('dataset/replies_dataset/detailed_super_dict.json', 'r') as fp:
        data_tweet_ids = json.load(fp)

    print(len(data_tweet_ids['results']))

    predicted_labels = list()
    true_labels = list()
    idx = 0

    for item in data_tweet_ids['results']:
        features_df = pd.DataFrame(columns=['user_id', 'text', 'tweet_text'])

        tweet_id = item['tweet_id']
        user_id = item['user_id']
        text = item['text']
        instance_tweet_text = item['tweet_text']

        temp_df = pd.DataFrame({'user_id': user_id,
                                'text': text,
                                'tweet_text': instance_tweet_text}, index=[0])

        features_df = features_df.append(temp_df, ignore_index=True)

        for reply in item['replies']:
            tweet_id = reply['tweet_id']
            user_id = reply['user_id']
            text = reply['text']
            tweet_text = reply['tweet_text']

            temp_df = pd.DataFrame({'user_id': user_id,
                                    'text': text,
                                    'tweet_text': tweet_text}, index=[0])

            features_df = features_df.append(temp_df, ignore_index=True)
            features_df = features_df.drop_duplicates()

        features_df = features_df.drop(['user_id'], axis=1).reset_index(drop=True)
        features_df['user_id'] = features_df.index

        # get readability features
        data_readability = feature_extraction.get_readability_features(features_df)

        # get personality features
        data_personality = feature_extraction.get_personality_features(features_df)

        # get sentiment features
        data_sentiment = feature_extraction.get_sentiment_features(features_df)

        # apply all pre-processing steps
        features_df['text'] = features_df['text'].apply(clean_text)

        # get gender features from cleaned text
        data_gender = feature_extraction.get_gender_features(features_df)

        # get tf-idf label
        data_tfidf = feature_extraction.get_tf_idf_features(features_df[['user_id', 'text']])

        i = 0
        features = list()
        features.append(
            [data_tfidf, data_readability, data_sentiment, data_personality, data_gender])

        for feature_combination in features:
            features = pd.concat([i.set_index('user_id') for i in feature_combination], axis=1, join='outer')

            X = features

            # load fake news spreader classifier
            filename = "classifiers/fake_news/Random Forest_tfidf_readability_sentiment_personality_gender_0.8200000000000001.sav"
            clf = pickle.load(open(filename, 'rb'))

            y = clf.predict(X)
            features_df['label'] = y

            final_df = (features_df[['tweet_text', 'label']])

            print(final_df['tweet_text'])
            final_df['tweet_text'] = final_df.tweet_text.apply(clean_text)
            print(final_df['tweet_text'])

            instance_df = final_df.loc[[0]]
            final_df = final_df.iloc[1:]

            vectorizer = TfidfVectorizer(stop_words='english', min_df=0.01, max_df=0.90)

            vectors = vectorizer.fit_transform(final_df['tweet_text'])

            # save sparse tfidf vectors to dataframe to use with other features
            vectors_pd = pd.DataFrame(vectors.toarray())

            X = vectors_pd
            y = final_df['label']

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
            y_proba = clf.predict_proba(instance_vector)[0]
            no_of_replies = len(item['replies'])

            print(idx)
            print(instance_tweet_text)
            print('BB prediction:', true_label)
            print('LR prediction:', y_instance)
            print('with probability: :', y_proba[y_instance])
            print('number of replies: :', no_of_replies)

            # Print the weights assigned by the linear model for each word/feature in the instance to explain
            weights = clf.coef_

            model_weights = pd.DataFrame(
                {'features': list(X.columns), 'weights': list(weights[0] * instance_vector.toarray()[0])})
            model_weights = model_weights.reindex(model_weights['weights'].abs().sort_values(ascending=False).index)
            model_weights = model_weights[(model_weights["weights"] != 0)]

            mapping = vectorizer.vocabulary_
            inv_map = {v: k for k, v in mapping.items()}

            model_weights['words'] = model_weights['features'].map(inv_map)
            print(model_weights.head(n=20))

            create_csv(idx, instance_tweet_text, y_instance, true_label, no_of_replies)
            # final_df.to_csv('tweets_replies_labels.csv', mode='a', header=True)
            idx += 1
            predicted_labels.append(y_instance)
            true_labels.append(true_label)

    print(len(predicted_labels))
    print(predicted_labels)
    print(true_labels)

    print(accuracy_score(predicted_labels, true_labels))
