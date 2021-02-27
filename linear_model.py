import csv
import json
import pickle
import warnings

import pandas as pd
from pandas.core.common import SettingWithCopyWarning
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity

import fake_news_spreader_feature_extraction as feature_extraction
from utils.preprocessing import clean_text, clean_text_lm

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


def create_homophily_csv(idx, tweet_text, predicted_labels, true_labels, replies):
    """
        Creates csv with results only for those that have all the same label in the replies
        :return:
        """
    with open('output/' + dataset_name + '_' + latent_representation + '_' + 'lalala_homophily_final_results' + '.csv',
              mode='a', encoding="utf-8") as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        if idx == 0:
            writer.writerow(["idx", "tweet_text", "homophily_pred", "bb_pred", "replies"])

        writer.writerow([idx, tweet_text, predicted_labels, true_labels, replies])


def create_csv(idx, tweet_text, predicted_labels, true_labels, replies):
    """
        Creates csv with results for further analysis
        :return:
        """
    with open('output/' + dataset_name + '_' + latent_representation + '_' + 'lalala_final_results' + '.csv', mode='a',
              encoding="utf-8") as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        if idx == 0:
            writer.writerow(["idx", "tweet_text", "lr_pred", "bb_pred", "replies"])

        writer.writerow([idx, tweet_text, predicted_labels, true_labels, replies])


if __name__ == "__main__":

    # us_elections or oovid
    dataset_name = 'us_elections'
    latent_representation = 'tf-idf'
    with open('dataset/replies_dataset/' + dataset_name + '_detailed_super_dict.json', 'r') as fp:
        data_tweet_ids = json.load(fp)

    print(len(data_tweet_ids['results']))

    predicted_labels = list()
    true_labels = list()
    bb_probas = list()
    lr_probas = list()
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

        features_df.rename(
            columns={'user_id': 'tweet_user_id'}, inplace=True)
        # features_df = features_df.drop(['user_id'], axis=1).reset_index(drop=True)
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
        data_tfidf = feature_extraction.get_tfidf_vectors_from_pickle(features_df[['user_id', 'text']])

        i = 0
        features = list()
        features.append(
            [data_tfidf, data_readability, data_sentiment, data_personality, data_gender])

        for feature_combination in features:
            features = pd.concat([i.set_index('user_id') for i in feature_combination], axis=1, join='outer')

            X = features

            # load fake news spreader classifier
            filename = "classifiers/fake_news/Gradient Boosting_phase_C_tfidf_readability_sentiment_personality_gender_0.7300000000000001.sav"
            fake_news_clf = pickle.load(open(filename, 'rb'))

            y = fake_news_clf.predict(X)
            features_df['label'] = y

            initial_df = (features_df[['tweet_user_id', 'tweet_text', 'label']])
            final_df = (features_df[['tweet_user_id', 'tweet_text', 'label']])

            final_df['tweet_text'] = final_df.tweet_text.apply(clean_text_lm)

            instance_df = final_df.loc[[0]]
            replies_df = final_df.iloc[1:]
            unprocessed_replies_list = initial_df.iloc[1:]['tweet_text'].to_list()

            no_of_replies = len(item['replies'])

            if latent_representation == 'bow':
                vectorizer = CountVectorizer(stop_words='english')
            elif latent_representation == 'tf-idf':
                vectorizer = TfidfVectorizer(stop_words='english')

            vectors = vectorizer.fit_transform(replies_df['tweet_text'])

            # save sparse tfidf vectors to dataframe to use with other features
            vectors_pd = pd.DataFrame(vectors.toarray())

            X = vectors_pd
            y = replies_df['label']

            # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

            if len(y.unique()) != 2:
                print("All labels in same class, assigning that to prediction...")
                true_label = instance_df['label'].loc[[0]].values[0]  # based on fake news spreader classifier
                y_assigned = replies_df['label'].to_list()[0]
                # print(y_assigned)
                # print(true_label)
                # predicted_labels.append(y_assigned)
                # true_labels.append(true_label)
                create_homophily_csv(idx, instance_tweet_text, y_assigned, true_label, no_of_replies)

                idx += 1
                continue

            lrc = LogisticRegression(solver='lbfgs', class_weight='balanced')
            lrc.fit(X, y)

            instance = instance_df['tweet_text'].loc[[0]].values[0]
            true_label = instance_df['label'].loc[[0]].values[0]

            instance = [instance]

            instance_vector = vectorizer.transform(instance)

            y_instance = lrc.predict(instance_vector)[0]
            y_proba = lrc.predict_proba(instance_vector)[0]
            bb_proba = fake_news_clf.predict_proba(features.loc[[0]])[0]

            print(idx)
            print(instance_tweet_text)
            print('BB prediction: ', true_label)
            print('with probability: ', bb_proba[1])
            print('LR prediction: ', y_instance)
            print('with probability: ', y_proba[1])
            print('number of replies: ', no_of_replies)

            instance_df['lr_label'] = y_instance
            # print(instance_df)

            # with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', None):
            #    print(replies_df)

            instance_df.to_csv('my_csv2.csv', mode='a', header=False)
            replies_df.to_csv('my_csv2.csv', mode='a', header=False)

            # Print the weights assigned by the linear model for each word/feature in the instance to explain
            weights = lrc.coef_

            # print(weights[0])
            instance_vector_sparse = (instance_vector.toarray()[0])

            model_weights = pd.DataFrame(
                {'features': list(X.columns), 'weights': list(weights[0] * instance_vector_sparse)})
            model_weights = model_weights.reindex(model_weights['weights'].abs().sort_values(ascending=False).index)
            model_weights = model_weights[(model_weights["weights"] != 0)]

            mapping = vectorizer.vocabulary_
            inv_map = {v: k for k, v in mapping.items()}

            model_weights['words'] = model_weights['features'].map(inv_map)
            print(model_weights.head(n=20))

            summa = sum(weights[0] * instance_vector.toarray()[0])
            intercept = lrc.intercept_[0]
            result = ""
            if (summa + intercept > 0):
                result = " > 0 -> 1"
            else:
                result = " <= 0 -> 0"

            print('')
            print("Sum(weights*instance): " + str(summa) + " + Intercept (Bias): " + str(intercept) + " = " + str(
                summa + intercept) + result)

            results = {}

            for index, row in X.iterrows():
                d = cosine_similarity(instance_vector_sparse.reshape(1, -1), row.values.reshape(1, -1))
                results[index] = d

            # sort our results, so that the higher similarity are at the front of the list
            results = (sorted([(v, k) for (k, v) in results.items()], reverse=True))

            negative_examples = []
            positive_examples = []

            for item in results:
                index = item[1]
                label = int(y.to_list()[index])

                if len(negative_examples) < 2 and label == 0:
                    negative_examples.append(unprocessed_replies_list[index])
                if len(positive_examples) < 2 and label == 1:
                    positive_examples.append(unprocessed_replies_list[index])
                if len(positive_examples) >= 2 and len(negative_examples) >= 2:
                    break

            print("Real news spreaders say:")
            for item in negative_examples:
                print(item)

            print("Fake news spreaders say:")
            for item in positive_examples:
                print(item)

            create_csv(idx, instance_tweet_text, y_instance, true_label, no_of_replies)
            # final_df.to_csv('tweets_replies_labels.csv', mode='a', header=True)
            idx += 1
            predicted_labels.append(y_instance)
            true_labels.append(true_label)
            bb_probas.append(bb_proba[1])
            lr_probas.append(y_proba[1])

    print(len(predicted_labels))
    print(predicted_labels)
    print(true_labels)

    print(bb_probas)
    print(lr_probas)

    print(accuracy_score(predicted_labels, true_labels))
    print(accuracy_score(bb_probas, lr_probas))
