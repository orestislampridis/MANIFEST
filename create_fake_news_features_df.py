import pandas as pd

import fake_news_spreader_feature_extraction as feature_extraction
from fake_news_spreader_feature_extraction import clean_relics
from preprocessing import clean_text


def main():
    # data_combined = pd.read_csv("dataset/data_csv/data_combined.csv", sep=",", encoding="utf8")
    data_combined = pd.read_csv("dataset/data_csv/LIWC2015_Results.csv", sep=",", encoding="utf8")

    data_ground_truth = data_combined[['user_id', 'ground_truth']]

    # keep only LIWC related data by dropping everything else from the data_combined df
    data_liwc = data_combined.drop(
        ['text', 'ground_truth'], axis=1)

    # count various readability features
    data_readability = feature_extraction.get_readability_features(data_combined)

    # get personality features
    data_personality = feature_extraction.get_personality_features(data_combined)

    # get sentiment features
    data_sentiment = feature_extraction.get_sentiment_features(data_combined)

    print(data_sentiment['anger'])
    print(data_sentiment['fear'])
    print(data_sentiment['joy'])
    print(data_sentiment['sadness'])

    # convert to lower and remove punctuation or special characters
    data_combined['text'] = data_combined['text'].apply(clean_relics)
    data_combined['text'] = data_combined['text'].apply(clean_text)

    # get gender features from cleaned text
    data_gender = feature_extraction.get_gender_features(data_combined)

    print(data_combined.columns.values.tolist())
    data_tfidf = feature_extraction.get_tf_idf_features(data_combined[['user_id', 'text']])

    # tf_idf_model = pickle

    # use scaler to scale our data to [0,1] range
    # scaler = MinMaxScaler()
    # data_readability[['avg_word_count', 'emoji_count', 'slang_count', 'capitalized_count', 'full_capitalized_count',
    #                  'retweets_count', 'user_mentions_count', 'hashtags_count', 'url_count']] = scaler.fit_transform(
    #    data_readability[['avg_word_count', 'emoji_count', 'slang_count', 'capitalized_count', 'full_capitalized_count',
    #                      'retweets_count', 'user_mentions_count', 'hashtags_count', 'url_count']])

    # separate our data to features/labels - X, y

    print("tf-idf")
    print(data_tfidf.columns.values.tolist())
    print("readability")
    print(data_readability.columns.values.tolist())
    print("sentiment")
    print(data_sentiment.columns.values.tolist())
    print("liwc")
    print(data_liwc.columns.values.tolist())
    print("personality")
    print(data_personality.columns.values.tolist())
    print("gender")
    print(data_gender.columns.values.tolist())

    i = 0

    feature_names = [
        "tfidf_readability_liwc_personality_gender",
    ]

    features = list()
    features.append(
        [data_tfidf, data_readability, data_sentiment, data_liwc, data_personality, data_gender, data_ground_truth])

    for feature_combination in features:
        print("feature_combination: " + str(feature_names[i]))

        features = pd.concat([i.set_index('user_id') for i in feature_combination], axis=1, join='outer')

        print(features)

        features.to_csv('final_complete_features_with_labels_and_ids')


if __name__ == "__main__":
    main()
