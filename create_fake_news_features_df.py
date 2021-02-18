import warnings

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import fake_news_spreader_feature_extraction as feature_extraction
from fake_news_spreader_feature_extraction import clean_relics
from utils.preprocessing import clean_text

warnings.filterwarnings('ignore')


def main():
    # data_combined = pd.read_csv("dataset/data_csv/data_combined.csv", sep=",", encoding="utf8")
    data_combined = pd.read_csv("dataset/data_csv/LIWC2015_Results.csv", sep=",", encoding="utf8")

    data_ground_truth = data_combined[['user_id', 'ground_truth']]

    # keep only LIWC related data by dropping everything else from the data_combined df
    data_liwc = data_combined.drop(
        ['text', 'ground_truth'], axis=1)

    data_liwc = data_liwc[['user_id', 'Analytic', 'Clout', 'Authentic', 'Tone']]

    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    #    print(data_liwc)

    # count various readability features
    data_readability = feature_extraction.get_readability_features(data_combined)

    # get personality features
    data_personality = feature_extraction.get_personality_features(data_combined)
    data_personality.rename(
        columns={'ANXIETY': 'anxiety', 'E': 'extraversion', 'A': 'agreeableness', 'O': 'openness',
                 'C': 'conscientiousness', 'N': 'neuroticism', 'AVOIDANCE': 'avoidance'}, inplace=True)

    # print(data_personality.columns)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(data_personality)

    # get sentiment features
    data_sentiment = feature_extraction.get_sentiment_features(data_combined)

    # convert to lower and remove punctuation or special characters
    data_combined['text'] = data_combined['text'].apply(clean_relics)
    data_combined['text'] = data_combined['text'].apply(clean_text)

    # get gender features from cleaned text
    data_gender = feature_extraction.get_gender_features(data_combined)

    # get tf-idf features from cleaned text
    data_tfidf = feature_extraction.get_tfidf_vectors(data_combined[['user_id', 'text']])

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
        "tfidf_readability_sentiment_liwc_personality_gender",
    ]

    features = list()
    features.append(
        [data_tfidf, data_readability, data_sentiment, data_liwc, data_personality, data_gender, data_ground_truth])

    for feature_combination in features:
        print("feature_combination: " + str(feature_names[i]))

        features = pd.concat([i.set_index('user_id') for i in feature_combination], axis=1, join='outer')

        print(features)
        features.to_csv('final_final_complete_features_with_labels_and_ids')

        # use scaler to scale our data to [0,1] range
        x = features.values  # returns a numpy array
        scaler = MinMaxScaler()
        x_scaled = scaler.fit_transform(x)
        df = pd.DataFrame(x_scaled, columns=features.columns)
        print(df)
        df.to_csv('scaled_final_complete_features_with_labels_and_ids')


if __name__ == "__main__":
    main()
