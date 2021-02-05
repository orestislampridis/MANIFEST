import pickle

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_score

import fake_news_spreader_feature_extraction as feature_extraction
from fake_news_spreader_feature_extraction import cleanPunc, clean_relics


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

    # convert to lower and remove punctuation or special characters
    data_combined['text'] = data_combined['text'].str.lower()
    data_combined['text'] = data_combined['text'].apply(cleanPunc)
    data_combined['text'] = data_combined['text'].apply(clean_relics)

    # get sentiment features from cleaned text
    data_sentiment = feature_extraction.get_sentiment_features(data_combined)

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

        features.to_csv('complete_features_with_labels_and_ids')

        X = features.drop(['ground_truth'], axis=1)
        y = features[['ground_truth']].values.ravel()

        print(X)
        print(y)
        # train-test split
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # try many classifiers to find the best performing
        names = [
            # "Nearest Neighbors",
            # "Linear SVC",
            # "RBF SVM",
            # "Decision Tree",
            "Random Forest",
            # "AdaBoost",
            # "Naive Bayes",
            "XGBoost"]

        classifiers = [
            # KNeighborsClassifier(),
            # SVC(kernel="linear", C=0.025),
            # SVC(gamma=2, C=1),
            # DecisionTreeClassifier(),
            RandomForestClassifier(),
            # AdaBoostClassifier(),
            # GaussianNB(),
            xgb.XGBClassifier(objective="binary:logistic", random_state=42)]

        best_clf = None
        best_classifier = ""
        best_accuracy = 0

        for clf, name in zip(classifiers, names):

            print("Classifier:", name)
            clf.fit(X, y)

            loo = LeaveOneOut()
            scores = cross_val_score(clf, X, y, scoring='accuracy', cv=loo)

            print((scores))
            cv_std = np.std(scores)
            cv_accuracy = np.mean(scores)

            print("Cross validation accuracy:", cv_accuracy)
            print("Cross validation standard deviation:", cv_std)

            print("\n")

            if best_accuracy < cv_accuracy:
                best_clf = clf
                best_classifier = name
                best_accuracy = cv_accuracy

        print("Best classifier: ", best_classifier)
        print("Best accuracy: \n", best_accuracy)

        # save the model to disk
        filename = 'classifiers/fake_news/' + str(best_classifier) + '_' + str(feature_names[i]) + '_' + str(
            best_accuracy) + '.sav'
        pickle.dump(best_clf, open(filename, 'wb'))
        i += 1


if __name__ == "__main__":
    main()
