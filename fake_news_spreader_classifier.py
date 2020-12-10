import pickle
import re

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, LeaveOneOut, cross_val_score
from sklearn.naive_bayes import GaussianNB

import fake_news_spreader_feature_extraction as feature_extraction


# function to clean the word of any punctuation or special characters
# we keep slang and emojis as they might aid in differentiating between fake and real news spreaders
# TODO: improve text preprocessing
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


def main():
    # data_combined = pd.read_csv("dataset/data_csv/data_combined.csv", sep=",", encoding="utf8")
    data_combined = pd.read_csv("dataset/data_csv/LIWC2015_Results.csv", sep=",", encoding="utf8")

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

    # get gender features from cleaned text
    data_gender = feature_extraction.get_gender_features(data_combined)

    print(data_combined.columns.values.tolist())
    data_tfidf = feature_extraction.get_tfidf_vectors(data_combined[['user_id', 'text']])

    # use scaler to scale our data to [0,1] range
    # scaler = MinMaxScaler()
    # data_readability[['avg_word_count', 'emoji_count', 'slang_count', 'capitalized_count', 'full_capitalized_count',
    #                  'retweets_count', 'user_mentions_count', 'hashtags_count', 'url_count']] = scaler.fit_transform(
    #    data_readability[['avg_word_count', 'emoji_count', 'slang_count', 'capitalized_count', 'full_capitalized_count',
    #                      'retweets_count', 'user_mentions_count', 'hashtags_count', 'url_count']])

    # separate our data to features/labels - X, y

    print("tf_idf")
    print(data_tfidf.columns.values.tolist())
    print("readability")
    print(data_readability.columns.values.tolist())
    print("liwc")
    print(data_liwc.columns.values.tolist())
    print("personality")
    print(data_personality.columns.values.tolist())
    print("gender")
    print(data_gender.columns.values.tolist())

    i = 0
    features = list()
    # features.append([data_tfidf])
    features.append([data_tfidf, data_readability])
    features.append([data_tfidf, data_readability, data_personality, data_gender])
    features.append([data_tfidf, data_readability, data_liwc])
    features.append([data_tfidf, data_readability, data_liwc, data_personality])
    features.append([data_tfidf, data_readability, data_liwc, data_gender])
    features.append([data_readability, data_liwc, data_personality, data_gender])
    features.append([data_readability, data_liwc, data_personality])
    features.append([data_readability, data_liwc, data_gender])

    for feature_combination in features:

        print("feature_combination: " + str(i))

        features = pd.concat([i.set_index('user_id') for i in feature_combination], axis=1, join='outer')

        X = features
        y = data_combined['ground_truth']

        # train-test split
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        # try many classifiers to find the best performing
        names = [
            # "Nearest Neighbors",
            # "Linear SVC",
            # "RBF SVM",
            # "Decision Tree",
            "Random Forest",
            "AdaBoost",
            "Naive Bayes",
            "XGBoost"]

        classifiers = [
            # KNeighborsClassifier(),
            # SVC(kernel="linear", C=0.025),
            # SVC(gamma=2, C=1),
            # DecisionTreeClassifier(),
            RandomForestClassifier(),
            AdaBoostClassifier(),
            GaussianNB(),
            xgb.XGBClassifier(objective="binary:logistic", random_state=42)]

        # try with several different classifiers to find best one

        best_clf = None
        best_classifier = ""
        best_accuracy = 0

        for clf, name in zip(classifiers, names):

            print("Classifier:", name)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            loo = LeaveOneOut()
            scores = cross_val_score(clf, X, y, scoring='accuracy', cv=loo)

            cv_std = np.std(scores)
            cv_accuracy = np.mean(scores)

            print("Cross validation accuracy:", cv_accuracy)
            print("Cross validation standard deviation:", cv_std)

            print("Precission:", precision_score(y_test, y_pred, average='macro'))
            print("Recal:", recall_score(y_test, y_pred, average='macro'))
            print("f1_score:", f1_score(y_test, y_pred, average='macro'))
            print("\n")

            if best_accuracy < cv_accuracy:
                best_clf = clf
                best_classifier = name
                best_accuracy = cv_accuracy

        i += 1
        print("Best classifier: ", best_classifier)
        print("Best accuracy: ", best_accuracy)

    # save the model to disk
    filename = best_classifier + '_final.sav'
    pickle.dump(best_clf, open(filename, 'wb'))


if __name__ == "__main__":
    main()
