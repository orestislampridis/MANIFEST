import pickle
import re

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_score

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

    # convert to lower and remove punctuation or special characters
    data_combined['text'] = data_combined['text'].str.lower()
    data_combined['text'] = data_combined['text'].apply(cleanPunc)
    data_combined['text'] = data_combined['text'].apply(clean_relics)

    data_tfidf = feature_extraction.get_tfidf_vectors(data_combined[['user_id', 'text']])

    i = 0

    feature_names = [
        "tfidf"]

    features = list()
    features.append([data_tfidf])

    for feature_combination in features:

        print("feature_combination: " + str(feature_names[i]))

        features = pd.concat([i.set_index('user_id') for i in feature_combination], axis=1, join='outer')

        X = features
        y = data_combined['ground_truth']

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

        # try with several different classifiers to find best one

        best_clf = None
        best_classifier = ""
        best_accuracy = 0

        for clf, name in zip(classifiers, names):

            print("Classifier:", name)
            clf.fit(X, y)

            loo = LeaveOneOut()
            scores = cross_val_score(clf, X, y, scoring='accuracy', cv=loo)

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
        filename = 'models/' + str(best_classifier) + '_' + str(feature_names[i]) + '_' + str(
            best_accuracy) + '.sav'
        pickle.dump(best_clf, open(filename, 'wb'))
        i += 1


if __name__ == "__main__":
    main()
