import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split

from utils import random_search, grid_search

warnings.filterwarnings('ignore')


def main():
    df = pd.read_csv("scaled_final_complete_features_with_labels_and_ids", sep=",", encoding="utf8")

    data_tfidf = df[[str(x) for x in range(1000)]]
    data_readability = df[
        ['avg_word_count', 'emoji_count', 'slang_count', 'capitalized_count', 'full_capitalized_count',
         'retweets_count', 'user_mentions_count', 'hashtags_count', 'url_count']]
    data_sentiment = df[
        ['anger', 'fear', 'joy', 'sadness', 'negation', 'vader_compound_score', 'textblob_polarity_score']]
    data_personality = df[['extraversion', 'avoidance', 'conscientiousness', 'openness', 'neuroticism',
                           'agreeableness', 'anxiety']]
    data_gender = df[['gender']]
    data_liwc = df[['Analytic', 'Clout', 'Authentic', 'Tone']]
    data_ground_truth = df[['ground_truth']]

    i = 0
    feature_names = [
        # "explanations_tfidf",
        "phase_C_tfidf_readability_sentiment_personality_gender",
        # "explanations_readability_sentiment_personality_gender",
        # "explanations_readability_sentiment_personality_gender_liwc"
    ]

    features = list()
    # features.append([data_tfidf, data_ground_truth])
    features.append([data_tfidf, data_readability, data_sentiment, data_personality, data_gender, data_ground_truth])
    # features.append([data_readability, data_sentiment, data_personality, data_gender, data_ground_truth])
    # features.append([data_readability, data_sentiment, data_personality, data_gender, data_liwc, data_ground_truth])

    for feature_combination in features:
        print("feature_combination: " + str(feature_names[i]))

        features = pd.concat([i for i in feature_combination], axis=1)

        # print(features)
        features.to_csv('features_with_labels_and_ids')

        X = features.drop(['ground_truth'], axis=1).reset_index(drop=True)
        y = features[['ground_truth']].values.ravel()
        # print(X)
        # print(y)
        # print(X.shape)
        # print(y.shape)

        # train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # try many classifiers to find the best performing
        clf_names = [
            # "knn",
            # "Naive Bayes",
            # "Logistic Regression",
            # "SVM",
            # "Random Forest",
            "Gradient Boosting",
            # "NN"
        ]

        best_clf = None
        best_classifier = ""
        best_accuracy = 0

        for name in clf_names:
            print("Classifier:", name)

            # Perform random search or grid search for each clf to find the best hyper-parameters
            if name == "KNN":
                clf = random_search.get_knn_random_grid(X_train, y_train)

            if name == "Naive Bayes":
                clf = random_search.get_NB_random_grid(X_train, y_train)

            if name == "Logistic Regression":
                clf = random_search.get_logistic_regression_random_grid(X_train, y_train)
                # clf = grid_search.get_logistic_regression_grid_search(X_train, y_train)

            if name == "SVM":
                clf = random_search.get_SVM_random_grid(X_train, y_train)
                # clf = grid_search.get_SVM_grid_search(X_train, y_train)

            if name == "Random Forest":
                clf = random_search.get_random_forest_random_grid(X_train, y_train)
                # clf = grid_search.get_random_forest_grid_search(X_train, y_train)

            if name == "Gradient Boosting":
                # clf = random_search.get_gradient_boosting_random_grid(X_train, y_train)
                clf = grid_search.get_gradient_boosting_grid_search(X_train, y_train)

            if name == "NN":
                clf = random_search.get_neural_network_random_grid(X_train, y_train)
                # clf = grid_search.get_neural_network_grid_search(X_train, y_train)

            clf.fit(X_train, y_train)
            y_preds = clf.predict(X_test)

            # Training accuracy
            print("The training accuracy is: ")
            print(accuracy_score(y_train, clf.predict(X_train)))

            # Test accuracy
            print("The test accuracy is: ")
            print(accuracy_score(y_test, y_preds))

            # Classification report
            print("Classification report")
            print(classification_report(y_test, y_preds))

            # Cross validation scoring
            print("Cross validation scoring")
            accuracy = cross_val_score(clf, X, y, scoring='accuracy', cv=10)
            print('Accuracy:', np.mean(accuracy))
            precision = cross_val_score(clf, X, y, cv=10, scoring='precision')
            print('Precision:', np.mean(precision))
            recall = cross_val_score(clf, X, y, scoring='recall', cv=10)
            print('Recall:', np.mean(recall))
            f1 = cross_val_score(clf, X, y, cv=10, scoring='f1')
            print('F1:', np.mean(f1))

            cv_accuracy = np.mean(accuracy)
            print("\n")

            if best_accuracy <= cv_accuracy:
                clf.fit(X, y)
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
