import pickle

import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import *
from sklearn.tree import DecisionTreeClassifier

import fake_news_spreader_feature_extraction as feature_extraction
from preprocessing import clean_text


def list2string(list):
    return ','.join(map(str, list))


file_tweets = "new_gender_combined.csv"
file_genders = "ground_truth_gender.csv"

data_tweets = pd.read_csv(file_tweets, sep=",", encoding="utf8")
data_genders = pd.read_csv(file_genders, sep=",", encoding="utf8").drop(['tweets'], axis=1)

print(data_tweets)
print(data_genders)

# Join the two dataframes together
merged_df = pd.merge(data_tweets, data_genders, on='twitter_uid', how='inner')
merged_df = merged_df.dropna().reset_index()

print(merged_df.columns)

data_combined = merged_df[['twitter_uid', 'statuses', 'Gender', 'tweets', 'retweets', 'urls']]
data_combined.rename(columns={'statuses': 'text'}, inplace=True)

print(data_combined.columns)

# count various readability features
data_readability = feature_extraction.get_readability_features(data_combined)

data_combined['text'] = merged_df.statuses.apply(clean_text)
data_combined['text'] = [list2string(list) for list in data_combined['text']]

vectorizer = TfidfVectorizer(stop_words='english', max_features=10000, min_df=0.01, max_df=0.90, ngram_range=(1, 4))
vectors = vectorizer.fit_transform(data_combined['text'])

# save sparse tfidf vectors to dataframe to use with other features
vectors_pd = pd.DataFrame(vectors.toarray())

# save tfidf vector
pickle.dump(vectorizer, open("tfidf_gender.pkl", "wb"))

X = pd.concat([vectors_pd, data_readability], axis=1)
y = data_combined['Gender']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

names = [
    "Nearest Neighbors",
    "Linear SVC",
    "RBF SVM",
    "Decision Tree",
    "Random Forest",
    "AdaBoost",
    "XGBoost"]

classifiers = [
    KNeighborsClassifier(12),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    xgb.XGBClassifier(objective="binary:logistic", random_state=42)
]

# try with several different classifiers to find best one

best_classifier = ""
best_accuracy = 0
best_clf = None

for clf, name in zip(classifiers, names):

    print("Classifier:", name)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precission:", precision_score(y_test, y_pred, average='macro'))
    print("Recal:", recall_score(y_test, y_pred, average='macro'))
    print("f1_score:", f1_score(y_test, y_pred, average='macro'))
    print("\n")

    if best_accuracy < accuracy_score(y_test, y_pred):
        best_clf = clf
        best_classifier = name
        best_accuracy = accuracy_score(y_test, y_pred)

print("Best classifier:", best_classifier)
print("Best accuracy:", best_accuracy)

# save the model to disk
filename = best_classifier + '_final.sav'
pickle.dump(best_clf, open(filename, 'wb'))
