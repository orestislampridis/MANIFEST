import csv
import pickle
import re

import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import *
from sklearn.tree import DecisionTreeClassifier

from fake_news_spreader_classifier import capitalized_count, full_capitalized_count
from preprocessing import clean_text


def list2string(list):
    return ','.join(map(str, list))


# function to count RTs, User mentions, hashtags, urls
def count_twitter_relics(text):
    user_mentions = len(re.findall('@', text))
    hashtags = len(re.findall('#', text))
    return user_mentions, hashtags


# function to count emojis
def emoji_count(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               "]+", flags=re.UNICODE)

    counter = 0
    data = list(text)  # split the text into characters

    for word in data:
        counter += len(emoji_pattern.findall(word))
    return counter


# function to count slang words
def slang_count(text):
    slang_data = []
    with open("../slang.txt", 'r', encoding="utf8") as exRtFile:
        exchReader = csv.reader(exRtFile, delimiter=':')

        for row in exchReader:
            slang_data.append(row[0].lower())

    counter = 0
    data = text.lower().split()
    for word in data:
        for slang in slang_data:
            if slang == word:
                counter += 1
    return counter


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

print(data_combined.columns)

# count various readability features
data_combined['avg_word_count'] = data_combined['statuses'].str.split().str.len() / data_combined['tweets']
data_combined['slang_count'] = 0
data_combined['emoji_count'] = 0
data_combined['capitalized_count'] = 0
data_combined['full_capitalized_count'] = 0
data_combined['user_mentions_count'] = 0
data_combined['hashtags_count'] = 0

for i in range(0, len(data_combined)):
    data_combined['slang_count'].iloc[i] = slang_count(data_combined['statuses'].iloc[i])
    data_combined['emoji_count'].iloc[i] = emoji_count(data_combined['statuses'].iloc[i])
    data_combined['capitalized_count'].iloc[i] = capitalized_count(data_combined['statuses'].iloc[i])
    data_combined['full_capitalized_count'].iloc[i] = full_capitalized_count(data_combined['statuses'].iloc[i])

    user_mentions, hashtags = count_twitter_relics(data_combined['statuses'].iloc[i])
    data_combined['user_mentions_count'].iloc[i] = user_mentions
    data_combined['hashtags_count'].iloc[i] = hashtags

print(data_combined['statuses'])
data_combined['statuses'] = merged_df.statuses.apply(clean_text)
print(data_combined['statuses'])

print(data_combined.head(3))

data_combined['statuses'] = [list2string(list) for list in data_combined['statuses']]
print(data_combined['statuses'])

vectorizer = TfidfVectorizer(stop_words='english', max_features=10000, min_df=0.01, max_df=0.90, ngram_range=(1, 4))
print(len(data_combined['statuses']))
vectors = vectorizer.fit_transform(data_combined['statuses'])
print(len(vectors.toarray()))
print(vectors.toarray())

# save sparse tfidf vectors to dataframe to use with other features
vectors_pd = pd.DataFrame(vectors.toarray())
print(vectors_pd)

# save tfidf vector
pickle.dump(vectorizer.vocabulary_, open("tfidf_gender.pkl", "wb"))

# keep only the columns that we need
data_readability = data_combined[
    ['avg_word_count', 'emoji_count', 'slang_count', 'capitalized_count', 'full_capitalized_count',
     'retweets', 'user_mentions_count', 'hashtags_count', 'urls']]

print(data_readability)
print(data_combined['Gender'])

X = pd.concat([vectors_pd, data_readability], axis=1)
print(X)
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
