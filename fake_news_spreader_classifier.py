import csv
import pickle
import re

import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# function to clean the word of any punctuation or special characters
# we keep slang and emojis as they might aid in differentiating between fake and real news spreaders
# TODO: improve text preprocessing
def cleanPunc(sentence):
    cleaned = re.sub(r'[?|!|\'|"|#]', r'', sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]', r' ', cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n", " ")
    return cleaned


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
    with open("slang.txt", 'r', encoding="utf8") as exRtFile:
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


# function to count RTs, User mentions, hashtags, urls
def count_relics(text):
    retweets = len(re.findall('RT', text))
    user_mentions = len(re.findall('#USER#', text))
    hashtags = len(re.findall('#HASHTAG#', text))
    urls = len(re.findall('#URL#', text))
    return retweets, user_mentions, hashtags, urls


# function to clean relics of dataset
def clean_relics(text):
    text = re.sub(r"RT", "", text)
    text = re.sub(r"#USER#", "", text)
    text = re.sub(r"#HASHTAG#", "", text)
    text = re.sub(r"#URL#", "", text)
    return text


# function to count capitalized words (e.g. Apple, Woof, Dog but not APPLE etc)
def capitalized_count(text):
    text = clean_relics(text)
    t = re.findall('([A-Z][a-z]+)', text)
    return len(t)


# function to count capitalized words (e.g. APPLE, WOOF)
def full_capitalized_count(text):
    text = clean_relics(text)
    t = re.findall('([A-Z][A-Z]+)', text)
    return len(t)


def main():
    # data_combined = pd.read_csv("dataset/data_csv/data_combined.csv", sep=",", encoding="utf8")
    data_liwc = pd.read_csv("dataset/data_csv/LIWC2015_Results.csv", sep=",", encoding="utf8")
    data_personality = pd.read_csv("personality/users_with_personality.csv", sep=",", encoding="utf8")

    # Join the two dataframes together
    data_combined = pd.merge(data_liwc, data_personality, on='user_id', how='inner')

    print(data_combined)

    # count various readability features
    data_combined['avg_word_count'] = data_combined['tweet_text'].str.split().str.len() / 300
    data_combined['slang_count'] = 0
    data_combined['emoji_count'] = 0
    data_combined['capitalized_count'] = 0
    data_combined['full_capitalized_count'] = 0
    data_combined['retweets_count'] = 0
    data_combined['user_mentions_count'] = 0
    data_combined['hashtags_count'] = 0
    data_combined['url_count'] = 0

    for i in range(0, len(data_combined)):
        data_combined['slang_count'].iloc[i] = slang_count(data_combined['tweet_text'].iloc[i])
        data_combined['emoji_count'].iloc[i] = emoji_count(data_combined['tweet_text'].iloc[i])
        data_combined['capitalized_count'].iloc[i] = capitalized_count(data_combined['tweet_text'].iloc[i])
        data_combined['full_capitalized_count'].iloc[i] = full_capitalized_count(data_combined['tweet_text'].iloc[i])

        retweets, user_mentions, hashtags, urls = count_relics(data_combined['tweet_text'].iloc[i])
        data_combined['retweets_count'].iloc[i] = retweets
        data_combined['user_mentions_count'].iloc[i] = user_mentions
        data_combined['hashtags_count'].iloc[i] = hashtags
        data_combined['url_count'].iloc[i] = urls

    # convert to lower and remove punctuation or special characters
    data_combined['tweet_text'] = data_combined['tweet_text'].str.lower()
    data_combined['tweet_text'] = data_combined['tweet_text'].apply(cleanPunc)
    data_combined['tweet_text'] = data_combined['tweet_text'].apply(clean_relics)

    print(data_combined['url_count'])
    print(data_combined.head())

    # keep only the columns that we need for the readability features
    data_readability = data_combined[
        ['avg_word_count', 'emoji_count', 'slang_count', 'capitalized_count', 'full_capitalized_count',
         'retweets_count', 'user_mentions_count', 'hashtags_count', 'url_count']]
    print(data_readability.head())

    # keep only personality data
    data_personality = data_combined[['E', 'AVOIDANCE', 'C', 'O', 'N', 'A', 'ANXIETY']]
    print(data_personality.head())

    # keep only LIWC related data by dropping everything else from the data_combined df
    data_liwc = data_combined.drop(
        ['user_id', 'tweet_text', 'ground_truth', 'avg_word_count', 'emoji_count', 'slang_count', 'capitalized_count',
         'full_capitalized_count', 'retweets_count', 'user_mentions_count', 'hashtags_count', 'url_count', 'E',
         'AVOIDANCE', 'C', 'O', 'N', 'A', 'ANXIETY'], axis=1)

    print(data_liwc.columns)

    # convert description to tf idf vector and pickle save vectorizer
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000, ngram_range=(1, 3))
    vectors = vectorizer.fit_transform(data_combined['tweet_text'])
    pickle.dump(vectorizer.vocabulary_, open("tfidf_age.pkl", "wb"))  # save tfidf vector

    # save sparse tfidf vectors to dataframe to use with other features
    vectors_pd = pd.DataFrame(vectors.toarray())

    print(vectors_pd)
    print(data_liwc)

    # use scaler to scale our data to [0,1] range
    # scaler = MinMaxScaler()
    # data_readability[['avg_word_count', 'emoji_count', 'slang_count', 'capitalized_count', 'full_capitalized_count',
    #                  'retweets_count', 'user_mentions_count', 'hashtags_count', 'url_count']] = scaler.fit_transform(
    #    data_readability[['avg_word_count', 'emoji_count', 'slang_count', 'capitalized_count', 'full_capitalized_count',
    #                      'retweets_count', 'user_mentions_count', 'hashtags_count', 'url_count']])

    # separate our data to features/labels - X, y
    X = pd.concat([vectors_pd, data_readability, data_liwc, data_personality], axis=1)
    y = data_combined['ground_truth']

    print(X)
    print(y)

    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

    # try many classifiers to find the best performing
    names = [
        "Nearest Neighbors",
        "Linear SVC",
        "RBF SVM",
        "Decision Tree",
        "Random Forest",
        "AdaBoost",
        "Naive Bayes",
        "XGBoost"]

    classifiers = [
        KNeighborsClassifier(),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        DecisionTreeClassifier(),
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

    # data_combined['tweet_text'] = data_combined.tweet_text.apply(clean_text)
    # data_combined['tweet_text'] = [list2string(list) for list in data_combined['tweet_text']]

    # print(data_combined['tweet_text'])


if __name__ == "__main__":
    main()
