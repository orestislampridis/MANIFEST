import csv
import os
import pickle
import re

import emoji
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from emotion.EmojiEmotionFeature import EmojiEmotionFeature
from personality_features import get_lang_based_scores

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))


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


# function to count negation words
def negation(tokens: list):
    negations = ["not", "no"]
    negative = 0

    for token in tokens:
        if token in negations:
            negative = 1

    return negative


# function to count slang words
def slang_count(text):
    slang_data = []
    with open(os.path.join(__location__, "slang.txt"), 'r', encoding="utf8") as exRtFile:
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


def demoji(tokens):
    """
    This function describes each emoji with a text that can be later used for vectorization and ML predictions
    :param tokens:
    :return:
    """
    emoji_description = []
    for token in tokens:
        detect = emoji.demojize(token)
        emoji_description.append(detect)
    return emoji_description


def get_emotion(doc):
    """
    This method used to detect emoji. All the emoji are separating in 4 categories anger, fear, joy, sadness and
    all of the detected emoji add a value to their category.
    :param doc: Array of string tokens
    :return: anger, fear, joy, sadness average value
    """
    emoji_emo = EmojiEmotionFeature()
    emojies = demoji(doc)
    emoji_emotion_values = []
    for e in emojies:
        if emoji_emo.is_emoji_name_like(e):

            if emoji_emo.exists_emoji_name(e):
                emotions = emoji_emo.emotions_of_emoji_named(e)
                emoji_emotion_values.append(emotions)

    if len(emoji_emotion_values) > 0:
        anger_total = 0
        fear_total = 0
        joy_total = 0
        sadness_total = 0

        for emotions in emoji_emotion_values:
            anger_total += emotions[0]
            fear_total += emotions[1]
            joy_total += emotions[2]
            sadness_total += emotions[3]

        anger = anger_total / len(emoji_emotion_values)
        fear = fear_total / len(emoji_emotion_values)
        joy = joy_total / len(emoji_emotion_values)
        sadness = sadness_total / len(emoji_emotion_values)

        return anger, fear, joy, sadness
    return 0, 0, 0, 0


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


def get_tfidf_vectors(df):
    # convert description to tf idf vector and pickle save vectorizer
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000, ngram_range=(1, 3), min_df=0.01, max_df=0.90)
    vectors = vectorizer.fit_transform(df['text'])
    pickle.dump(vectorizer, open("tfidf_fake_news.pkl", "wb"))  # save tfidf vector

    vectors_pd = pd.DataFrame(vectors.toarray())
    df = df.drop(columns=['text'])

    new_df = pd.concat([df, vectors_pd], axis=1)

    return new_df


def get_tfidf_vectors_from_pickle(df):
    vectorizer = pickle.load(open("tfidf_fake_news.pkl", 'rb'))

    vectors = vectorizer.transform(df['text'])
    vectors_pd = pd.DataFrame(vectors.toarray())
    df = df.drop(columns=['text'])

    new_df = pd.concat([df, vectors_pd], axis=1)

    return new_df


def get_readability_features(df):
    df['avg_word_count'] = df['text'].str.split().str.len() / 300
    df['slang_count'] = 0
    df['emoji_count'] = 0
    df['capitalized_count'] = 0
    df['full_capitalized_count'] = 0
    df['retweets_count'] = 0
    df['user_mentions_count'] = 0
    df['hashtags_count'] = 0
    df['url_count'] = 0

    for i in range(0, len(df)):
        df['slang_count'].iloc[i] = slang_count(df['text'].iloc[i])
        df['emoji_count'].iloc[i] = emoji_count(df['text'].iloc[i])
        df['capitalized_count'].iloc[i] = capitalized_count(df['text'].iloc[i])
        df['full_capitalized_count'].iloc[i] = full_capitalized_count(df['text'].iloc[i])

        retweets, user_mentions, hashtags, urls = count_relics(df['text'].iloc[i])
        df['retweets_count'].iloc[i] = retweets
        df['user_mentions_count'].iloc[i] = user_mentions
        df['hashtags_count'].iloc[i] = hashtags
        df['url_count'].iloc[i] = urls

    return df[
        ['user_id', 'avg_word_count', 'emoji_count', 'slang_count', 'capitalized_count', 'full_capitalized_count',
         'retweets_count', 'user_mentions_count', 'hashtags_count', 'url_count']]


def get_personality_features(df):
    df['text'] = df['text'].apply(clean_relics)

    users_with_personality = get_lang_based_scores(df)  # returns a df with user_id and personality scores
    # users_with_personality.to_csv("users_with_personality.csv", index=False)
    return users_with_personality


def get_gender_features(df):
    vectorizer = pickle.load(open("./gender/tfidf_gender.pkl", "rb"))

    data_readability = get_readability_features(df).drop(['user_id'], axis=1)
    vectors = vectorizer.transform(df['text'])

    # save sparse tfidf vectors to dataframe to use with other features
    vectors_pd = pd.DataFrame(vectors.toarray())

    X = pd.concat([vectors_pd, data_readability], axis=1)

    # load gender classifier
    filename = './gender/XGBoost_0.7426900584795322_final.sav'
    clf = pickle.load(open(filename, 'rb'))

    y = clf.predict(X)

    # convert M and F to float representation
    gender = {'M': 0, 'F': 1}
    y = [gender[item] for item in y]
    df['gender'] = y

    return df[['user_id', 'gender']]


def get_sentiment_features(df):
    vader = SentimentIntensityAnalyzer()

    df['anger'] = 0
    df['fear'] = 0
    df['joy'] = 0
    df['sadness'] = 0
    df['negation'] = 0
    df['vader_score'] = 0
    df['textblob_score'] = 0

    vader_score = []
    textblob_score = []

    for i in range(0, len(df)):
        df['anger'].iloc[i], df['fear'].iloc[i], df['joy'].iloc[i], df['sadness'].iloc[i] \
            = get_emotion(df['text'].iloc[i])
        df['negation'].iloc[i] = negation(df['text'].iloc[i])
        k = vader.polarity_scores(df['text'].iloc[i])
        l = TextBlob(df['text'].iloc[i]).sentiment

        vader_score.append(k)
        textblob_score.append(l)

    vader_compound_score = pd.DataFrame.from_dict(vader_score)
    textblob_polarity_score = pd.DataFrame.from_dict(textblob_score)

    df['vader_compound_score'] = vader_compound_score['compound']
    df['textblob_polarity_score'] = textblob_polarity_score['polarity']

    return df[['user_id', 'anger', 'fear', 'joy', 'sadness', 'negation',
               'vader_compound_score', 'textblob_polarity_score']]


def get_tf_idf_features(df):
    vectorizer = pickle.load(open("tfidf_fake_news.pkl", "rb"))

    vectors = vectorizer.transform(df['text'])

    # save sparse tfidf vectors to dataframe to use with other features
    vectors_pd = pd.DataFrame(vectors.toarray())
    X = pd.concat([vectors_pd], axis=1)

    # load stored tf-idf classifier
    filename = 'models/tf_idf_classifier_Naive Bayes_0.7733333333333333.sav'
    clf = pickle.load(open(filename, 'rb'))

    y = clf.predict(X)
    print(y)
    df['tf_idf'] = y

    return df[['user_id', 'tf_idf']]
