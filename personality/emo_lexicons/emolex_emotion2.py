import csv
import json

import numpy as np
import pandas as pd


def init_lex():
    LEXICON_FILE = 'emo_lexicons/lexicons/lexicons_compiled.csv'
    CATEGORIES_FILE = 'emo_lexicons/data/categories.json'

    # Init
    vocabulary = []
    words = []
    categories = {}
    category_headers = []
    data = []

    # Read vocabulary
    with open(LEXICON_FILE, 'r', encoding='utf_8') as f:
        rows = csv.reader(f, delimiter=',')
        headers = next(rows, None)  # remove header
        for row in rows:
            entry = {}
            for i, h in enumerate(headers):
                entry[h] = row[i]
            vocabulary.append(entry)
        words = [v['word'] for v in vocabulary]

    # Read categories
    with open(CATEGORIES_FILE) as f:
        categories = json.load(f)
        category_headers = categories.keys()

    return vocabulary, words, categories, category_headers, data


def addData(word, chapter, vocabulary, words, categories, category_headers, data):
    match = -1
    for i, w in enumerate(words):
        if w == word:
            match = i
            break

    if match >= 0:
        entry = vocabulary[match]
        row = []
        for category in category_headers:
            if entry[category]:
                row.append(categories[category].index(entry[category]))
            else:
                row.append(-1)
        row.append(chapter)
        data.append(row)


def emotion_extraction(mydf):
    # Initializing words from lexicons

    vocabulary, words, categories, category_headers, data = init_lex()

    # Pass the dataframe with one text column for analysis
    for i in range(0, len(mydf)):
        words = mydf.iloc[i]['cwords'].split(' ')
        for word in words:
            addData(word, i, vocabulary, words, categories, category_headers, data)

    # Create the dataframe from calculated data
    df = pd.DataFrame.from_records(data,
                                   columns=['emotion', 'color', 'orientation', 'sentiment', 'subjectivity', 'tweet'])

    # Create emotion, sentiment and subjectivity dataframes
    emotions = pd.DataFrame(0, index=np.arange(0, df.tweet.nunique()),
                            columns=['anger_emo', 'fear_emo', 'anticipation_emo', 'trust_emo', 'surprise_emo',
                                     'sadness_emo', 'joy_emo', 'disgust_emo'])
    sentiment = pd.DataFrame(0, index=np.arange(0, df.tweet.nunique()),
                             columns=['positive_sentiment_emo', 'negative_sentiment_emo'])

    # Analyze the data
    for i in range(0, len(df)):
        current_tweet = df.iloc[i]['tweet']

        # emotions analysis
        emo_el = df.iloc[i]['emotion']
        if emo_el == 0:
            val = emotions.iloc[current_tweet]['anger_emo']
            emotions.set_value(current_tweet, 'anger_emo', val + 1)
        elif emo_el == 1:
            val = emotions.iloc[current_tweet]['fear_emo']
            emotions.set_value(current_tweet, 'fear_emo', val + 1)
        elif emo_el == 2:
            val = emotions.iloc[current_tweet]['anticipation_emo']
            emotions.set_value(current_tweet, 'anticipation_emo', val + 1)
        elif emo_el == 3:
            val = emotions.iloc[current_tweet]['trust_emo']
            emotions.set_value(current_tweet, 'trust_emo', val + 1)
        elif emo_el == 4:
            val = emotions.iloc[current_tweet]['surprise_emo']
            emotions.set_value(current_tweet, 'surprise_emo', val + 1)
        elif emo_el == 5:
            val = emotions.iloc[current_tweet]['sadness_emo']
            emotions.set_value(current_tweet, 'sadness_emo', val + 1)
        elif emo_el == 6:
            val = emotions.iloc[current_tweet]['joy_emo']
            emotions.set_value(current_tweet, 'joy_emo', val + 1)
        elif emo_el == 7:
            val = emotions.iloc[current_tweet]['disgust_emo']
            emotions.set_value(current_tweet, 'disgust_emo', val + 1)

        # sentiment analysis
        sent_el = df.iloc[i]['sentiment']
        if sent_el == 0:
            val = sentiment.iloc[current_tweet]['positive_sentiment_emo']
            sentiment.set_value(current_tweet, 'positive_sentiment_emo', val + 1)
        elif sent_el == 1:
            val = sentiment.iloc[current_tweet]['negative_sentiment_emo']
            sentiment.set_value(current_tweet, 'negative_sentiment_emo', val + 1)

    # Normalize data
    def normalize_df(df):
        for colname, col in df.iteritems():
            df[colname] = df[colname] / df[colname].max()
        return df

    emotions = normalize_df(emotions)
    # drop anticipation not in 6 emotions
    emotions = emotions.drop('anticipation_emo', axis=1)

    sentiment = normalize_df(sentiment)

    # Concat to the original dataframe
    result = pd.concat([mydf, emotions], axis=1)
    result = pd.concat([result, sentiment], axis=1)

    return result
