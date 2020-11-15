# -*- coding: utf-8 -*-

# This script analyzes text data based on lexicons
#   Usage: python analyze_data.py output/moby_dick_data.csv output/moby_dick_analysis.json 400 200

import csv
import json

import numpy as np
import pandas as pd


def emotion_extraction(mydf):  # or mydf ctweets
    LEXICON_FILE = 'emo_lexicons/lexicons/lexicons_compiled.csv'
    CATEGORIES_FILE = 'emo_lexicons/data/categories.json'

    # Init
    vocabulary = []
    words = []
    categories = {}
    category_headers = []
    data = []

    # Read vocabulary
    with open(LEXICON_FILE, 'rb') as f:
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

    # Check if word matches any lexicons
    def addData(word, chapter):
        global data
        global vocabulary
        global categories
        global category_headers
        global words

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

    # Pass the dataframe with one text column for analysis
    for i in range(0, len(mydf)):
        words = mydf.iloc[i]['cwords'].split(' ')
        for word in words:
            addData(word, i)

    # Create the dataframe from calculated data
    df = pd.DataFrame.from_records(data,
                                   columns=['emotion', 'color', 'orientation', 'sentiment', 'subjectivity', 'tweet'])

    # Create emotion, sentiment and subjectivity dataframes
    emotions = pd.DataFrame(0, index=np.arange(0, df.tweet.nunique()),
                            columns=['anger', 'fear', 'anticipation', 'trust', 'surprise', 'sadness', 'joy', 'disgust'])
    sentiment = pd.DataFrame(0, index=np.arange(0, df.tweet.nunique()),
                             columns=['positive_sentiment', 'negative_sentiment'])

    # Analyze the data
    for i in range(0, len(df)):
        current_tweet = df.iloc[i]['tweet']

        # emotions analysis
        emo_el = df.iloc[i]['emotion']
        if emo_el == 0:
            val = emotions.iloc[current_tweet]['anger']
            emotions.set_value(current_tweet, 'anger', val + 1)
        elif emo_el == 1:
            val = emotions.iloc[current_tweet]['fear']
            emotions.set_value(current_tweet, 'fear', val + 1)
        elif emo_el == 2:
            val = emotions.iloc[current_tweet]['anticipation']
            emotions.set_value(current_tweet, 'anticipation', val + 1)
        elif emo_el == 3:
            val = emotions.iloc[current_tweet]['trust']
            emotions.set_value(current_tweet, 'trust', val + 1)
        elif emo_el == 4:
            val = emotions.iloc[current_tweet]['surprise']
            emotions.set_value(current_tweet, 'surprise', val + 1)
        elif emo_el == 5:
            val = emotions.iloc[current_tweet]['sadness']
            emotions.set_value(current_tweet, 'sadness', val + 1)
        elif emo_el == 6:
            val = emotions.iloc[current_tweet]['joy']
            emotions.set_value(current_tweet, 'joy', val + 1)
        elif emo_el == 7:
            val = emotions.iloc[current_tweet]['disgust']
            emotions.set_value(current_tweet, 'disgust', val + 1)

        # sentiment analysis
        sent_el = df.iloc[i]['sentiment']
        if sent_el == 0:
            val = sentiment.iloc[current_tweet]['positive_sentiment']
            sentiment.set_value(current_tweet, 'positive_sentiment', val + 1)
        elif sent_el == 1:
            val = sentiment.iloc[current_tweet]['negative_sentiment']
            sentiment.set_value(current_tweet, 'negative_sentiment', val + 1)

    # Normalize data
    def normalize_df(df):
        for colname, col in df.iteritems():
            df[colname] = df[colname] / df[colname].max()
        return df

    emotions = normalize_df(emotions)
    sentiment = normalize_df(sentiment)

    # Concat to the original dataframe
    result = pd.concat([mydf, emotions], axis=1)
    result = pd.concat([result, sentiment], axis=1)

    return result

    # # Output analysis as json
    # with open(ANALYSIS_FILE, 'w') as f:
    #     json.dump(data, f)
    #     # print('Successfully wrote '+str(len(data))+' entries to file: '+ANALYSIS_FILE)
