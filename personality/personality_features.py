"""
Taken from https://github.com/demeterkara/personality-modeling
"""

import pickle
import re
import string

import joblib
import nltk
import numpy as np
import pandas as pd
import scipy as sp
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from emo_lexicons import emolex_emotion2
from wnaffect.wnaffect import WNAffect


def text_cleaning(df, text_col):
    stop = stopwords.words('english')  # define stopwords list

    # cleaning
    df['cwords'] = text_col.str.replace(
        'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
        ' ')  # .replace('http\S+', ' ')
    df['cwords'] = df['cwords'].str.replace('RT @.*?(?=\s|$)', ' ')  # clean RT
    df['cwords'] = df['cwords'].str.replace('@.*?(?=\s|$)', ' ')  # clean mentions
    df['cwords'] = df['cwords'].apply(lambda x: " ".join(x.lower() for x in x.split()))  # to lowercase
    df['cwords'] = df['cwords'].apply(lambda x: ''.join([i for i in x if i not in string.digits]))  # remove digits
    df['cwords'] = df['cwords'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))  # remove stopwords
    df['cwords'] = df['cwords'].str.replace('[^\w\s#@/:%.,_-]', '', flags=re.UNICODE)  # remove unicodes and emojis
    df['cwords'] = df['cwords'].apply(
        lambda x: ''.join([i for i in x if i not in string.punctuation]))  # remove punctuations
    # Tokenization
    df['words'] = df['cwords'].apply(lambda x: word_tokenize(x))
    # Lemmatization
    WNL = nltk.WordNetLemmatizer()
    df['lwords'] = df['words'].apply(lambda x: [WNL.lemmatize(y) for y in x])
    # POS tagging
    df['pos_tag'] = df['words'].apply(pos_tag)
    return df


def dummy_fun(doc):
    return doc


def tfidf(col):
    transformer = TfidfTransformer()

    def dummy(doc):
        return doc

    weightFile = open("models/tfidf_voc.pkl", "rb")
    voc = pickle.load(weightFile, encoding='latin1')
    loaded_vec = CountVectorizer(decode_error="ignore", tokenizer=dummy, preprocessor=dummy, vocabulary=voc)
    # loaded_vec = CountVectorizer(decode_error="replace", tokenizer=dummy, preprocessor=dummy,vocabulary=pickle.load(open("tfidf_voc.pkl", "rb")))
    tfidf = transformer.fit_transform(loaded_vec.fit_transform(col))

    return tfidf


def vectorizer(col):
    def dummy_fun(doc):
        return doc

    vectorizer = CountVectorizer(tokenizer=dummy_fun, preprocessor=dummy_fun)
    vect = vectorizer.fit_transform(col)
    return vect


def posNgrams(text_tags, n):
    taglist = []
    output = []
    for item in text_tags:
        taglist.append(item[1])
    for i in range(len(taglist) - n + 1):
        g = '_'.join(taglist[i:i + n])
        output.append(g)
    return output


def get_text_features(df):
    def dummy(doc):
        return doc

    # Tokenization
    df['words'] = df['cwords'].apply(lambda x: word_tokenize(x))

    # Lemmatization
    WNL = nltk.WordNetLemmatizer()
    df['lwords'] = df['words'].apply(lambda x: [WNL.lemmatize(y) for y in x])

    # POS tagging
    df['pos_tag'] = df['words'].apply(pos_tag)

    # create a tag vector
    df['tags'] = df['pos_tag'].apply(lambda x: [pos for word, pos in (x)])

    # bigrams tagging
    df['bigrams_pos_tag'] = df['pos_tag'].apply(lambda x: posNgrams(x, 2))
    # trigrams tagging
    df['trigrams_pos_tag'] = df['pos_tag'].apply(lambda x: posNgrams(x, 3))

    # get bi_tri_grams
    cv = CountVectorizer(ngram_range=(2, 4), token_pattern=r'\b\w+\b', min_df=1, decode_error="ignore", tokenizer=dummy,
                         preprocessor=dummy, vocabulary=pickle.load(open("models/person_ngrams_voc.pkl", "rb")))
    bi_tri_grams = cv.fit_transform(df['words'])

    # get tfidf of lemmatized words
    # Tfidf
    transformer = TfidfTransformer()
    loaded_vec = CountVectorizer(decode_error="replace", tokenizer=dummy, preprocessor=dummy,
                                 vocabulary=pickle.load(open("models/person_tfidf_voc.pkl", "rb")))
    tfidf_lwords = transformer.fit_transform(loaded_vec.fit_transform(df['lwords']))

    # get POS tag features

    vect = CountVectorizer(tokenizer=dummy, preprocessor=dummy,
                           vocabulary=pickle.load(open("models/person_tag_voc.pkl", "rb")))
    tag_vector = vect.fit_transform(df['tags'])

    vect = CountVectorizer(tokenizer=dummy, preprocessor=dummy,
                           vocabulary=pickle.load(open("models/person_bigramstag_voc.pkl", "rb")))
    bigrams_tag_vector = vect.fit_transform(df['bigrams_pos_tag'])

    vect = CountVectorizer(tokenizer=dummy, preprocessor=dummy,
                           vocabulary=pickle.load(open("models/person_trigramstag_voc.pkl", "rb")))
    trigrams_tag_vector = vect.fit_transform(df['trigrams_pos_tag'])

    return df, bi_tri_grams, tfidf_lwords, tag_vector, bigrams_tag_vector, trigrams_tag_vector


def emolex_emotions(df):
    df = emolex_emotion2.emotion_extraction(df)
    df = df.fillna(0)
    return (df)


def primary_emotion_extraction_WNA(mydf):
    wna = WNAffect('wnaffect/wordnet-1.6/', 'wnaffect/wn-domains-3.2/')

    primary_emotions = ['anger', 'disgust', 'negative-fear', 'joy', 'sadness', 'surprise', 'trust',
                        'positive-emotion', 'negative-emotion']

    # Detect emotions based on wna
    data = []
    for i in range(0, len(mydf)):
        pos_tag = mydf.iloc[i]['pos_tag']
        emotions = []
        for word, pos in pos_tag:
            emo = wna.get_emotion(word, pos)
            emo_name = str(emo)
            if (emo_name != "None"):
                while (not (any(emo_name in x for x in primary_emotions))):
                    emo = emo.get_level(emo.level - 1)
                    emo_name = str(emo)
                emotions.append(emo_name)
        data.append(emotions)

    # Analyze detected emotions
    emo = pd.DataFrame.from_records(data)
    df = pd.DataFrame(0, index=np.arange(0, len(emo)),
                      columns=['anger_wna', 'disgust_wna', 'fear_wna', 'joy_wna', 'sadness_wna', 'surprise_wna',
                               'trust_wna', 'positive-emotion_wna', 'negative-emotion_wna'])

    columns = emo.columns
    for i in range(0, len(emo)):
        for col in columns:
            # emotions analysis
            emo_el = emo.iloc[i][col]
            if emo_el == 'anger':
                val = df.iloc[i]['anger_wna']
                df.set_value(i, 'anger_wna', val + 1)
            elif emo_el == 'negative-fear':
                val = df.iloc[i]['fear_wna']
                df.set_value(i, 'fear_wna', val + 1)
            elif emo_el == 'trust':
                val = df.iloc[i]['trust_wna']
                df.set_value(i, 'trust_wna', val + 1)
            elif emo_el == 'surprise':
                val = df.iloc[i]['surprise_wna']
                df.set_value(i, 'surprise_wna', val + 1)
            elif emo_el == 'sadness':
                val = df.iloc[i]['sadness_wna']
                df.set_value(i, 'sadness_wna', val + 1)
            elif emo_el == 'joy':
                val = df.iloc[i]['joy_wna']
                df.set_value(i, 'joy_wna', val + 1)
            elif emo_el == 'positive-emotion':
                val = df.iloc[i]['positive-emotion_wna']
                df.set_value(i, 'positive-emotion_wna', val + 1)
            elif emo_el == 'negative-emotion':
                val = df.iloc[i]['negative-emotion_wna']
                df.set_value(i, 'negative-emotion_wna', val + 1)

    # Normalize data
    def normalize_df(df):
        for colname, col in df.iteritems():
            df[colname] = df[colname] / df[colname].max()
        return df

    df = df.apply(pd.to_numeric, errors='coerce')
    df = normalize_df(df)
    df = df.fillna(0)

    # Concat to the original dataframe
    result = pd.concat([mydf, df], axis=1)

    return result


def emo_wna_emotions(mydf):
    mydf['anger'] = mydf['anger_emo'] + mydf['anger_wna']
    mydf['anger'] = mydf['anger'] / mydf['anger'].max()

    mydf['fear'] = mydf['fear_emo'] + mydf['fear_wna']
    mydf['fear'] = mydf['fear'] / mydf['fear'].max()

    mydf['trust'] = mydf['trust_emo'] + mydf['trust_wna']
    mydf['trust'] = mydf['trust'] / mydf['trust'].max()

    mydf['surprise'] = mydf['surprise_emo'] + mydf['surprise_wna']
    mydf['surprise'] = mydf['surprise'] / mydf['surprise'].max()

    mydf['sadness'] = mydf['sadness_emo'] + mydf['sadness_wna']
    mydf['sadness'] = mydf['sadness'] / mydf['sadness'].max()

    mydf['joy'] = mydf['joy_emo'] + mydf['joy_wna']
    mydf['joy'] = mydf['joy'] / mydf['joy'].max()

    mydf['disgust'] = mydf['disgust_emo'] + mydf['disgust_wna']
    mydf['disgust'] = mydf['disgust'] / mydf['disgust'].max()

    mydf['positive_sentiment_lex'] = mydf['positive_sentiment_emo'] + mydf['positive-emotion_wna']
    mydf['positive_sentiment_lex'] = mydf['positive_sentiment_lex'] / mydf['positive_sentiment_lex'].max()

    mydf['negative_sentiment_lex'] = mydf['negative_sentiment_emo'] + mydf['negative-emotion_wna']
    mydf['negative_sentiment_lex'] = mydf['negative_sentiment_lex'] / mydf['negative_sentiment_lex'].max()

    mydf.drop(
        ['anger_emo', 'anger_wna', 'fear_emo', 'fear_wna', 'trust_emo', 'trust_wna', 'surprise_emo', 'surprise_wna',
         'sadness_emo', 'sadness_wna', 'joy_emo', 'joy_wna', 'disgust_emo', 'disgust_wna', 'positive_sentiment_emo',
         'positive-emotion_wna', 'negative_sentiment_emo', 'negative-emotion_wna'], axis=1, inplace=True)

    return mydf


def emojis_emotions(df, text_col):
    def emoji_affect_score(affect_list, text):
        # load df - emojis with sentiment
        emoji_sent_df = pd.read_csv('emo_lexicons/emojis_sent_score.csv', sep=';')
        score = []
        for emo in affect_list:
            # text = text.encode('utf-8')
            if emo in text:
                score.append(emoji_sent_df[emoji_sent_df['Unicode'] == emo]['Sentiment score emoji'].values[0])
        return np.mean(score)

    # clean
    new_df = pd.DataFrame()
    new_df['words'] = text_col.str.replace(
        'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
        ' ')  # clean links
    new_df['words'] = new_df['words'].str.replace('RT @.*?(?=\s|$)', ' ')  # clean RT
    new_df['words'] = new_df['words'].str.replace('@.*?(?=\s|$)', ' ')  # clean mentions

    # emojis with affects
    joy_emojis_f = ['U0001f600', 'U0001f602', 'U0001f603', 'U0001f604', 'U0001f606', 'U0001f607', 'U0001f609',
                    'U0001f60a', 'U0001f60e', 'U0001f60f', 'U0001f31e', 'U000263a', 'U0001f60b', 'U0001f60c',
                    'U0001f60d']
    joy_emojis_s = ['U0001f618', 'U0001f61c', 'U0001f61d', 'U0001f61b', 'U0001f63a', 'U0001f638', 'U0001f639',
                    'U0001f63b', 'U0001f63c', 'U0002764', 'U0001f496', 'U0001f495', 'U0001f601', 'U0002665']
    anger_emojis = ['U0001f62c', 'U0001f620', 'U0001f610', 'U0001f611', 'U0001f620', 'U0001f621', 'U0001f616',
                    'U0001f624', 'U0001f63e']
    disgust_emojis = ['U0001f4a9']
    fear_emojis = ['U0001f605', 'U0001f626', 'U0001f627', 'U0001f631', 'U0001f628', 'U0001f630', 'U0001f640']
    sad_emojis = ['U0001f614', 'U0001f615', 'U0002639', 'U0001f62b', 'U0001f629', 'U0001f622', 'U0001f625', 'U0001f62a',
                  'U0001f613', 'U0001f62d', 'U0001f63f', 'U0001f494']
    surprise_emojis = ['U0001f633', 'U0001f62f', 'U0001f635', 'U0001f632']

    # sentiment scores from emojis with affect
    new_df['joy_emojis_f'] = new_df['words'].apply(lambda x: emoji_affect_score(joy_emojis_f, x))
    new_df['joy_emojis_s'] = new_df['words'].apply(lambda x: emoji_affect_score(joy_emojis_s, x))
    new_df['joy_emojis'] = new_df.fillna(0)['joy_emojis_f'] + new_df.fillna(0)['joy_emojis_s']
    new_df.drop(['joy_emojis_s', 'joy_emojis_f'], axis=1, inplace=True)

    new_df['anger_emojis'] = new_df['words'].apply(lambda x: emoji_affect_score(anger_emojis, x)).fillna(0)
    new_df['disgust_emojis'] = new_df['words'].apply(lambda x: emoji_affect_score(disgust_emojis, x)).fillna(0)
    new_df['fear_emojis'] = new_df['words'].apply(lambda x: emoji_affect_score(fear_emojis, x)).fillna(0)
    new_df['sad_emojis'] = new_df['words'].apply(lambda x: emoji_affect_score(sad_emojis, x)).fillna(0)
    new_df['surprise_emojis'] = new_df['words'].apply(lambda x: emoji_affect_score(surprise_emojis, x)).fillna(0)

    df['joy_emojis'] = new_df['joy_emojis'].values
    df['joy_emojis'] = df['joy_emojis'].abs()
    df['anger_emojis'] = new_df['anger_emojis'].values
    df['anger_emojis'] = df['anger_emojis'].abs()
    df['disgust_emojis'] = new_df['disgust_emojis'].values
    df['disgust_emojis'] = df['disgust_emojis'].abs()
    df['fear_emojis'] = new_df['fear_emojis'].values
    df['fear_emojis'] = df['fear_emojis'].abs()
    df['sad_emojis'] = new_df['sad_emojis'].values
    df['sad_emojis'] = df['sad_emojis'].abs()
    df['surprise_emojis'] = new_df['surprise_emojis'].values
    df['surprise_emojis'] = df['surprise_emojis'].abs()

    return df


# requires preprocessed df, with cword and pos_tag column
def get_emotion_features(df):
    # Emotions from EMOLEX
    df = emolex_emotions(df)

    # Emotions from WNAffect
    df = primary_emotion_extraction_WNA(df)

    # combine
    df = emo_wna_emotions(df)

    # emojis
    df = emojis_emotions(df, df['text'])

    return df


# Requires preprocessed df with get_emotion_features
def predict_emotion(df):
    df = get_emotion_features(df)

    # flexicon sentiments
    sentiment = df[['positive_sentiment_lex', 'negative_sentiment_lex']].values

    # Emotion Proxys Feature Vectors - E = <em1, em2, .., em6> anger fear sadness disgust joy surprise
    emotions = df[['anger', 'fear', 'sadness', 'disgust', 'joy', 'surprise']].values

    # Affective Emojis Vectors - AE = < aes1, aes2, ..., aes6> anger fear sadness disgust joy surprise
    emojis = df[['anger_emojis', 'fear_emojis', 'sad_emojis', 'disgust_emojis', 'joy_emojis', 'surprise_emojis']].values

    transformer = TfidfTransformer()

    def dummy(doc):
        return doc

    weightFile = open("models/tfidf_voc.pkl", "rb")
    voc = pickle.load(weightFile, encoding='latin1')
    loaded_vec = CountVectorizer(decode_error="ignore", tokenizer=dummy, preprocessor=dummy, vocabulary=voc)
    # loaded_vec = CountVectorizer(decode_error="replace", tokenizer=dummy, preprocessor=dummy,vocabulary=pickle.load(open("tfidf_voc.pkl", "rb")))
    tfidf = transformer.fit_transform(loaded_vec.fit_transform(df['lwords']))

    # Feauture Union
    X = sp.sparse.hstack((tfidf, emotions, sentiment, emojis), format='csr')

    # load pretrained model and
    # annotate the original dataset
    emotion_model = joblib.load('models/emo_detector_model.sav')
    emotions = emotion_model.predict(X)
    emotions = pd.DataFrame.from_records(emotions, columns=['anger', 'fear', 'sadness', 'disgust', 'joy', 'surprise'])

    # drop features used for emotion detection
    df.drop(['anger', 'fear', 'trust', 'sadness', 'disgust', 'joy', 'surprise', 'positive_sentiment_lex',
             'negative_sentiment_lex', 'joy_emojis', 'anger_emojis', 'disgust_emojis', 'fear_emojis',
             'sad_emojis', 'surprise_emojis'], axis=1, inplace=True)

    # concat the emotion detected to tweets df
    df = pd.concat([df, emotions], axis=1)

    return df, emotions


def tweets_analysis(tweets_df):
    tweets_df['tweet_length'] = tweets_df['text'].str.len()

    # average word length
    def avg_word(sentence):
        words = sentence.split()
        if len(words) != 0:
            res = sum(len(word) for word in words) / len(words)
        else:
            res = 0
        return (res)

    tweets_df['avg_word'] = tweets_df['text'].astype('str').apply(lambda x: avg_word(x))

    # no of uppercase words
    tweets_df['upper'] = tweets_df['text'].astype('str').apply(lambda x: len([x for x in x.split() if x.isupper()]))

    # no of external links in each tweet
    tweets_df['no_of_links'] = [len(c) for c in (tweets_df['text'].str.findall(
        'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'))]

    # remove links in order not to cause problems with # in hashtags
    tweets_df['text'] = tweets_df['text'].str.replace(
        'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
        ' ')
    # no of hashtags in each tweet

    tweets_df['hashtags'] = tweets_df['text'].str.findall(r'#.*?(?=\s|$)')
    tweets_df['no_of_hashtags'] = [len(c) for c in (tweets_df['text'].str.findall(r'#(\w+)'))]

    # no of user mentions
    tweets_df['no_of_user_mentions'] = [len(c) for c in (tweets_df['text'].str.findall(r'@.*?(?=\s|$)'))]

    # retweets
    tweets_df['is_a_retweet'] = tweets_df['text'].str.startswith('RT')  # count the Trues

    return tweets_df


def features_extraction(user_df, tweets_df):
    # user behavioral/demographic features
    user_df = user_df.rename(columns={'twitter_id': 'user_id'})

    # # days at the service
    # user_df['days_at_Twitter'] = (datetime.datetime.today() - pd.to_datetime((user_df['created_at'])).dt.days)

    # length of screenname and description
    user_df['screename_length'] = user_df['screen_name'].str.len()
    user_df['description_length'] = user_df['description'].str.len()

    # # # frequency of status updates
    # user_df['freq_status_updates'] = user_df['statuses_count'] / user_df['days_at_Twitter']

    # features extracted from Tweets
    # tweets_df.drop(['_id', 'created_at'], axis=1, inplace=True)
    tweets_df.set_index(['user_id'])

    # group and merge dfs

    group_df = tweets_df.sort_values(by=['user_id']).groupby('user_id', group_keys=False).agg(
        {'favorite_count': ['mean'],
         'retweet_count': ['mean'],
         'tweet_length': ['mean'],
         'no_of_hashtags': ['mean'],
         'no_of_links': ['mean'],
         'no_of_user_mentions': ['mean'],
         'avg_word': ['mean'],
         'upper': ['mean'],
         'joy': ['mean'],
         'sadness': ['mean'],
         'fear': ['mean'],
         'anger': ['mean'],
         'disgust': ['mean'],
         'surprise': ['mean']})

    group_df.columns = group_df.columns.get_level_values(0)
    user_df = user_df.set_index(['user_id'])
    users_data = user_df.join(group_df)
    return users_data


def get_behavioral_features(user_df, tweets_df):
    # outliers removal based on certain columns
    # statuses count, friends count, followers count
    #     user_df = user_df[np.abs(user_df.statuses_count - user_df.statuses_count.mean()) <= (
    #                 3 * user_df.statuses_count.std())]  # keep only the ones that are within +3 to -3 standard deviations in the column 'Data'.
    #     user_df = user_df[np.abs(user_df.followers_count - user_df.followers_count.mean()) <= (
    #                 3 * user_df.followers_count.std())]  # keep only the ones that are within +3 to -3 standard deviations in the column 'Data'.
    #     user_df = user_df[np.abs(user_df.friends_count - user_df.friends_count.mean()) <= (
    #                 3 * user_df.friends_count.std())]  # keep only the ones that are within +3 to -3 standard deviations in the column 'Data'.

    # Extract features from tweets
    tweets_df = tweets_analysis(tweets_df)

    # Extract features from users
    users_data = features_extraction(user_df, tweets_df)

    users_data = users_data.sort_values(by=['user_id'])

    # Create the behavior dataframe
    behavior = users_data[
        ['description_length', 'favorite_count', 'favourites_count', 'followers_count', 'friends_count',
         'listed_count', 'no_of_hashtags', 'no_of_links', 'no_of_user_mentions', 'retweet_count', 'screename_length',
         'statuses_count', 'tweet_length', 'upper']].copy()

    # fill na with the mean of every column
    behavior = behavior.fillna(behavior.mean())

    return users_data, behavior


def get_personality_scores(user_df, text_df):
    # basic tect preprocessing
    text_df = text_cleaning(text_df, text_df['text'])

    # extract emotion features
    text_df, emotions = predict_emotion(text_df)
    # extract behavior features
    user_df, behavior = get_behavioral_features(user_df, text_df)

    # preprocess for user-level text features
    text_df = text_df.sort_values(by=['user_id']).groupby('user_id', group_keys=False).agg(
        {'cwords': lambda x: '. '.join(x)})
    # keep only indexes in user-df
    text = text_df[text_df.index.isin(user_df.index)]

    # extract text features
    text, bi_tri_grams, tfidf_lwords, tag_vector, bigrams_tag_vector, trigrams_tag_vector = get_text_features(text)

    # user-level emotions
    emotions = user_df[['joy', 'anger', 'fear', 'sadness', 'disgust', 'surprise']].copy()

    # Normalization
    behavior_scaler = joblib.load('models/behavior_scaler.sav')
    emotion_scaler = joblib.load('models/emotion_scaler.sav')
    behavior = pd.DataFrame(behavior_scaler.transform(behavior), index=behavior.index, columns=behavior.columns)
    emotions = pd.DataFrame(emotion_scaler.transform(emotions), index=emotions.index, columns=emotions.columns)

    # Feature Union
    X = sp.sparse.hstack(
        (bi_tri_grams, tfidf_lwords, tag_vector, bigrams_tag_vector, trigrams_tag_vector, behavior, emotions),
        format='csr')

    # Load model and annotate
    personality_model = joblib.load('models/chain_holistic_model_4150623.sav')  # scikit 0.22rc2.post1
    personality_scores = personality_model.predict(X)
    personality_scores = pd.DataFrame.from_records(personality_scores,
                                                   columns=['E', 'AVOIDANCE', 'A', 'ANXIETY', 'N', 'O', 'C'])

    user_df.reset_index(inplace=True)
    df = pd.concat([user_df, personality_scores], axis=1)

    return df


def get_lang_based_scores(text_df):
    text_df = text_cleaning(text_df, text_df['text'])

    # preprocess for user-level text features
    text_df = text_df.sort_values(by=['user_id']).groupby('user_id', group_keys=False).agg(
        {'cwords': lambda x: '. '.join(x)})

    text, bi_tri_grams, tfidf_lwords, tag_vector, bigrams_tag_vector, trigrams_tag_vector = get_text_features(text_df)

    # concat features
    X = sp.sparse.hstack((bi_tri_grams, tfidf_lwords, tag_vector, bigrams_tag_vector, trigrams_tag_vector),
                         format='csr')

    # load pretrained OCEAN model and annotate the original dataset
    personality_model = joblib.load('models/chain_lang_holistic_model_4132650.sav')  # scikit 0.22rc2.post1
    personality_scores = personality_model.predict(X)
    personality_scores = pd.DataFrame.from_records(personality_scores,
                                                   columns=['E', 'AVOIDANCE', 'C', 'O', 'N', 'A', 'ANXIETY'])

    text_df.reset_index(inplace=True)
    df = pd.concat([text_df['user_id'], personality_scores], axis=1)
    return df
