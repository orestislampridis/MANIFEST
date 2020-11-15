import re

import pandas as pd
from personality_features import get_lang_based_scores


# function to clean relics of dataset
def clean_relics(text):
    text = re.sub(r"RT", "", text)
    text = re.sub(r"#USER#", "", text)
    text = re.sub(r"#HASHTAG#", "", text)
    text = re.sub(r"#URL#", "", text)
    return text


# read our dataset
tweets = pd.read_csv("../dataset/data_csv/data_separated.csv", sep=",", encoding="utf8")
tweets = tweets.drop(['ground_truth'], axis=1)
tweets.rename(columns={'tweet_text': 'text'}, inplace=True)

tweets['text'] = tweets['text'].apply(clean_relics)

users_with_personality = get_lang_based_scores(tweets)  # returns a df with user_id and personality scores
users_with_personality.to_csv("users_with_personality.csv", index=False)
