import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from skmultilearn.problem_transform import ClassifierChain

from utils.preprocessing import clean_text


def list2string(list):
    return ','.join(map(str, list))


file_tweets = "new_personality_combined.csv"
file_personalities = "personality-data.txt"

data_tweets = pd.read_csv(file_tweets, sep=",", encoding="utf8", index_col=0)
data_personalities = pd.read_csv(file_personalities, sep="\t", encoding="utf8", index_col=10)

print(data_tweets)

# Join the two dataframes together
merged_df = pd.merge(data_tweets, data_personalities, on='twitter_uid', how='inner')
merged_df.reset_index(drop=True, inplace=True)

# Drop the statues (the text)
personality_categories = list(merged_df.columns.values)[2:]

# Print dataset statistics
print("Final number of data in personality dataset =", merged_df.shape[0])
print("Number of personality categories =", len(personality_categories))
print("Personality categories =", ', '.join(personality_categories))

print(merged_df['statuses'])
merged_df['statuses'] = merged_df.statuses.apply(clean_text)
print(merged_df['statuses'])

merged_df['statuses'] = [list2string(list) for list in merged_df['statuses']]

# Split the personality categories into 3 quantiles to convert the problem to classification
bins = 3
labels = [0, 1, 2]

merged_df['ADMIRATION'] = pd.cut(merged_df['ADMIRATION'], bins, labels=labels)
merged_df['AGRE'] = pd.cut(merged_df['AGRE'], bins, labels=labels)
merged_df['ANXIETY'] = pd.cut(merged_df['ANXIETY'], bins, labels=labels)
merged_df['AVOIDANCE'] = pd.cut(merged_df['AVOIDANCE'], bins, labels=labels)
merged_df['CONS'] = pd.cut(merged_df['CONS'], bins, labels=labels)
merged_df['EXTR'] = pd.cut(merged_df['EXTR'], bins, labels=labels)
merged_df['NARCISSISM'] = pd.cut(merged_df['NARCISSISM'], bins, labels=labels)
merged_df['NEUR'] = pd.cut(merged_df['NEUR'], bins, labels=labels)
merged_df['OPEN'] = pd.cut(merged_df['OPEN'], bins, labels=labels)
merged_df['RIVALRY'] = pd.cut(merged_df['RIVALRY'], bins, labels=labels)

print(merged_df)

# Split the data to train and test
train, test = train_test_split(merged_df, random_state=42, test_size=0.25)

x_train = train['statuses']
x_test = test['statuses']

y_train = train.drop(labels=['statuses', 'tweets'], axis=1).to_numpy(dtype=int)
y_test = test.drop(labels=['statuses', 'tweets'], axis=1).to_numpy(dtype=int)

print(y_train)

print(type(x_train))
print(type(y_train))

# Classifier Chains approach
print("---Classifier Chains---")
classifier_chains = Pipeline([
    ('tfidf', TfidfVectorizer(encoding='utf-8', ngram_range=(1, 3))),
    ('clf', ClassifierChain(LinearSVC())),
])

classifier_chains.fit(x_train, y_train)

predictions_chains = classifier_chains.predict(x_test)
predictions_chains = predictions_chains.toarray().astype(dtype=int)

print("predictions: ", predictions_chains)
print(y_test)

# print("accuracy_score:", accuracy_score(y_test, predictions_chains))
# print("Hamming_loss:", hamming_loss(y_test, predictions_chains))

directory = '../classifiers/personality'
pickle.dump(classifier_chains, open(directory + '/classifier_chains_SVC', 'wb'))
