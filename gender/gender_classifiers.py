import pickle
import warnings

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import fake_news_spreader_feature_extraction as feature_extraction
from utils import random_search, grid_search
from utils.preprocessing import clean_text

warnings.filterwarnings('ignore')

file_tweets = "new_gender_combined.csv"
file_genders = "ground_truth_gender.csv"

data_tweets = pd.read_csv(file_tweets, sep=",", encoding="utf8")
data_genders = pd.read_csv(file_genders, sep=",", encoding="utf8").drop(['tweets'], axis=1)

# Join the two dataframes together
merged_df = pd.merge(data_tweets, data_genders, on='twitter_uid', how='inner')
merged_df = merged_df.dropna().reset_index()

data_combined = merged_df[['twitter_uid', 'statuses', 'Gender', 'tweets', 'retweets', 'urls']]
data_combined.rename(
    columns={'statuses': 'text', 'twitter_uid': 'user_id', 'retweets': 'retweets_count', 'urls': 'url_count'},
    inplace=True)

print(data_combined.columns)

# count various readability features
data_readability = feature_extraction.get_features_for_gender(data_combined)
data_other = data_combined[['retweets_count', 'url_count']]

data_combined['text'] = data_combined.text.apply(clean_text)

vectorizer = TfidfVectorizer(max_features=1000, min_df=0.01, max_df=0.90, ngram_range=(1, 4))
vectors = vectorizer.fit_transform(data_combined['text'])

# save sparse tfidf vectors to dataframe to use with other features
vectors_pd = pd.DataFrame(vectors.toarray())

# save tfidf vector
pickle.dump(vectorizer, open("tfidf_gender.pkl", "wb"))

features = pd.concat([vectors_pd, data_readability, data_other], axis=1).drop(['user_id'], axis=1) \
    .reset_index(drop=True)
y = data_combined['Gender']

# use scaler to scale our data to [0,1] range
x = features.values  # returns a numpy array
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)
X = pd.DataFrame(x_scaled, columns=features.columns)
print(X)
print(X.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# try many classifiers to find the best performing
clf_names = [
    # "KNN",
    # "Naive Bayes",
    "Logistic Regression",
    "SVM",
    "Random Forest",
    "Gradient Boosting"
]

best_clf = None
best_classifier = ""
best_accuracy = 0

for name in clf_names:
    print("Classifier:", name)

    if name == "KNN":
        clf = random_search.get_knn_random_grid(X_train, y_train)

    if name == "Naive Bayes":
        clf = random_search.get_NB_random_grid(X_train, y_train)

    if name == "Logistic Regression":
        # clf = random_search.get_logistic_regression_random_grid(X_train, y_train)
        clf = grid_search.get_logistic_regression_grid_search(X_train, y_train)

    if name == "SVM":
        # clf = random_search.get_SVM_random_grid(X_train, y_train)
        clf = grid_search.get_SVM_grid_search(X_train, y_train)

    if name == "Random Forest":
        # clf = random_search.get_random_forest_random_grid(X_train, y_train)
        clf = grid_search.get_random_forest_grid_search(X_train, y_train)

    if name == "Gradient Boosting":
        # clf = random_search.get_gradient_boosting_random_grid(X_train, y_train)
        clf = grid_search.get_gradient_boosting_grid_search(X_train, y_train)

    clf.fit(X_train, y_train)
    y_preds = clf.predict(X_test)

    # Training accuracy
    print("The training accuracy is: ")
    print(accuracy_score(y_train, clf.predict(X_train)))

    # Test accuracy
    print("The test accuracy is: ")
    print(accuracy_score(y_test, y_preds))

    # Other metrics
    print("Precission:", precision_score(y_test, y_preds, average='macro'))
    print("Recal:", recall_score(y_test, y_preds, average='macro'))
    print("f1_score:", f1_score(y_test, y_preds, average='macro'))
    print("\n")

    if best_accuracy < accuracy_score(y_test, y_preds):
        best_clf = clf
        best_classifier = name
        best_accuracy = accuracy_score(y_test, y_preds)

print("Best classifier:", best_classifier)
print("Best accuracy:", best_accuracy)

# save the model to disk
filename = '../classifiers/gender/' + str(best_classifier) + '_' + str(best_accuracy) + '_final.sav'
pickle.dump(best_clf, open(filename, 'wb'))
