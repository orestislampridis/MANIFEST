import pandas as pd
from matplotlib import pyplot as plt
from numpy import mean
from numpy import std
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("../scaled_final_complete_features_with_labels_and_ids", sep=",", encoding="utf8")

data_tfidf = df[[str(x) for x in range(1000)]]
data_readability = df[
    ['avg_word_count', 'emoji_count', 'slang_count', 'capitalized_count', 'full_capitalized_count',
     'retweets_count', 'user_mentions_count', 'hashtags_count', 'url_count']]
data_sentiment = df[
    ['anger', 'fear', 'joy', 'sadness', 'negation', 'vader_compound_score', 'textblob_polarity_score']]
data_personality = df[['extraversion', 'avoidance', 'conscientiousness', 'openness', 'neuroticism',
                       'agreeableness', 'anxiety']]
data_gender = df[['gender']]
data_liwc = df[['Analytic', 'Clout', 'Authentic', 'Tone']]
data_ground_truth = df[['ground_truth']]

i = 0
feature_names = [
    # "explanations_tfidf",
    "phase_C_tfidf_readability_sentiment_personality_gender",
    # "explanations_readability_sentiment_personality_gender",
    # "explanations_readability_sentiment_personality_gender_liwc"
]

features = list()
# features.append([data_tfidf, data_ground_truth])
features.append([data_tfidf, data_readability, data_sentiment, data_personality, data_gender, data_ground_truth])
# features.append([data_readability, data_sentiment, data_personality, data_gender, data_ground_truth])
# features.append([data_readability, data_sentiment, data_personality, data_gender, data_liwc, data_ground_truth])

for feature_combination in features:
    print("feature_combination: " + str(feature_names[i]))

    features = pd.concat([i for i in feature_combination], axis=1)

    # print(features)
    features.to_csv('features_with_labels_and_ids')

    X = features.drop(['ground_truth'], axis=1).reset_index(drop=True)
    y = features[['ground_truth']].values.ravel()


# get a list of models to evaluate
def get_models():
    models = dict()

    for i in range(4, 105, 10):
        print(i)
        rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=i)
        model = GradientBoostingClassifier()
        models[str(i)] = Pipeline(steps=[('s', rfe), ('m', model)])

    for i in range(204, 925, 100):
        print(i)
        rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=i)
        model = GradientBoostingClassifier()
        models[str(i)] = Pipeline(steps=[('s', rfe), ('m', model)])

    i = 1024
    print(i)
    rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=i)
    model = GradientBoostingClassifier()
    models[str(i)] = Pipeline(steps=[('s', rfe), ('m', model)])

    return models


# evaluate a give model using cross-validation
def evaluate_model(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    return scores


# summarize the dataset
print(X.shape, y.shape)

# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
    scores = evaluate_model(model, X, y)
    results.append(scores)
    names.append(name)
    print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
# plot model performance for comparison
plt.boxplot(results, labels=names, showmeans=True)
plt.show()
