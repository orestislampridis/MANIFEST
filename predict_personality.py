import pickle
import pandas as pd
from personality.preprocessing import clean_text


def list2string(list):
    return ','.join(map(str, list))


def main():
    data_separated = pd.read_csv("dataset/data_csv/data_separated.csv", sep=",", encoding="utf8")
    data_combined = pd.read_csv("dataset/data_csv/data_combined.csv", sep=",", encoding="utf8")

    print(data_separated)
    print(data_combined)
    print(data_combined.columns)
    print(data_combined['tweet_text'])

    data_combined['tweet_text'] = data_combined.tweet_text.apply(clean_text)
    data_combined['tweet_text'] = [list2string(list) for list in data_combined['tweet_text']]

    print(data_combined['tweet_text'])

    # Load pickled personality classifier and predict
    filename = 'classifiers/personality/classifier_chains_SVC'
    classifier = pickle.load(open(filename, 'rb'))
    preds = classifier.predict(data_combined['tweet_text'])

    preds = preds.toarray().astype(dtype=int)

    print("predictions: ", preds)


if __name__ == "__main__":
    main()
