import pandas as pd


def count_words(list):
    return ','.join(map(str, list))


def main():
    data_separated = pd.read_csv("dataset/data_csv/data_separated.csv", sep=",", encoding="utf8")
    data_combined = pd.read_csv("dataset/data_csv/data_combined.csv", sep=",", encoding="utf8")

    print(data_separated)
    print(data_combined)
    print(data_separated)

    data_combined['avg_word_count'] = data_combined['tweet_text'].str.split().str.len() / 300
    data_combined['avg_sent_len'] = data_combined['tweet_text'].str.split('.')

    print(data_combined['avg_sent_len'][1])
    print(len(data_combined['avg_sent_len'][1]))
    print(data_combined)

    # data_combined['tweet_text'] = data_combined.tweet_text.apply(clean_text)
    # data_combined['tweet_text'] = [list2string(list) for list in data_combined['tweet_text']]

    # print(data_combined['tweet_text'])


if __name__ == "__main__":
    main()
