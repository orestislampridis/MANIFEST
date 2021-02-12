import re

import pandas as pd


def clean(text):
    # Format tweets in a single line
    text = re.sub(r'\r\n', ' ', text)

    return text


if __name__ == "__main__":
    df = pd.read_csv("../output/us_elections_results.csv")

    df['tweet_text'] = df.tweet_text.apply(clean)

    print(df['lr_pred'].value_counts())

    s0 = df.bb_pred[df.bb_pred.eq(0)].sample(random_state=42, n=100).index
    s1 = df.bb_pred[df.bb_pred.eq(1)].sample(random_state=42, n=100).index

    df = df.loc[s0.union(s1)]
    print(df)
    df_eval = df[['idx', 'tweet_text']]
    df_eval.to_csv("../output/us_elections_eval_dataset.csv", mode='w', index=False)
