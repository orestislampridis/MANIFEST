import pandas as pd

file_gender = "ground_truth_gender.csv"

data = pd.read_csv(file_gender, sep=",", encoding="utf8", index_col=0)
print(data.head())

# Keep only the columns we can use as features
data = data[[[['Gender', 'Avg_Word_length', 'Avg_Punctuation', 'Avg_Capitalized_Words', 'Avg_Slang_words', 'Avg_emojis',
               'Avg_words_found', 'tweets', 'retweets', 'urls']]]]
