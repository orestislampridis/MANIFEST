import pandas as pd
from nltk.corpus import wordnet

words = pd.read_csv('emowords.csv', delimiter=";")
new_words = pd.DataFrame(columns=['word', 'emotion'])

for i, row in words.iterrows():
    synonyms = []
    for syn in wordnet.synsets(row['word']):
        for l in syn.lemmas():
            synonyms.append(l.name())
        # print(set(synonyms))
        for el in set(synonyms):
            new_words = new_words.append({'word': el, 'emotion': row['emotion']}, ignore_index=True)

df = new_words.drop_duplicates('word', inplace=False)
frames = [words, df]
result = pd.concat(frames)

affect_words_csv = result.to_csv('affect_words.csv', index=None,
                                 header=True)  # Don't forget to add '.csv' at the end of the path
