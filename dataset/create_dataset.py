import pandas as pd

from xml_reader import read_tweet_text, read_all


def get_users():
    users = []
    with open("data/truth.txt") as file_in:
        for line in file_in:
            array = line.split(":::")
            user = array[0]
            truth = int(array[1][0])
            users.append([user, truth])
    return users


def create_separated_set():
    data = []
    users = get_users()
    for user in users:
        read_all(data, user[0] + '.xml', user[1])
    return data


def assemble_pandas(dataset, column):
    dataset = pd.DataFrame(dataset)
    dataset.columns = column
    return dataset


def get_combined_dataset():
    users = get_users()
    data = create_combined_set(users)
    columns = ['user_id', 'tweet_text', 'ground_truth']
    dataset = assemble_pandas(data, columns)
    return dataset


def get_separated_dataset():
    data = create_separated_set()
    columns = ['user_id', 'tweet_text', 'ground_truth']
    dataset = assemble_pandas(data, columns)
    return dataset


def create_combined_set(users):
    data = []
    for user in users:
        posts = read_tweet_text(user[0] + ".xml")
        data.append([user[0], posts, user[1]])
    return data
