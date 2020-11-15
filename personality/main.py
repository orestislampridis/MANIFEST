import csv

import pandas as pd
import tweepy
from tweepy import TweepError

import config as cfg

# authorization tokens
consumer_key = cfg.consumer_key
consumer_secret = cfg.consumer_secret
access_key = cfg.access_key
access_secret = cfg.access_secret


def get_user_ids(api):
    # read personality dataset
    data = pd.read_csv('personality-data.txt', sep="\t")

    # screen name of the account to be fetched
    user_ids = data['twitter_uid'].tolist()
    return user_ids


def create_combined_csv(api, filename, user_ids):
    csvfile = open(filename + "_combined.csv", 'w', encoding="utf-8")
    c = csv.writer(csvfile)
    # write the header row for CSV file
    c.writerow(["twitter_uid",
                "statuses",
                "tweets"])
    # add each member to the csv
    print(user_ids)

    for user_id in user_ids:
        # catch exception when user id doesn't exist
        try:
            # fetching the statuses
            statuses = api.user_timeline(user_id=user_id, count=100)
        except TweepError:
            continue

        s = """"""
        # printing the statuses
        for status in statuses:
            print(status.created_at)
            print(status.text, end="\n\n")
            s += status.text

        print(str(len(statuses)) + " number of statuses have been fetched.")
        tweets = len(statuses)

        c.writerow([user_id, s, tweets])

    # close and save the CSV
    csvfile.close()


def main():
    # authorization of consumer key and consumer secret
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    # set access to user's access key and access secret
    auth.set_access_token(access_key, access_secret)
    # calling the api
    api = tweepy.API(auth)

    # provide name for new CSV    
    filename = "new_personality"
    # create list of all members of the Twitter list
    user_ids = get_user_ids(api)
    # create new CSV and fill it
    create_combined_csv(api, filename, user_ids)


if __name__ == '__main__':
    main()
