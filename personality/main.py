import csv

import pandas as pd
import tweepy

# authorization tokens
from tweepy import TweepError

consumer_key = 'v0xKMKsBMFN5h2WUmWTG1leh8'
consumer_secret = 'rsSy7BfKhXU61ktvbn7VF9SHbCcTNZJ65xcvYWcc8dLhzAEbuY'
access_key = '133859328-QITghlxAxmVaDJim41H7hxmDSzTUk2pusFVPc6sS'
access_secret = 'ZJUF0Enx27RYltuz2cB7ItFxhinBlZx38PinZqEvmae5T'


def get_user_ids(api):
    # read personality dataset
    data = pd.read_csv('personality-data.txt', sep="\t")

    # screen name of the account to be fetched
    user_ids = data['twitter_uid'].tolist()
    return user_ids


def create_separated_csv(filename, user_ids):
    csvfile = open(filename + "_separated.csv", 'w')
    c = csv.writer(csvfile)
    # write the header row for CSV file
    c.writerow(["twitter_uid",
                "statuses"])
    # add each member to the csv
    for id in user_ids:
        user_info = get_userinfo(name)
        c.writerow(user_info)
    # close and save the CSV
    csvfile.close()


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
