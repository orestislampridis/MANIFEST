import csv

import pandas as pd
import tweepy

import config as cfg

# Authorization tokens
consumer_key = cfg.consumer_key
consumer_secret = cfg.consumer_secret
access_key = cfg.access_key
access_secret = cfg.access_secret

# Dictionary of list of keywords to search for each education level
basic_kws = {"basic": ["high school student", "college dropout", "university dropout", "quit college"]}
higher_kws = {"higher": ["undergraduate student", "bachelor student", "student at university", "college", "university",
                         "community college", "fraternity", "sorority", "alumni"]}
highest_kws = {"highest": ["graduate student", "post graduate student", "phd student", "phd candidate", "postdoc",
                           "Doctor of Philosophy", "professor at", "research fellow", "research assistant"]}


def clean_duplicates(filename):
    basic = pd.read_csv(filename + ".csv").drop_duplicates()
    basic.to_csv(filename + "_no_duplicates.csv", index=False)


def create_csv(api, keywords):
    current_kw = (list(keywords.keys())[0])

    if current_kw != "basic":
        page_max = 6
    else:
        page_max = 16

    filename = current_kw
    csvfile = open(filename + ".csv", 'w', newline='', encoding="utf-8")
    c = csv.writer(csvfile)
    # write the header row for CSV file
    c.writerow(["twitter_uid",
                "screen_name",
                "description",
                "keyword"])

    for query in list(keywords.values())[0]:
        print(query)
        for page in range(1, page_max):
            print(page)
            users = api.search_users(q=query, page=page, count=20, include_entities=False)

            for user in users:
                print("user id: ", user.id)
                print("user screen name: ", user.screen_name)
                print("user description: ", user.description)

                c.writerow([user.id, user.screen_name, user.description, query])

    # close and save the CSV
    csvfile.close()
    # helper function to remove duplicates from above procedure
    clean_duplicates(filename)


if __name__ == "__main__":
    # complete authorization and initialize API endpoint
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)

    # give as argument one of three list of keywords
    create_csv(api, basic_kws)
    print("\njob's done!")
