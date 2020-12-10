"""
Get replies to tweets by utilizing Twitter API 2 and twint.

Twitter API doesn't support the returning of replies to tweet posts directly. Newly introduced in
Twitter API 2 there is an option to get the conversation_id of a tweet given its tweet id. The
conversation_id is the same for all tweets that belong in the same thread which means that the
original post along with all of its replies will share the same conversation id.

Thus to find all replies to an original post all that needs to be done is find all replies to
the user_id and then check if those replies conversation_id is equal to the original tweet
conversation_id.

Twitter API only allows the search of tweets directed to a particular user for only the last week.
To overcome this limitation twint is used which has no restriction on how far back the tweets can be scraped.
"""

import json
import logging
import time
from datetime import datetime, timedelta
from json.decoder import JSONDecodeError

import requests
import twint

import config as cfg
from connect_mongo import read_mongo


def get_conversation_id(tweet_id):
    token = cfg.bearer_token
    prefix = 'Bearer'
    headers = {"Authorization": '%s %s' % (prefix, token)}
    params = 'tweet.fields=author_id,conversation_id,created_at,in_reply_to_user_id,' \
             'referenced_tweets&expansions=author_id,in_reply_to_user_id,referenced_tweets.id&user.fields=name,username'

    retries = 1
    success = False
    while not success:
        try:
            response = requests.get("https://api.twitter.com/2/tweets/" + str(tweet_id),
                                    params=params, headers=headers)
            success = True
            try:
                coversation_id = response.json()['data']['conversation_id']
                return coversation_id
            except KeyError:
                logging.info("Tweet doesn't exist anymore")
                return None
            except JSONDecodeError:
                logging.info("json decode error")
                return None
            except:
                return None

        except Exception as e:
            wait = retries * 30
            print('Error! Waiting %s secs and re-trying...' % wait)
            time.sleep(wait)
            retries += 1


def get_replies(tweet_id, user_name, created_at, created_limit):
    conversation_id = get_conversation_id(tweet_id)

    if conversation_id is None:
        return [], []

    replies = twint.Config()
    replies.Since = created_at
    replies.Until = created_limit
    replies.Pandas = True
    replies.To = user_name
    replies.Hide_output = True
    twint.run.Search(replies)
    df = twint.storage.panda.Tweets_df
    if df.empty:
        return [], []
    else:
        tweet_id_replies_list = list()
        user_id_replies_list = list()

        for index, row in df.iterrows():
            if row['conversation_id'] == conversation_id and int(row['id']) != int(tweet_id):
                tweet_id_replies_list.append(row['id'])
                user_id_replies_list.append(row['user_id'])

        return tweet_id_replies_list, user_id_replies_list


if __name__ == "__main__":
    # Get our initial df with the columns that we need
    df = read_mongo(db='master_thesis_db', collection='us_elections',
                    query={'id': 1, 'created_at': 1, 'user.screen_name': 1, 'user.id': 1}).iloc[0:1000]
    df['created_at_limit'] = df['created_at'].apply(
        lambda x: datetime.strftime((datetime.strptime(x, '%a %b %d %H:%M:%S +0000 %Y') + timedelta(days=5)),
                                    '%Y-%m-%d'))
    df['created_at'] = df['created_at'].apply(
        lambda x: datetime.strftime(datetime.strptime(x, '%a %b %d %H:%M:%S +0000 %Y'), '%Y-%m-%d'))

    print(df['created_at_limit'])
    tweet_id_replies_dict = dict()
    user_id_replies_dict = dict()

    added = 0
    total = 0
    for index, row in df.iterrows():
        print(str(added) + "/" + str(total))
        tweet_id = row['id']
        user_id = row['user']['id']
        user_name = row['user']['screen_name']
        created_at = row['created_at']
        created_at_limit = row['created_at_limit']

        tweet_id_replies_list, user_id_replies_list = get_replies(tweet_id, user_name, created_at, created_at_limit)
        total += 1

        # Check if returned list is empty
        if not tweet_id_replies_list:
            continue
        else:
            tweet_id_replies_dict[tweet_id] = tweet_id_replies_list
            user_id_replies_dict[user_id] = user_id_replies_list

        added += 1

    print(tweet_id_replies_dict)
    print(user_id_replies_dict)

    dictionary = {
        'results': []
    }

    for (user_index, user_ids), (tweet_index, tweet_ids) in zip(user_id_replies_dict.items(),
                                                                tweet_id_replies_dict.items()):
        dictionary['results'].append({'tweet_id': tweet_index,
                                      'user_id': user_index,
                                      'reply_tweet_ids': tweet_ids,
                                      'reply_user_ids': user_ids})

    with open('replies_tweet_ids_2.json', 'w') as fp:
        json.dump(tweet_id_replies_dict, fp)

    with open('replies_user_ids_2.json', 'w') as fp:
        json.dump(user_id_replies_dict, fp)

    with open('super_dict_2.json', 'w') as fp:
        json.dump(dictionary, fp)
