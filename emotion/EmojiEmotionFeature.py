import re

from emotion.emoji_emotion import emoji_emotions


class EmojiEmotionFeature:

    def __init__(self):
        self.none = None

    def is_emoji_name_like(self, string):
        """
        Test whether a string is a valid emoji description
        :param string: the string to test
        :return: bool - is string valid
        """
        return re.search(r":[A-Za-z\-_]+:", string) is not None

    def exists_emoji_name(self, name):
        """
        Check whether the emoji name exists in the emoji_emotions dictionary
        :param name: the emoji name
        :return: bool
        """
        return name in emoji_emotions.keys()

    def emotions_of_emoji_named(self, name):
        """
        Get an array of the emotions [anger, fear, joy, sadness] associated with the emoji name.
        :param name: the emoji name
        :return: list - associated emotions
        """
        if name in emoji_emotions.keys():
            return emoji_emotions[name]
        else:
            return [0.0, 0.0, 0.0, 0.0]
