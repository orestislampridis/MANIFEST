# noinspection PyUnresolvedReferences
from xml.dom.minidom import parse


def get_documents(x):
    DOMTree = parse("data/%s" % x)
    documents = DOMTree.getElementsByTagName("document")
    return documents


def read_tweet_text(x):
    s = """"""
    documents = get_documents(x)
    for document in documents:
        s += document.firstChild.wholeText
    return s


def read_all(data, x, y):
    documents = get_documents(x)
    for document in documents:
        name = x.split('.')[0]
        data.append([name, document.firstChild.wholeText, y])


def read_single(x):
    data = []
    documents = get_documents(x)
    for document in documents:
        data.append(document.firstChild.wholeText)
    return data
