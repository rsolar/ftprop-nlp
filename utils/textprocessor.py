import re
from string import punctuation

from nltk.tokenize import casual_tokenize
from nltk.tokenize.casual import URLS


def clean_tweet(tweet):
    tweet = re.sub(r"https?://\S+", "", tweet)
    # tweet = re.sub(URLS, "", tweet)
    toks = casual_tokenize(tweet, preserve_case=False, reduce_len=True, strip_handles=True)
    # toks = [''.join(c for c in s if c not in punctuation) for s in toks]
    toks = [s for s in toks if s]
    return toks


def tokenize(text):
    return [tok for tok in clean_tweet(text)]


if __name__ == '__main__':
    print("before:", "RT @ #happyfuncoding: this is a typical Twitter tweet :-)")
    print("after: ", clean_tweet("RT @ #happyfuncoding: this is a typical Twitter tweet :-)"))
    print("before:", "@someone did you check out this #superawesome!! it's very cool http://t.co/ydfY2")
    print("after: ", clean_tweet("@someone did you check out this #superawesome!! it's very cool http://t.co/ydfY2"))
    print("before:", "@ellelovexx haaaaa i want mac &amp; cheese toooooo!!!  hahahaha")
    print("after: ", clean_tweet("@ellelovexx haaaaa i want mac &amp; cheese toooooo!!!  hahahaha"))
