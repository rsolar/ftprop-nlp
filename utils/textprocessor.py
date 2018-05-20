import re
from string import punctuation

from nltk.tokenize import casual_tokenize


def clean_tweet(tweet):
    tweet = re.sub(r"https?://\S+", "", tweet)
    toks = casual_tokenize(tweet, preserve_case=False, reduce_len=True, strip_handles=True)
    toks = [''.join(c for c in s if c not in punctuation) for s in toks]
    toks = [s for s in toks if s]
    return toks


def tokenize(text):
    return [tok for tok in clean_tweet(text)]


if __name__ == '__main__':
    print(clean_tweet("RT @ #happyfuncoding: this is a typical Twitter tweet :-)"))
    print(clean_tweet("HTML entities &amp; other Web oddities can be an &aacute;cute <em class='grumpy'>pain</em> >:(')"))
    print(clean_tweet("It's perhaps noteworthy that phone numbers like +1 (800) 123-4567, (800) 123-4567, and 123-4567 are treated as words despite their whitespace."))
