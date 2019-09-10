from newspaper import Article, Config
import re
from collections import Counter

def body(url, proxies = None):
    cfg = None
    # If a proxy dict is passed, hand it to newspaper 
    if(proxies != None):
        cfg = Config()
        # needed for corporate network
        cfg.proxies = proxies
    article = Article(url, config = cfg)
    print("Created!")
    article.download()
    print("Downloaded!")
    article.parse()
    print("Parsed!")
    txt = article.text
    words = re.compile("[^A-Za-z]+").split(txt)
    print("Text splitted!")
    counters = Counter(words)
    print("Words counted!")
    return counters
    