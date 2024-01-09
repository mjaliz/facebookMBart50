from bs4 import BeautifulSoup
import requests
import pandas as pd
import time

with open('./../data/wordlist.txt') as file:
    words = file.read()

words = words.split('\n')
examples = []

try:
    for word in words:
        link = 'https://fastdic.com/word/{}'.format(word)
        soup = BeautifulSoup(requests.get(link).content)
        for i in soup.find_all(class_='result__sentences'):
            print(i.text.strip().split('\n'))
            examples.append(i.text.strip().split('\n'))
finally:
    df = pd.DataFrame(examples)
    df.to_csv(f'fastdict_examples_{time.time()}.csv')
