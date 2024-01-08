from bs4 import BeautifulSoup
import requests
import pandas as pd

with open('./../data/words_alpha.txt') as file:
    words = file.read()

words = words.split('\n')
examples = []

for word in words:
    link = 'https://fastdic.com/word/{}'.format(word)
    soup = BeautifulSoup(requests.get(link).content)
    for i in soup.find_all(class_='result__sentences'):
        print(i.text.strip().split('\n'))
        examples.append(i.text.strip().split('\n'))

df = pd.DataFrame(examples)

df.to_csv('fastdict_examples.csv')
