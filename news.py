import re
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests
from bs4 import BeautifulSoup

def sentiment_score(review):
    tokens = tokenizer.encode(review, return_tensors='pt', max_length=512, truncation=True)
    result = model(tokens)
    max_value, max_index = torch.max(result.logits, dim=1)
    return int(max_index) + 1
    # return int(torch.argmax(result.logits)) + 1

tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

url = 'https://www.yelp.com/biz/social-brew-cafe-pyrmont'
try:
    r = requests.get(url)
    r.raise_for_status()  # Raise an exception if the request fails
    soup = BeautifulSoup(r.text, 'html.parser')
    regex = re.compile('.*comment.*')
    results = soup.find_all('p', {'class': regex})
    reviews = [result.text for result in results]
except requests.exceptions.RequestException as e:
    print(f"Error fetching data from Yelp: {e}")
    reviews = []

if reviews:
    df = pd.DataFrame(np.array(reviews), columns=['review'])
    df['sentiment'] = df['review'].apply(lambda x: sentiment_score(x))
    print(df)
else:
    print("No reviews found on the webpage.")