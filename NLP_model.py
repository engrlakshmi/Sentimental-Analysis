import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import style
style.use('ggplot')
import csv
df=pd.read_csv("Dataset-SA.csv",encoding="unicode_escape")
df.head()
#check the shape of the given dataset
print(f'The dataset has {df.shape[0]} number of rows and {df.shape[1]} number of columns.')
df['product_name'].nunique()
df.isnull().sum()
df.dropna(inplace=True, axis=0)
from bs4 import BeautifulSoup
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

# Download the stop words corpus
nltk.download('stopwords')

stops = set(stopwords.words('english')) #english stopwords

stemmer = SnowballStemmer('english') #SnowballStemmer

def review_to_words(raw_review):
    # 1. Delete HTML
    review_text = BeautifulSoup(raw_review, 'html.parser').get_text()
    # 2. Make a space
    letters_only = re.sub('[^a-zA-Z]', ' ', review_text)
    # 3. lower letters
    words = letters_only.lower().split()
    # 5. Stopwords
    meaningful_words = [w for w in words if not w in stops]
    # 6. Stemming
    stemming_words = [stemmer.stem(w) for w in meaningful_words]
    # 7. space join words
    return( ' '.join(stemming_words))
#apply review_to_words function on reviews
df['Review'] = df['Review'].apply(review_to_words)
from sklearn.preprocessing import LabelEncoder

# apply label encoding to the sentiment column
encoder = LabelEncoder()
df['sentiment_encoded'] = encoder.fit_transform(df['Sentiment'])
#import all the necessary packages

from sklearn.model_selection import train_test_split #import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer #import TfidfVectorizer
from sklearn.metrics import confusion_matrix #import confusion_matrix
from sklearn.naive_bayes import MultinomialNB #import MultinomialNB
from sklearn.ensemble import RandomForestClassifier  #import RandomForestClassifier
vectorizer = TfidfVectorizer()
reviews_corpus = vectorizer.fit_transform(df.Review)
reviews_corpus.shape
#dependent feature
sentiment = df['sentiment_encoded']
sentiment.shape
#split the data in train and test
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(reviews_corpus,sentiment,test_size=0.33,random_state=42)
print('Train data shape ',X_train.shape,Y_train.shape)
print('Test data shape ',X_test.shape,Y_test.shape)

clf = RandomForestClassifier().fit(X_train, Y_train)


pred = clf.predict(X_test)

print("Accuracy: %s" % str(clf.score(X_test, Y_test)))
print("Confusion Matrix")
print(confusion_matrix(pred, Y_test))

import pickle
with open('nlp_model.pickle', 'wb') as f:
    pickle.dump(clf, f)