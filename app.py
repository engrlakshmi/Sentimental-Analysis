import numpy as np
from sklearn.model_selection import train_test_split #import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer #import TfidfVectorizer
from sklearn.metrics import confusion_matrix #import confusion_matrix
from sklearn.naive_bayes import MultinomialNB #import MultinomialNB
from sklearn.ensemble import RandomForestClassifier  #import RandomForestClassifier
import pickle
import pandas as pd
#from flasgger import Swagger
import streamlit as st

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
df=pd.read_csv('Dataset-SA.csv')
df.dropna(inplace=True, axis=0)

# Train a Random Forest classifier on the dataset
vectorizer=TfidfVectorizer()
X=vectorizer.fit_transform(df['Review'])
y=df['Sentiment']
clf=RandomForestClassifier()
clf.fit(X, y)

# Define the Streamlit app
st.title('Sentiment Analysis')
product_name = st.text_input('Enter the product name:')
price = st.number_input('Enter the price:')
review = st.text_input('Enter a review:')
if st.button('Predict'):
    # Convert the input to a vector using the same TfidfVectorizer
    x_input=vectorizer.transform([review])

    # Predict the sentiment using the trained Random Forest classifier
    prediction=clf.predict(x_input)[0]

    # Display the prediction
    if prediction == 'positive':
        st.write('Positive review')
    else:
        st.write('Negative review')
