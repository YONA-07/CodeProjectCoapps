import streamlit as st
import pandas as pd


fake_news = pd.read_csv("Fake.csv")
real_news = pd.read_csv("True.csv")

real_news.head()
fake_news.head()

real_news['Isfake'] = 0
fake_news['Isfake'] = 1

df = pd.concat([real_news, fake_news])

# to remove the stop words
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

# to make it in string format
from nltk.stem.porter import PorterStemmer

port_stem = PorterStemmer()

import re,string
stop_words = set(stopwords.words('english'))
punctuation = list(string.punctuation)
stop_words.update(punctuation)

from bs4 import BeautifulSoup
def string_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def remove_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

def remove_URL(text):
    return re.sub(r'http\S+', '', text)

def remove_stopwords(text):
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stop_words:
            final_text.append(i.strip())
    return " ".join(final_text)

def clean_text_data(text):
    text = string_html(text)
    text = remove_square_brackets(text)
    text = remove_stopwords(text)
    text = remove_URL(text)
    return text

df['text'] = df['text'].apply(clean_text_data)

# divide it into dependent and independent
x = df['text']
y = df['Isfake']

# divide it into train and test
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
# feed them using tfid vetorizer
from sklearn.feature_extraction.text import TfidfVectorizer

vect = TfidfVectorizer()
# convert them to vector
x_train = vect.fit_transform(x_train)
x_test = vect.transform(x_test)

# Linear SVC
from sklearn.svm import LinearSVC

linearsvc = LinearSVC()
linearsvc.fit(x_train, y_train)

# to create a web page
st.title("Fake News Detection Using NLP")
input_text = st.text_area("Enter news Article")

# Train the vectorizer and model
def prediction(input_text):
    input_data = vect.transform([input_text])
    prediction = linearsvc.predict(input_data)
    return prediction[0]


if st.button("Predict"):
    if input_text:
        pred = prediction(input_text)
        if pred == 1:
            st.write("The News is Fake")
        else:
            st.write("The News is True")
    else:
        st.warning("Please enter a news article")
# if input_text:
#     pred = prediction(input_text)
#     if pred == 1:
#         st.write("The News is Fake")
#     else:
#         st.write("The News is True")
