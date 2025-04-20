import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    words = text.lower().split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

@st.cache_data
def load_model():
    url = 'https://raw.githubusercontent.com/datasets/sentiment-analysis-imdb/master/data/imdb-reviews.csv'
    data = pd.read_csv(url)
    data['review'] = data['review'].apply(clean_text)

    X = data['review']
    y = data['sentiment']

    vectorizer = CountVectorizer()
    X_vec = vectorizer.fit_transform(X)

    model = MultinomialNB()
    model.fit(X_vec, y)

    return model, vectorizer

model, vectorizer = load_model()

st.title("Simple Sentiment Analyzer")
user_input = st.text_area("Type your sentence here:")

if st.button("Analyze"):
    cleaned = clean_text(user_input)
    vect_input = vectorizer.transform([cleaned])
    prediction = model.predict(vect_input)[0]
    st.write(f"**Sentiment:** {prediction.capitalize()}")
