import streamlit as st
import streamlit.components.v1 as components
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pickle
import re
import string
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import pandas as pd
from nltk.tokenize import word_tokenize


def preprocess_text(text):
    # Case folding
    text = text.lower()

    # Filtering
    #menghapus username twitter
    text = re.sub('@[^\s]+','',text)
    # remove mention, link, hashtag
    text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())
    # Memperbaiki pemisahan kata yang salah menggunakan regex
    text = re.sub(r'([a-zA-Z])\-([a-zA-Z])', r'\1 \2', text)
    # remove tab, new line, ans back slice
    text = text.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\',"")
    # remove non ASCII (emoticon, chinese word, .etc)
    text = text.encode('ascii', 'replace').decode('ascii')
    #remove punctuation
    text = re.sub(r'[{}]+'.format(re.escape(string.punctuation)), ' ', text)
    # remove white space
    text = text.strip()
    # Remove numbers
    text = re.sub(r'\d+', '', text)  # Menghapus semua angka
    #remove multiple white space
    text = re.sub('\s+',' ',text)
    # Mengganti "tiktokshop" menjadi "tiktok shop"
    text = re.sub(r'tiktokshop', 'tiktok shop', text, flags=re.IGNORECASE)
    # remove incomplete URL
    text = text.replace("http://", " ").replace("https://", " ")
    # Tokenizing
    words = word_tokenize(text)

    # Load slang words
    kamusSlang = eval(open("./data/slangwords.txt").read())
    pattern = re.compile(r'\b(' + '|'.join(kamusSlang.keys()) + r')\b')
    content = []
    for kata in words:
        filterSlang = pattern.sub(lambda x: kamusSlang[x.group()], kata)
        content.append(filterSlang.lower())
    
    # Stemming
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    stemmed_text = [stemmer.stem(word) for word in content]

    # Mengembalikan teks yang telah diproses
    return " ".join(stemmed_text)



def predict(input_text, model,tfidf_vectorizer):
    clean_input_text = preprocess_text(input_text)
    input_tfidf = tfidf_vectorizer.transform([clean_input_text]).toarray()

    # Lakukan prediksi
    predicted_label = model.predict(input_tfidf)

    if predicted_label == 1:
        kelas = 'Netral'
    elif predicted_label == 0:
        kelas = 'Negatif'
    elif predicted_label == 2:
        kelas = 'Positif'

    return kelas


def load_dataset():
    df = pd.read_csv('./data/data_ttshop_preprocessed.csv')
    df = df['preprocessed'].tolist()
    return df


def load_csv_data(file_path):
    df = pd.read_csv(file_path)
    texts = df['text'].tolist()
    return texts


def app():
    st.title("Prediction")
    st.subheader("User Input Text Analysis")

    with open('./model/model-svm-pso.pickle', 'rb') as file:
        vectorizer, clf = pickle.load(file)
        loaded_model = clf
    input_option = st.radio("Pilih sumber input:", ("Teks", "File CSV"))

    if input_option == "Teks":
        input_text = st.text_area('Teks', height=100)
        if st.button("PREDICT", type="primary"):
            prediction = predict(input_text, loaded_model,vectorizer)
            st.subheader(f"Hasil Prediksi: {prediction}")
    elif input_option == "File CSV":
        uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
        if uploaded_file is not None:
            csv_data = load_csv_data(uploaded_file)
            if st.button("PREDICT", type="primary"):
                predictions = [predict(text, loaded_model,vectorizer) for text in csv_data]
                table_data = {"Text": csv_data, "Prediction": predictions}
                st.table(pd.DataFrame(table_data))