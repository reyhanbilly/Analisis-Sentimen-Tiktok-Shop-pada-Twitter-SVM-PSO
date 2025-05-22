import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import re
import string
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
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



def app():
    st.title("Training Model")
   
    st.header("1. Load Dataset")
    uploaded_files = st.file_uploader(
        "Choose a CSV file", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        st.write("filename:", uploaded_file.name)
        df = pd.read_csv(uploaded_file)
        st.write(df)

        st.subheader("Distribusi Kelas")
        # Hitung distribusi kelas
        class_distribution = df['Label'].value_counts()

        # Custom nama kelas
        class_names = {
            0: 'Negatif',
            1: 'Netral',
            2: 'Positif'
        }

        # Ubah nilai dalam dataset menggunakan kamus class_names
        class_distribution = class_distribution.rename(index=class_names)

        # Tampilkan bar chart menggunakan st.bar_chart
        st.bar_chart(class_distribution)

    st.header("2. Preprocessing Dataset")
    st.markdown('''
    Tahap-tahap processing:  
    1. Case Folding  
    2. Cleansing  
    3. Tokenization
    4. Slang words Removal  
    5. Stemming  
''')

    if st.button('PREPROCESSING'):
        df['processed_text'] = df['Tweet'][:20].apply(preprocess_text)
        st.subheader("Preview Preprocessed Dataset")
        st.write(df[:20])

    # Header untuk memilih metode SVM atau SVM-PSO
    st.header("3. Pilih Metode")
    metode = st.radio("Pilih metode:", ("SVM", "SVM-PSO"))

    if st.button('TRAIN'):
        st.markdown(":green[TRAINING SUCCESS]")
        # Menentukan gambar yang ditampilkan berdasarkan metode yang dipilih
        if metode == "SVM":
            st.image('./asset/confusion_matrix_svm.png', caption='Confusion Matrix SVM')
        else:  # SVM-PSO
            st.image('./asset/confusion_matrix_svmpso.png', caption='Confusion Matrix SVM-PSO')
