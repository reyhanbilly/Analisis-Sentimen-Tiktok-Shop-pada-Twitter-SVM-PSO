import streamlit as st
from streamlit_option_menu import option_menu
from apps import training, prediction

st.set_page_config(page_title="Analisis Sentimen Tiktok Shop", layout="wide")
# Custom CSS for styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #2b2d42; /* Ganti dengan warna latar belakang yang diinginkan */
    }
    .stButton>button {
        background-color: #007BFF; /* Ganti dengan warna tombol yang diinginkan */
        color: white; /* Ganti dengan warna teks tombol yang diinginkan */
    }
    .stButton>button:hover {
        background-color: #0056b3; /* Ganti dengan warna tombol saat hover */
    }
    .stSidebar {
        background-color: #2b2d42; /* Ganti dengan warna latar belakang yang diinginkan */
    }
    </style>
    """,
    unsafe_allow_html=True
)
apps = [
    {"func": training.app, "title": "Training", "icon": "graph-up"},
    {"func": prediction.app, "title": "Prediksi", "icon": "card-text"},
]

titles = [app["title"] for app in apps]
titles_lower = [title.lower() for title in titles]
icons = [app["icon"] for app in apps]

with st.sidebar:
    selected = option_menu(
        "Menu Aplikasi",
        options=titles,
        icons=icons,
        menu_icon="cast",
        default_index=0,
    )

for app in apps:
    if app["title"] == selected:
        app["func"]()
        break
