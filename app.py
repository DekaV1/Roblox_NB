import streamlit as st
import joblib
import re
import nltk

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ===============================
# FIX NLTK DOWNLOAD
# ===============================
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

# ===============================
# LOAD MODEL
# ===============================
model = joblib.load("bernoulli_nb.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# ===============================
# PREPROCESSING
# ===============================
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Cache stopwords to avoid repeated calls
@st.cache_data
def get_stopwords():
    return set(stopwords.words('indonesian'))

stop_words = get_stopwords()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Handle NLTK punkt tokenizer fallback
    try:
        tokens = word_tokenize(text)
    except LookupError:
        # Fallback to simple whitespace tokenizer if punkt not available
        tokens = text.split()
    
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words and len(w) > 1]
    return ' '.join(tokens)

label_map = {
    0: "Negatif 😡",
    1: "Netral 😐",
    2: "Positif 😊"
}

# ===============================
# UI (FRONT END)
# ===============================
st.title("📊 Analisis Sentimen Roblox Indonesia")
st.write("Model: **Bernoulli Naive Bayes + TF-IDF**")

text_input = st.text_area(
    "Masukkan komentar Platform X:",
    placeholder="Contoh: Game roblox makin seru setelah update terbaru",
    height=120
)

if st.button("Prediksi Sentimen", type="primary"):
    if text_input.strip() == "":
        st.warning("⚠️ Teks tidak boleh kosong")
    else:
        try:
            clean_text = preprocess(text_input)
            vector = tfidf.transform([clean_text])
            prediction = model.predict(vector)[0]
            
            # Display result with color coding
            if prediction == 0:  # Negatif
                st.error(f"Hasil Sentimen: **{label_map[prediction]}**")
            elif prediction == 1:  # Netral
                st.warning(f"Hasil Sentimen: **{label_map[prediction]}**")
            else:  # Positif
                st.success(f"Hasil Sentimen: **{label_map[prediction]}**")
            
            # Show processed text (optional)
            with st.expander("Lihat teks yang diproses"):
                st.write(f"**Teks asli:** {text_input}")
                st.write(f"**Teks diproses:** {clean_text}")
                
        except Exception as e:
            st.error(f"Terjadi kesalahan: {str(e)}")
            st.info("Silakan coba input teks yang berbeda atau refresh halaman")
