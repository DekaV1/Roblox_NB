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
            # TAMPILKAN PROSES PREPROCESSING
            st.markdown("---")
            st.subheader("📝 Tahapan Preprocessing")
            
            # Step 1: Original text
            st.write("**1. Teks Asli:**")
            st.code(text_input, language="text")
            
            # Step 2: Lowercase
            text_lower = text_input.lower()
            st.write("**2. Lowercase:**")
            st.code(text_lower, language="text")
            
            # Step 3: Remove special characters
            text_clean = re.sub(r'[^a-z\s]', '', text_lower)
            st.write("**3. Hapus karakter khusus:**")
            st.code(text_clean, language="text")
            
            # Step 4: Tokenization
            try:
                tokens = word_tokenize(text_clean)
                tokenizer_name = "NLTK word_tokenize"
            except LookupError:
                tokens = text_clean.split()
                tokenizer_name = "Simple split()"
            
            st.write(f"**4. Tokenization ({tokenizer_name}):**")
            st.write(tokens)
            
            # Step 5: Stopwords removal
            stopwords_removed = [w for w in tokens if w in stop_words]
            filtered_tokens = [w for w in tokens if w not in stop_words and len(w) > 1]
            
            st.write("**5. Stopwords yang dihapus:**")
            if stopwords_removed:
                st.write(stopwords_removed)
            else:
                st.write("Tidak ada stopwords")
            
            st.write("**6. Tokens setelah filter:**")
            st.write(filtered_tokens)
            
            # Step 6: Stemming
            stemmed_tokens = []
            for w in filtered_tokens:
                stemmed = stemmer.stem(w)
                stemmed_tokens.append(stemmed)
            
            st.write("**7. Stemming hasil:**")
            # Tampilkan mapping sebelum dan sesudah stemming
            for original, stemmed in zip(filtered_tokens, stemmed_tokens):
                if original != stemmed:
                    st.write(f"  {original} → {stemmed}")
                else:
                    st.write(f"  {original} (tidak berubah)")
            
            # Step 7: Final clean text
            clean_text = ' '.join(stemmed_tokens)
            st.write("**8. Teks akhir untuk prediksi:**")
            st.success(clean_text)
            
            # PREDIKSI
            st.markdown("---")
            st.subheader("🔮 Hasil Prediksi")
            
            # Transform ke TF-IDF
            vector = tfidf.transform([clean_text])
            
            # Get prediction
            prediction = model.predict(vector)[0]
            
            # Show prediction value
            st.write(f"**Nilai prediksi:** {prediction}")
            
            # Display result with color coding
            if prediction == 0:  # Negatif
                st.error(f"### Hasil Sentimen: {label_map[prediction]}")
            elif prediction == 1:  # Netral
                st.warning(f"### Hasil Sentimen: {label_map[prediction]}")
            else:  # Positif
                st.success(f"### Hasil Sentimen: {label_map[prediction]}")
            
            # Optional: Show probabilities if available
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(vector)[0]
                st.markdown("---")
                st.subheader("📊 Probabilitas")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Negatif", f"{probabilities[0]:.1%}")
                with col2:
                    st.metric("Netral", f"{probabilities[1]:.1%}")
                with col3:
                    st.metric("Positif", f"{probabilities[2]:.1%}")
                
            # Show processed text summary
            with st.expander("📋 Ringkasan Proses"):
                st.write(f"**Teks asli:** {text_input}")
                st.write(f"**Teks diproses:** {clean_text}")
                st.write(f"**Jumlah token awal:** {len(tokens)}")
                st.write(f"**Jumlah token setelah filter:** {len(filtered_tokens)}")
                st.write(f"**Stopwords dihapus:** {len(stopwords_removed)}")
                
        except Exception as e:
            st.error(f"Terjadi kesalahan: {str(e)}")
            st.info("Silakan coba input teks yang berbeda atau refresh halaman")
