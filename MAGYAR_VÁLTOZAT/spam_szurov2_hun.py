import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score
import streamlit as st
from nltk.corpus import stopwords
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

# Magyar stop szavak letöltése
nltk.download('stopwords')
hungarian_stop_words = stopwords.words('hungarian')

# Adatok beolvasása
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'emails.xlsx')

data = pd.read_excel(file_path)
data['Label'] = data['Label'].map({'spam': 1, 'ham': 0})

# Szövegelőfeldolgozás függvény magyar nyelvre optimalizálva
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)  
    text = re.sub(r'\s+', ' ', text)  
    text = text.strip() 
    return text

data['Text'] = data['Text'].apply(preprocess_text)

X = data['Text']
y = data['Label']

# TF-IDF vektorizáció magyar stop szavakkal
vectorizer = TfidfVectorizer(stop_words=hungarian_stop_words)
X = vectorizer.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

# Modell pontossági mutatók
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# Streamlit felület
st.set_page_config(page_title="Spam-szűrő_HUN", layout="wide", initial_sidebar_state="expanded")

st.title("A Spam-szűrő alkalmazás tanítása visszaigazolással")
st.write(f"Pontosság: {accuracy:.2f}, Precizitás: {precision:.2f}, Visszahívás: {recall:.2f}")

# Kezdeti állapot beállítása a Streamlit session-ben
if 'predicted_label' not in st.session_state:
    st.session_state.predicted_label = None

user_input = st.text_input("Írj be egy e-mail szövegét, hogy megtudhasd, spam-e vagy sem!")

# Modell dönt
if st.button("Küldés") and user_input:
    processed_input = preprocess_text(user_input)
    input_vector = vectorizer.transform([processed_input])
    st.session_state.predicted_label = "spam" if model.predict(input_vector)[0] == 1 else "ham"
    st.write(f"A modell döntése: {st.session_state.predicted_label}")

# Visszajelzés gombok
if st.session_state.predicted_label is not None:
    if st.button("A döntés hibás"):
        correct_label = 'ham' if st.session_state.predicted_label == 'spam' else 'spam'
        st.write("Köszönjük a visszajelzést! A válasz módosítva.")
        new_feedback = pd.DataFrame([[user_input, correct_label]], columns=["Text", "Label"])
        try:
            existing_data = pd.read_excel(file_path)
            updated_data = pd.concat([existing_data, new_feedback], ignore_index=True)
        except FileNotFoundError:
            updated_data = new_feedback
        updated_data.to_excel(file_path, index=False)

    if st.button("A döntés helyes"):
        st.write("Köszönjük a visszajelzést!")
        new_feedback = pd.DataFrame([[user_input, st.session_state.predicted_label]], columns=["Text", "Label"])
        try:
            existing_data = pd.read_excel(file_path)
            updated_data = pd.concat([existing_data, new_feedback], ignore_index=True)
        except FileNotFoundError:
            updated_data = new_feedback
        updated_data.to_excel(file_path, index=False)

    # Újra gomb az új input megadásához
    if st.button("Újra"):
        st.session_state.clear()
        st.write("Új e-mailt adhatsz meg.")

# Naive Bayes eloszlás grafikon
def plot_naive_bayes_distribution():
    spam_words = [word for word in data[data['Label'] == 1]['Text'].str.cat(sep=' ').split() if word not in hungarian_stop_words]
    ham_words = [word for word in data[data['Label'] == 0]['Text'].str.cat(sep=' ').split() if word not in hungarian_stop_words]

    spam_word_freq = pd.Series(spam_words).value_counts().head(10)
    ham_word_freq = pd.Series(ham_words).value_counts().head(10)

    df = pd.DataFrame({'Spam': spam_word_freq, 'Ham': ham_word_freq})
    ax = df.plot(kind='bar', figsize=(5, 2))
    ax.set_title('A leggyakoribb szavak eloszlása spam és ham e-mailekben', fontsize=12)
    ax.set_xlabel('Szavak', fontsize=10)
    ax.set_ylabel('Frekvencia', fontsize=10)
    plt.xticks(rotation=45, fontsize=5)
    plt.legend(["Spam", "Ham"])
    st.pyplot(plt)

plot_naive_bayes_distribution()

# Oldalsáv információkkal
st.sidebar.title("Információ")
st.sidebar.markdown("Ez az alkalmazás spam e-mailek azonosítására szolgál a Naive Bayes algoritmus segítségével. Ez a verzió felismeri az eddigi e-maileket, és tanítani tudja magát a visszajelzések alapján. A modell e verziója <span style='color:red; font-weight:bold;'>magyar</span> szövegekre van tanítva.", unsafe_allow_html=True)

st.sidebar.title("Példa e-mailek")
st.sidebar.write("Spam példa: 'Gratulálunk! Ön nyert egy ajándékkártyát!'")
st.sidebar.write("Ham példa: 'Holnap találkozunk 10 órakor a megbeszélésen.'")
