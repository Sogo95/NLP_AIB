import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from bertopic import BERTopic
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Charger les données à partir du fichier CSV
@st.cache_data
def load_data():
    try:
        data = pd.read_csv(r"F:\Master_en_IA\NLP\articles.csv")
        return data
    except FileNotFoundError:
        st.error("Le fichier CSV n'a pas été trouvé. Assurez-vous que le chemin d'accès est correct.")

# Modélisation de sujets avec LDA
@st.cache_data
def lda_modeling(data, num_topics):
    # Vectorisation TF-IDF
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(data)

    # Modèle LDA
    lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda_matrix = lda_model.fit_transform(tfidf_matrix)

    # Mots-clés de chaque sujet
    feature_names = tfidf_vectorizer.get_feature_names_out()
    topics = {}
    for topic_idx, topic in enumerate(lda_model.components_):
        topics[topic_idx+1] = [feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]
    return topics, lda_model

# Modélisation de sujets avec NMF
@st.cache_data
def nmf_modeling(data, num_topics):
    # Vectorisation TF-IDF
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(data)

    # Modèle NMF
    nmf_model = NMF(n_components=num_topics, random_state=42)
    nmf_matrix = nmf_model.fit_transform(tfidf_matrix)

    # Mots-clés de chaque sujet
    feature_names = tfidf_vectorizer.get_feature_names_out()
    topics = {}
    for topic_idx, topic in enumerate(nmf_model.components_):
        topics[topic_idx+1] = [feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]
    return topics, nmf_model

# Modélisation de sujets avec BERTopic
@st.cache_data
def bertopic_modeling(data, num_topics):
    # Modèle BERTopic
    topic_model = BERTopic(language="english")
    topics, _ = topic_model.fit_transform(data)

    # Mots-clés de chaque sujet
    topics_keywords = topic_model.get_topics()
    return topics_keywords, topic_model

# Interface Streamlit
st.title("Modélisation de sujets extrait sur le site de l'Agence d'Information du Burkina (AIB)")

# Sélection du type de modélisation et du nombre de sujets
model_type = st.selectbox("Choisissez le type de modélisation :", ["LDA", "NMF", "BERTopic"])
num_topics = st.slider("Nombre de sujets :", min_value=2, max_value=10, value=5)

# Chargement des données
data = load_data()

# Affichage des résultats
if st.button("Lancer la modélisation"):
    st.write(f"Modélisation de sujets avec {model_type} pour {num_topics} sujets :")
    if model_type == "LDA":
        topics, model = lda_modeling(data['content'], num_topics)
    elif model_type == "NMF":
        topics, model = nmf_modeling(data['content'], num_topics)
    elif model_type == "BERTopic":
        topics, model = bertopic_modeling(data['content'], num_topics)
    
    for topic, keywords in topics.items():
        st.write(f"Sujet {topic} : {keywords}")

    # Visualisation des sujets avec un nuage de mots
    fig, axs = plt.subplots(nrows=num_topics, figsize=(10, 6*num_topics))
    for i, (topic, keywords) in enumerate(topics.items()):
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(keywords))
        axs[i].imshow(wordcloud, interpolation='bilinear')
        axs[i].set_title(f"Sujet {topic}")
        axs[i].axis('off')
    st.pyplot(fig)
