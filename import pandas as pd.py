import pandas as pd
import os
import streamlit as st

@st.cache_data
def load_data():
    try:
        # Chemin relatif du fichier CSV
        csv_path = 'articles.csv'

        # Afficher le chemin absolu pour débogage
        abs_path = os.path.abspath(csv_path)
        st.write(f"Chemin absolu du fichier CSV : {abs_path}")

        # Imprimer le contenu du répertoire courant
        st.write("Contenu du répertoire courant :")
        st.write(os.listdir(os.getcwd()))

        # Vérifier si le fichier existe avant de le charger
        if os.path.exists(csv_path):
            data = pd.read_csv(csv_path)
            st.success("Fichier CSV chargé avec succès.")
            return data
        else:
            st.error("Le fichier CSV n'a pas été trouvé. Assurez-vous que le chemin d'accès est correct.")
            return None
    except FileNotFoundError:
        st.error("Le fichier CSV n'a pas été trouvé. Assurez-vous que le chemin d'accès est correct.")
        return None

# Charger les données
data = load_data()

if data is not None:
    st.write(data.head())
