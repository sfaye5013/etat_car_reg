"""
Application Streamlit — Prédiction du prix de vente d'une voiture d'occasion.

Lancement en local :  streamlit run app.py
"""

import numpy as np
import pandas as pd
import joblib as jb
import streamlit as st


# Configuration de la page
st.set_page_config(
    page_title="Prédiction du prix de vente d'une voiture",
    page_icon="🚗",
    layout="centered",
)

DESCRIPTION = (
    "Ce modèle de machine learning permet de prédire le prix de vente d'une "
    "voiture d'occasion à partir du kilométrage, du prix neuf, du type de "
    "carburant, du type de vendeur, de la transmission et de l'âge du véhicule."
)


# Chargement des artefacts (mis en cache : chargés une seule fois)

MODELS_DIR = "models"


@st.cache_resource
def load_artifacts():
    encoders = jb.load(f"{MODELS_DIR}/encoders.joblib")        # encodeurs (carburant, vendeur, transmission)
    uniques = jb.load(f"{MODELS_DIR}/uniques.joblib")          # valeurs uniques
    scaler = jb.load(f"{MODELS_DIR}/scaler.joblib")            # normaliseur
    model = jb.load(f"{MODELS_DIR}/model.joblib")              # modèle
    feature_cols = jb.load(f"{MODELS_DIR}/feature_cols.joblib")  # ordre des colonnes attendu
    return encoders, uniques, scaler, model, feature_cols


encoders, uniques, scaler, model, feature_cols = load_artifacts()


# Fonction de prédiction simple

def Pred_func(kms_driven, present_price, fuel_type, seller_type, transmission, age):
    # Encoder les variables catégorielles
    fuel_type = encoders[0].transform([fuel_type])[0]
    seller_type = encoders[1].transform([seller_type])[0]
    transmission = encoders[2].transform([transmission])[0]
    # Vecteur des valeurs dans l'ordre d'entraînement
    x_new = np.array([kms_driven, present_price, fuel_type, seller_type, transmission, age])
    x_new = x_new.reshape(1, -1)  # conversion en un tableau 2D
    # Normaliser les données
    x_new = scaler.transform(x_new)
    # Prédire
    y_pred = model.predict(x_new)
    return float(y_pred[0])


# Fonction de prédiction multiple

def Pred_func_csv(file):
    # Lire le fichier csv
    df = pd.read_csv(file)
    predictions = []
    # Boucle sur les lignes du dataframe
    for row in df.iloc[:, :].values:
        # prédiction simple
        y_pred = Pred_func(row[0], row[1], row[2], row[3], row[4], row[5])
        predictions.append(y_pred)
    df["prix_vente_predit"] = predictions
    return df


# Interface

st.title("🚗 Prédiction du prix de vente d'une voiture")

onglet1, onglet2 = st.tabs(["Prédiction simple", "Prédiction multiple"])

# ----------------------------- Onglet 1 -------------------------------
with onglet1:
    st.subheader("Prédire le prix de vente d'une voiture avec une entrée")
    st.write(DESCRIPTION)

    with st.form("formulaire_simple"):
        col1, col2 = st.columns(2)
        with col1:
            kms_driven = st.number_input("Kilométrage (Kms_Driven)", value=71822.0, step=1000.0, format="%.2f")
            present_price = st.number_input("Prix neuf (Present_Price, en lakhs)", value=15.38, step=0.5, format="%.2f")
            fuel_type = st.selectbox("Type de carburant", options=list(uniques[0]))
        with col2:
            seller_type = st.selectbox("Type de vendeur", options=list(uniques[1]))
            transmission = st.selectbox("Transmission", options=list(uniques[2]))
            age = st.number_input("Âge du véhicule (années)", value=8.0, step=1.0, format="%.2f")

        soumettre = st.form_submit_button("Prédire", type="primary")

    if soumettre:
        try:
            resultat = Pred_func(kms_driven, present_price, fuel_type, seller_type, transmission, age)
            st.success(f"**Prix de vente prédit :** {resultat:.2f} lakhs")
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")

# ----------------------------- Onglet 2 -------------------------------
with onglet2:
    st.subheader("Prédire le prix de vente de plusieurs voitures")
    st.write(DESCRIPTION)
    st.caption(
        "Le fichier CSV doit contenir, dans cet ordre, les colonnes : "
        "Kms_Driven, Present_Price, Fuel_Type, Seller_Type, Transmission, Age."
    )

    fichier = st.file_uploader("Importer un fichier CSV", type=["csv"])

    if fichier is not None:
        try:
            with st.spinner("Prédictions en cours…"):
                df_resultat = Pred_func_csv(fichier)

            st.success(f"{len(df_resultat)} prédiction(s) effectuée(s).")
            st.dataframe(df_resultat, use_container_width=True)

            st.download_button(
                label="⬇️ Télécharger le fichier CSV",
                data=df_resultat.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
                type="primary",
            )
        except Exception as e:
            st.error(f"Erreur lors du traitement du fichier : {e}")
