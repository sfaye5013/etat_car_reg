# 🚗 Prédiction du prix de vente d'une voiture

Application **Streamlit** qui prédit le prix de vente d'une voiture d'occasion
à partir de son kilométrage, de son prix neuf, de son type de carburant, de
son type de vendeur, de sa transmission et de son âge. Le modèle retenu est un
**Random Forest Regressor**, comparé à XGBoost dans le notebook d'exploration.

## Structure du projet

```
.
├── app.py                           # Application Streamlit (interface + inférence)
├── requirements.txt                 # Dépendances Python
├── data/
│   └── Car_data.csv                 # Jeu de données brut utilisé pour l'entraînement
├── models/
│   ├── model.joblib                 # Modèle entraîné (modèle utilisé par app.py)
│   ├── model_name.joblib            # Nom du modèle retenu (random_forest ou xgboost)
│   ├── scaler.joblib                # RobustScaler ajusté sur les variables
│   ├── encoders.joblib              # LabelEncoders (Fuel_Type, Seller_Type, Transmission)
│   ├── uniques.joblib                # Valeurs uniques des variables catégorielles (menus déroulants)
│   └── feature_cols.joblib          # Ordre des colonnes attendu par le modèle
└── Reg_Random_Forest_XGBoost.ipynb   # Exploration, préparation des données et entraînement des modèles
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

## Lancer l'application

```bash
streamlit run app.py
```

L'application propose deux modes :

- **Prédiction simple** : renseigner les caractéristiques d'une voiture via un
  formulaire et obtenir son prix de vente prédit.
- **Prédiction multiple** : importer un fichier CSV contenant, dans cet
  ordre, les colonnes `Kms_Driven, Present_Price, Fuel_Type, Seller_Type,
  Transmission, Age`, et télécharger le fichier avec la colonne
  `prix_vente_predit` ajoutée.

## Entraînement du modèle

Le notebook `Reg_Random_Forest_XGBoost.ipynb` contient le pipeline de
préparation des données :

1. Analyse exploratoire (distributions, corrélations, boxplots).
2. Encodage des variables catégorielles (`LabelEncoder`) et sauvegarde des
   encodeurs et des valeurs uniques.
3. Traitement des valeurs aberrantes par imputation KNN (`KNNImputer`).
4. Normalisation des variables numériques (`RobustScaler`).
5. Séparation des données en train / validation / test (80 % / 10 % / 10 %).
6. Recherche d'hyperparamètres (`GridSearchCV`) et entraînement de deux
   modèles : Random Forest et XGBoost.
7. Évaluation (R², MSE, MAE) et sélection du meilleur modèle, exporté avec
   `joblib` dans `models/`.

Le modèle actuellement déployé a été entraîné directement avec ce notebook :
Random Forest a été retenu avec un **R² ≈ 0,88** sur les données de
validation.

## Jeu de données

`data/Car_data.csv` contient 5861 lignes avec les colonnes suivantes :

| Colonne          | Description                                   |
|------------------|------------------------------------------------|
| `Kms_Driven`     | Kilométrage parcouru                          |
| `Present_Price`  | Prix neuf du véhicule (en lakhs)              |
| `Fuel_Type`      | Type de carburant (Petrol, Diesel, CNG)       |
| `Seller_Type`    | Type de vendeur (Dealer, Individual)          |
| `Transmission`   | Type de transmission (Manual, Automatic)      |
| `Age`            | Âge du véhicule (années)                      |
| `Selling_Price`  | Prix de vente du véhicule (variable cible)    |

## Déploiement

Cette application est conçue pour être déployée sur **Streamlit Community
Cloud** :

1. Pousser ce dépôt sur GitHub.
2. Aller sur [share.streamlit.io](https://share.streamlit.io), se connecter
   avec GitHub.
3. Cliquer sur **New app**, sélectionner le dépôt, la branche `main` et le
   fichier `app.py`.
4. Déployer : l'application sera accessible via une URL publique du type
   `https://<nom-app>.streamlit.app`.
