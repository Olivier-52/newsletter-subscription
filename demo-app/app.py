import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib

DATA_URL = "data/conversion_data_train.parquet"
MODEL_DF_URL = "data/models_summary.parquet"
MODEL_URL = "model/CatBoostClassifier.joblib"

### Config
st.set_page_config(
    page_title="Souscription à la newsletter",
    page_icon="📰",
    layout="wide"
)

### Data
@st.cache_data
def load_data():
    data = pd.read_parquet(DATA_URL, engine='pyarrow')
    data = data[data['age'] < 65] # Remove ouliers
    model_df = pd.read_parquet(MODEL_DF_URL, engine='pyarrow')

    return data, model_df

data, model_df = load_data()

### Load model
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_URL)
    return model

model = load_model()

### Streamlit pages

def dataset_page():
    st.title("Le jeu de données")
    st.write(f"Le jeu de données comporte **{data.shape[0]} lignes** et **{data.shape[1]} colonnes**.")
    
    meta_data = pd.DataFrame({
        "Colonnes": [
            "country",
            "age",
            "news_user",
            "source",
            "total_pages_visited",
            "converted"
            ],

        "Déscription": [
            "Pays de l'utilisateur",
            "Age de l'utilisateur",
            "L'utilisateur est-il un nouveau visiteur ?",
            "Origine de l'utilisateur",
            "Nombre de pages visitées",
            "L'utilisateur s'est-il abonné à la newsletter ?"
            ]
        })

    with st.expander("Afficher la déscription des colonnes"):
        st.dataframe(meta_data, hide_index=True, width='stretch')

    with st.expander("Afficher un apercu du jeu de données"):
        st.dataframe(data.head(), hide_index=True, width='stretch')

    with st.expander("Afficher les types de données"):
        data_types = data.dtypes
        st.dataframe(pd.DataFrame(data_types).T, hide_index=True, width='stretch')

    with st.expander("Afficher les statistiques descriptives"):
        st.write(data.describe())
        st.caption("Les utilisateurs avec un age strictement supérieur à 65 ont été supprimés (suppression des utilisateurs de 111 et 123 ans)")
    
    with st.expander("Afficher les données manquantes"):
        null_table = data.isnull().sum()
        st.dataframe(pd.DataFrame(null_table).T, hide_index=True, width='stretch')

def conversion_rate_analysis():
    st.title("Distribution de la variable cible")
    converted_perc_df = (data.value_counts('converted')/data.shape[0]).round(2)
    converted_perc_df.index = ['Non inscrit','Inscription']
    converted_perc_df.rename('pourcentage', inplace=True)
    fig = px.pie(converted_perc_df, values='pourcentage', names=converted_perc_df.index, title='Taux de conversion')
    fig.update_traces(textposition='inside', textinfo='percent+label', title_text='')
    st.plotly_chart(fig, width='stretch')

    st.badge("**Le taux de conversion est généralement très faible**", icon="📉", color="blue")

def country_analysis():
    st.title("Taux de conversion par pays")
    converted_perc_df = data.groupby('country')['converted'].mean().sort_values(ascending=False).round(2)
    converted_perc_df.index = converted_perc_df.index.map({'Germany': 'Allemagne', 
                                                            'China': 'Chine',
                                                            'UK': 'Royaume-Uni',
                                                            'US': 'États-Unis'
                                                            })
    fig = px.bar(x=converted_perc_df.index, y=converted_perc_df.values, color=converted_perc_df.index)
    fig.update_layout(yaxis_title='Taux de conversion', xaxis_title='Pays', title='Taux de conversion par pays')
    st.plotly_chart(fig, width='stretch')

    st.badge("**Les utilisateurs allemands sont plus susceptibles de s'inscrire au newsletter**", icon="🥨", color="blue")

def age_visits_analysis():
    st.title("Taux de conversion par age")
    age_converted = data.groupby('age')['converted'].mean().sort_values(ascending=False)
    fig = px.bar(x=age_converted.index, y=age_converted.values)
    fig.update_layout(yaxis_title='taux de conversion', xaxis_title='Age', title='Taux de conversion par age')
    st.plotly_chart(fig, width='stretch')

    st.badge("**Les jeuns utilisateurs sont plus susceptibles de s'inscrire au newsletter**", icon="👶", color="blue")

def visit_analysis():
    st.title("Taux de conversion par nombre de pages visitées")

    page_visited_converted = data.groupby('total_pages_visited')['converted'].mean().sort_values(ascending=False)
    fig = px.bar(x=page_visited_converted.index, y=page_visited_converted.values)
    fig.update_layout(yaxis_title='Taux de conversion', xaxis_title='Pages visitées', title='Taux de conversion par pages visitées')
    st.plotly_chart(fig, width='stretch')

    st.badge("**Les utilisateurs avec un fort engagement sont plus susceptibles de s'inscrire à la newsletter**", icon="🌐", color="blue")

def model_comparison():
    st.title("Comparaison des modèles")

    models_scores= model_df.groupby('model')[['f1', 'precision', 'recall']].max().round(2).reset_index()
    models_scores = models_scores.sort_values(['f1', 'recall', 'precision'], ascending=False)
    fig = px.bar(models_scores, x='model', y=['f1', 'precision', 'recall'], barmode='group')
    fig.update_layout(yaxis_title='Scores max', xaxis_title='Modèles', yaxis_range=[0.65, 0.88], title='Comparaison des modèles')
    st.plotly_chart(fig, width='stretch')

def optimizer_comparison():
    st.title("Comparaison des optimisateurs hyperparamétriques")

    optimizers_scores= model_df.groupby('optimizer')[['f1', 'precision', 'recall']].max().round(2).reset_index()
    optimizers_scores = optimizers_scores.sort_values(['f1', 'precision', 'recall'], ascending=False)
    fig = px.bar(optimizers_scores, x='optimizer', y=['f1', 'precision', 'recall'], barmode='group')
    fig.update_layout(yaxis_title='Scores max', xaxis_title='Optimisateurs hyperparamétriques', yaxis_range=[0.65, 0.88], title='Comparaison des optimisateurs hyperparamétriques')
    st.plotly_chart(fig, width='stretch')
    
def conversion_prediction_app():
    st.header("Predire la suscription d'un utilisateur à la newsletter")

## Inputs

    # Translate countries for pred_country
    mapping = {'Germany': 'Allemagne', 'China': 'Chine', 'UK': 'Royaume-Uni', 'US': 'États-Unis'}
    reverse_mapping = {v: k for k, v in mapping.items()}
    translated_countries = [mapping.get(country, country) for country in data["country"].unique()]
    selected_display = st.selectbox("Sélectionner un pays", translated_countries)

    # Get user inputs
    pred_country = reverse_mapping.get(selected_display, selected_display)
    pred_age = st.number_input("Selectionner un age", value=30, min_value=17, max_value=62, step=1)
    pred_visit = st.number_input("Selectionnner un nombre de pages visitees", value=5, min_value=1, step=1)
    pred_new = st.checkbox("Est-ce que cette personne est un nouvel utilisateur ?")

## Prediction
    if st.button("Valider"):
        data_to_predict = pd.DataFrame({
            "total_pages_visited": [pred_visit],
            "country": [pred_country],
            "new_user": [1 if pred_new else 0],
            "age": [pred_age]
            })
        
        with st.spinner("Predicting..."):
            prediction = model.predict(data_to_predict)
            if prediction == 1:
                st.success("Il est probable que cette personne **s'abonne à la newsletter**.")
            else:
                st.error("Il est probable que cette personne **ne s'abonne pas à la newsletter**.")

    ### Pages layout

pages = {
    "Application de démonstration": [
    st.Page(conversion_prediction_app, title="Prédictions", icon="📑"),
    ],
    "Informations sur les données": [
    st.Page(dataset_page, title="Le jeu de données", icon="📜"),
    st.Page(conversion_rate_analysis, title="Conversion", icon="💌"),
    st.Page(country_analysis, title="Pays", icon="🗺️"),
    st.Page(age_visits_analysis, title="Age", icon="👤"),
    st.Page(visit_analysis, title="Visites", icon="📈")
    ],
    "Informations sur les modèles": [
    st.Page(model_comparison, title="Comparaison des modèles", icon="⚙️"),
    st.Page(optimizer_comparison, title="Optimisateurs hyperparamétriques", icon="🦾"),
    ]
    }

pg = st.navigation(pages)

pg.run()