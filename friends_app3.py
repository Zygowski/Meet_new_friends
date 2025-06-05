import streamlit as st
import pandas as pd  # type: ignore
from pycaret.clustering import load_model, predict_model  # type: ignore
import plotly.express as px
import json 

MODEL_NAME = 'wytrenowany model'

DATA = 'welcome_survey_simple_v2.csv'

CLUSTER_NAMES_AND_DESCRIPTIONS = 'opis_clustrów.json'

@st.cache_data # informuje streamlita że ma zaczytać ten plik tylko raz anie za każdym razem
def get_cluster_names_and_descriptions():
    with open(CLUSTER_NAMES_AND_DESCRIPTIONS, "r", encoding='utf-8') as f:
        return json.loads(f.read())

@st.cache_data 
def get_model():
    return load_model(MODEL_NAME)

@st.cache_data
def get_all_participants():
    model = get_model()
    all_df = pd.read_csv(DATA, sep=';')
    df_with_clusters = predict_model(model, data=all_df)

    return df_with_clusters


with st.sidebar:
    st.header('Powiedź coś o sobie')
    st.markdown('Pomożemy ci znaleźć znajomych, którzy mają podobne zainteresowania.')
    age = st.selectbox(
        'Wiek',
        ['<18', '25-34', '45-54', '35-44', '18-24', '>=65', '55-64', 'unknown'],
    )
    edu_level = st.selectbox(
        'Wykształcenie',
        ['Podstawowe','Średnie','Wyższe',],
    )
    fav_animals = st.selectbox(
        "Ulubione zwierzęta",
          ['Brak ulubionych', 'Psy', 'Koty', 'Inne', 'Koty i Psy']
    )
    fav_place = st.selectbox(
        "Ulubione miejsce", 
        ['Nad wodą', 'W lesie', 'W górach', 'Inne']
    )
    gender = st.radio(
        "Płeć", 
        ['Mężczyzna', 'Kobieta']
    )

    person_df = pd.DataFrame([
        {
            'age': age,
            'edu_level': edu_level,
            'fav_animals': fav_animals,
            'fav_place': fav_place,
            'gender': gender
        }
    ])

model = get_model()
all_df = get_all_participants()
cluster_names_and_descriptions = get_cluster_names_and_descriptions()

predicted_cluster_id = predict_model(model, data=person_df)["Cluster"].values[0]  # wyświetla do jakiej grupy trafiam na podstawie moich odpowiedzi (person_df) i modelu (model)
predicted_cluster_data = cluster_names_and_descriptions[predicted_cluster_id]


st.header(f"Najbliżej Ci do grupy {predicted_cluster_data['name']}")
st.markdown(predicted_cluster_data['description'])

same_cluster_df = all_df[all_df["Cluster"] == predicted_cluster_id] # zaczytywanie wszystkich 'rekordów' z tego samego clusteru
st.metric("Liczba twoich znajomych", len(same_cluster_df))


st.header("Osoby z grupy")


fig = px.histogram(same_cluster_df.sort_values("age"), x="age")
fig.update_layout(
    title="Rozkład wieku w grupie",
    xaxis_title="Wiek",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)


fig = px.histogram(same_cluster_df, x="edu_level")
fig.update_layout(
    title="Rozkład wykształcenia w grupie",
    xaxis_title="Wykształcenie",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)


fig = px.histogram(same_cluster_df, x="fav_animals")
fig.update_layout(
    title="Rozkład ulubionych zwierząt w grupie",
    xaxis_title="Ulubione zwierzęta",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)


fig = px.histogram(same_cluster_df, x="fav_place")
fig.update_layout(
    title="Rozkład ulubionych miejsc w grupie",
    xaxis_title="Ulubione miejsce",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)


fig = px.histogram(same_cluster_df, x="gender")
fig.update_layout(
    title="Rozkład płci w grupie",
    xaxis_title="Płeć",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)
