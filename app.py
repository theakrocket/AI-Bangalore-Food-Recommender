import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Bangalore Food AI", layout="wide")

st.title("🍜 Bangalore AI Food Recommender")

st.write("Search food like: biryani indiranagar, cheap dosa, pizza")

@st.cache_data
def load_data():
    df = pd.read_csv("zomato.csv")
    df = df[['name','location','cuisines','rate','approx_cost(for two people)','rest_type']]
    df = df.dropna()

    # Larger dataset like the original version
    df = df.head(6000)

    df["combined"] = df["cuisines"] + " " + df["location"] + " " + df["rest_type"]

    return df

df = load_data()

vectorizer = TfidfVectorizer(stop_words="english")

vectors = vectorizer.fit_transform(df["combined"])

query = st.text_input("What food are you craving? Examples:cheap dosa,biryani indiranagar,pizza")

if query:

    user_vec = vectorizer.transform([query])

    similarity = cosine_similarity(user_vec, vectors)

    scores = similarity[0]

    top = scores.argsort()[-10:][::-1]

    st.subheader("Recommended Restaurants")

    for i in top:

        r = df.iloc[i]

        st.write("###", r["name"])
        st.write("📍 Location:", r["location"])
        st.write("🍽 Cuisine:", r["cuisines"])
        st.write("⭐ Rating:", r["rate"])
        st.write("💰 Cost for two:", r["approx_cost(for two people)"])
        st.write("---")

else:
    st.info("Type something like dosa, biryani or pizza")