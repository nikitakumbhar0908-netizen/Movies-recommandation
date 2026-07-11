import streamlit as st
import pandas as pd
import pickle
import requests
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Load Files
# -----------------------------
movies = pd.read_csv("clean_movies.csv")


if not os.path.exists("similarity.pkl"):
    if st.button("Genrate similarities"):
        cv = CountVectorizer(max_features=5000, stop_words='english')
        vectors = cv.fit_transform(movies['tags']).toarray()
        similarity = cosine_similarity(vectors)
        pickle.dump(movies, open("movie_list.pkl", "wb"))


movie_list = movies["title"].values
st.markdown("""
<style>

html, body, [class*="css"]{
font-fmily:'Poppins',sans-serif;
}

.stApp{
background:linear-gradient(#4682B);
}

h1{
text-align:center;
color:black;
font-size:25px;
}

.small-font{
font-size:20px;
color:#dbeafe;
text-align:center;
}

div[data-baseweb="select"]{
background:white;
border-radius:12px;
}

.stButton>button{
width:100%;
height:55px;
background:#5C4B8A;
color:white;
font-size:20px;
font-black:bold;
border-radius:12px;
border:none;
}

.stButton>button:hover{
background:#dc2626;
}

</style>
""",unsafe_allow_html=True)
st.sidebar.title("🎬 Movie Recommendation")

st.sidebar.markdown("---")

st.sidebar.subheader("👩‍💻 Developer")

st.sidebar.write("**Nikita Kumbhar**")

st.sidebar.write("🎓 BCA Student")

st.sidebar.write("💻 Data Science & Machine Learning")

st.sidebar.markdown("---")

st.sidebar.subheader("📊 Project")

st.sidebar.success(f"Total Movies : {len(movie_list)}")

st.sidebar.info("Content Based Recommendation System")

st.sidebar.markdown("---")

st.sidebar.caption("Made with ❤️ using Python & Streamlit")
st.markdown("<h1>🎬 Movie Recommendation System</h1>",unsafe_allow_html=True)

st.markdown("<p class='big-font'>Find movies similar to your favourite movie 🍿</p>",unsafe_allow_html=True)

st.write("")
selected_movie = st.selectbox(
"🎥 Select Movie",
movie_list
)
# ==========================
# STEP 6 : Recommendation
# ==========================

def recommend(movie):

    movie_index = movies[movies["title"] == movie].index[0]

    distances = similarity[movie_index]

    recommended_movies = sorted(
        list(enumerate(distances)),
        key=lambda x: x[1],
        reverse=True
    )[1:6]

    movie_names = []

    for i in recommended_movies:
        movie_names.append(movies.iloc[i[0]].title)

    return movie_names


if st.button("🎬 Recommend Movies"):

    recommendations = recommend(selected_movie)

    st.markdown("## 🌟 Recommended Movies")

    for i, movie in enumerate(recommendations, start=1):
        similarity = pickle.load(open("similarity.pkl", "rb"))


        st.markdown(f"""
        <div style="
        background:linear-gradient(#2C3E5);
        padding:4px;
        margin-bottom:5px;
        border-radius:5px;
        border-left:6px solid #8B5CF6;
        box-shadow:0px 4px 3px rgba(0,0,0,0.4);
        ">
        <h3 style="color:#800020;margin:0;">
        🎬 {i}. {movie}
        </h3>
        </div>
        """, unsafe_allow_html=True)

    st.balloons()
    st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 🎯 Highlights
    ✅ AI Powered Recommendation  
    ✅ Content Based Filtering  
    ✅ Cosine Similarity Algorithm  
    ✅ 3965+ Movies Dataset  
    ✅ Fast & Accurate Results
    """)

with col2:
    st.markdown("""
    ### 👩‍💻 Developer
    **Nikita Kumbhar**

    🎓 BCA Student

    💻 Data Science & Machine Learning Enthusiast

    🐍 Python | NLP | Streamlit
    """)

st.markdown("---")

st.success("🎉 Thank you for using Movie Recommendation System!")

st.caption("© 2026 Movie Recommendation System | Developed by Nikita Kumbhar ❤️")