import pickle

import pandas as pd
import requests
import streamlit as st


# Fetch movie poster from TMDB
def fetch_poster(movie_id):

    url = f"https://api.themoviedb.org/3/movie/{movie_id}"

    headers = {
        "accept": "application/json",
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiI1ZTJkZGM3NzVhNDcxZjk5NDk5OGU1NGYzOTQ2OWMzNCIsIm5iZiI6MTcwMDU1MDcwNS40MDcsInN1YiI6IjY1NWM1ODMxYjU0MDAyMTRkMDcxNzk4NCIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.ZwMKTuBI5wL0pHrYmT6N9GjfyZnYW18hEvXMG6jXYTU",
    }

    params = {"language": "en-US"}

    for attempt in range(3):

        try:

            response = requests.get(url, headers=headers, params=params, timeout=10)

            response.raise_for_status()

            data = response.json()

            if data.get("poster_path"):

                return "https://image.tmdb.org/t/p/w500" + data["poster_path"]

            return None

        except requests.exceptions.RequestException as e:

            print(f"TMDB Error for movie {movie_id} " f"(attempt {attempt + 1}/3): {e}")

            if attempt < 2:
                import time

                time.sleep(1)

    return None


# Recommendation function
def recommend(movie):

    movie_index = movies[movies["title"] == movie].index[0]

    distances = similarity[movie_index]

    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[
        1:6
    ]

    recommended_movies = []

    recommended_movies_posters = []

    for i in movies_list:

        movie_id = movies.iloc[i[0]].movie_id

        recommended_movies.append(movies.iloc[i[0]].title)

        recommended_movies_posters.append(fetch_poster(movie_id))

    return recommended_movies, recommended_movies_posters


# Load movie data
movies_dict = pickle.load(
    open("C:\\Users\\Pratham\\Movie recommend system\\movie_dict.pkl", "rb")
)

movies = pd.DataFrame(movies_dict)


# Load similarity data
similarity = pickle.load(
    open("C:\\Users\\Pratham\\Movie recommend system\\similarity.pkl", "rb")
)


# Streamlit UI
st.title("Movie Recommendation System")

st.write("Welcome to the Movie Recommendation System!")


# Movie selection
selected_movie_name = st.selectbox(
    "Select a movie to get recommendations:", movies["title"].values
)


# Recommendation button
if st.button("Get Recommendations"):

    st.write(f"Recommendations for {selected_movie_name}")

    recommendations = recommend(selected_movie_name)

    recommended_movies = recommendations[0]

    recommended_movies_posters = recommendations[1]

    # Create 5 columns
    cols = st.columns(5)

    # Display recommendations
    for i in range(5):

        with cols[i]:

            # Movie title
            st.write(recommended_movies[i])

            # Movie poster
            if recommended_movies_posters[i]:

                st.image(recommended_movies_posters[i], width="stretch")

            else:

                st.info("Poster unavailable")
