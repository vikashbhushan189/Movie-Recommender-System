# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import os

# Import the MovieRecommender class from api.py
# Make sure api.py is in the same directory as streamlit_app.py
from api import MovieRecommender 

# --- Streamlit Application ---
st.set_page_config(layout="wide", page_title="Movie Recommender")

st.title("🎬 Advanced Movie Recommender")

# Cache the recommender system instance to prevent reloading on every rerun
# This will call MovieRecommender's __init__ (which tries to load from disk first)
# only once.
@st.cache_resource 
def load_recommender_system():
    # Pass the path to where your data and saved models are
    RECS_MODEL_PATH = 'saved_models_deploy/'
    DATA_PATH = 'data/'
    
    st.info("Loading recommender system... This may take a moment on first run.")
    try:
        # Try to load saved models directly first. If _try_load_from_saved_models in __init__ fails,
        # it will fall back to full initialization.
        recs_system = MovieRecommender(data_path=DATA_PATH)
        return recs_system
    except Exception as e:
        st.error(f"Failed to load or initialize the recommender system: {e}")
        st.warning("Ensure all data files (`data/`) and saved models (`saved_models_deploy/`) are correctly placed.")
        st.stop() # Stop the app if core component fails


# Load the recommender system (cached by Streamlit)
recommender = load_recommender_system()

# Ensure the recommender system is properly initialized before proceeding
if recommender is None:
    st.error("Recommender system failed to load. Please check logs for details.")
    st.stop()


# Sidebar for options
st.sidebar.header("Recommendation Options")
recommendation_type = st.sidebar.selectbox(
    "Choose Recommendation Type:",
    ["Popular", "Content-Based (Movie)", "Collaborative Filtering (User)", "Hybrid (User & Movie)"]
)
num_recommendations = st.sidebar.slider("Number of Recommendations:", 5, 20, 10)


# Main Content Area
st.subheader("Get Your Recommendations!")

recommendations_df = pd.DataFrame()
error_message = ""

if recommendation_type == "Popular":
    with st.spinner("Getting popular movies..."):
        recommendations_df = recommender.get_top_popular_movies(num_recommendations)

elif recommendation_type == "Content-Based (Movie)":
    movie_title = st.text_input("Enter a Movie Title:", "The Dark Knight Rises")
    content_source = st.selectbox("Content Source:", ["metadata", "plot"], index=0)

    if st.button("Get Content-Based Recommendations"):
        if movie_title:
            with st.spinner(f"Finding movies similar to '{movie_title}' using {content_source} content..."):
                recs_df = recommender.get_content_based_recommendations(
                    movie_title=movie_title, content_source=content_source, num_recommendations=num_recommendations
                )
                if recs_df.empty or (isinstance(recs_df, pd.DataFrame) and not recs_df.empty and 'title' in recs_df.columns and str(recs_df.iloc[0]['title']).startswith("Movie not found")):
                    error_message = str(recs_df.iloc[0]['title']) if not recs_df.empty else "No recommendations found for this movie."
                else:
                    recommendations_df = recs_df
        else:
            error_message = "Please enter a movie title."

elif recommendation_type == "Collaborative Filtering (User)":
    user_id = st.number_input("Enter User ID (1-671 for small dataset):", min_value=1, max_value=671, value=1)

    if st.button("Get Collaborative Recommendations"):
        with st.spinner(f"Getting personalized recommendations for User ID {user_id}..."):
            recs_df = recommender.get_collaborative_recommendations(
                user_id=int(user_id), num_recommendations=num_recommendations
            )
            if recs_df.empty or (isinstance(recs_df, pd.DataFrame) and not recs_df.empty and 'title' in recs_df.columns and str(recs_df.iloc[0]['title']).startswith("User")):
                 error_message = str(recs_df.iloc[0]['title']) if not recs_df.empty else "No recommendations found for this user."
            else:
                recommendations_df = recs_df

elif recommendation_type == "Hybrid (User & Movie)":
    user_id_hybrid = st.number_input("Enter User ID (for personalization):", min_value=1, max_value=671, value=1)
    movie_title_hybrid = st.text_input("Enter a Seed Movie Title (for content aspect):", "Toy Story")
    content_source_hybrid = st.selectbox("Content Source for Hybrid:", ["metadata", "plot"], index=0)

    if st.button("Get Hybrid Recommendations"):
        if movie_title_hybrid and user_id_hybrid:
            with st.spinner(f"Combining recommendations for User ID {user_id_hybrid} and '{movie_title_hybrid}'..."):
                recs_df = recommender.get_recommendations( # This is the main unified method
                    user_id=int(user_id_hybrid), movie_title=movie_title_hybrid, content_source=content_source_hybrid, num_recommendations=num_recommendations
                )
                if recs_df.empty or (isinstance(recs_df, pd.DataFrame) and not recs_df.empty and 'title' in recs_df.columns and str(recs_df.iloc[0]['title']).startswith("Movie not found")):
                    error_message = str(recs_df.iloc[0]['title']) if not recs_df.empty else "No recommendations found for this user/movie combo."
                else:
                    recommendations_df = recs_df
        else:
            error_message = "Please enter both User ID and a Movie Title for hybrid recommendations."


# Display Recommendations or Error Message
if error_message:
    st.error(error_message)
elif not recommendations_df.empty:
    st.write("---")
    st.write("### Your Recommendations:")
    st.dataframe(recommendations_df)
else:
    st.info("Select options in the sidebar and click the button to get recommendations!")

st.sidebar.markdown("---")
st.sidebar.write("Developed by Your Vikash/IIT Patna")