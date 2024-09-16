import pandas as pd
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
import requests
import os
from dotenv import load_dotenv
import random

# Load the DataFrame and FAISS index
df = pd.read_csv("movies.csv")
model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index("movies_faiss.index")

# Path to the search log file
log_file = "search_log.csv"

# Function to fetch movie poster URL from SerpAPI
def fetch_poster(movie_title):
    api_key = os.getenv("API_KEY")
    search_url = f"https://serpapi.com/search?engine=google&tbm=isch&q={movie_title}&api_key={api_key}"
    try:
        response = requests.get(search_url)
        data = response.json()
        if 'images_results' in data and len(data['images_results']) > 0:
            return data['images_results'][0]['original']
        return "https://via.placeholder.com/500x750.png?text=Image+not+found"
    except Exception as e:
        print(f"Error fetching poster for movie title {movie_title}: {e}")
        return "https://via.placeholder.com/500x750.png?text=Image+not+found"

# Function to recommend movies based on a query
def recommend_movies(query):
    query_embeddings = model.encode([query]).astype('float32')
    k = 25  # Number of similar movies to retrieve for a 5x5 grid
    distances, indices = index.search(query_embeddings, k)
    
    # Create a list of tuples (distance, index) and sort by distance
    sorted_distances_indices = sorted(zip(distances[0], indices[0]), key=lambda x: x[0])
    
    movie_names = []
    movie_posters = []
    
    for distance, idx in sorted_distances_indices:
        movie_title = df.iloc[idx]['title']
        movie_names.append(movie_title)
        movie_posters.append(fetch_poster(movie_title))
        
    return movie_names, movie_posters

# Function to save the last 25 searches in the log
def save_search_log(query):
    if not os.path.exists(log_file):
        log_df = pd.DataFrame(columns=['query'])
    else:
        log_df = pd.read_csv(log_file)

    new_log_entry = pd.DataFrame({'query': [query]})
    log_df = pd.concat([log_df, new_log_entry], ignore_index=True)
    log_df = log_df.tail(25)
    log_df.to_csv(log_file, index=False)

# Function to load search history from log with rankings based on recency
def load_search_log():
    if os.path.exists(log_file):
        log_df = pd.read_csv(log_file)
        log_df = log_df.iloc[::-1].reset_index(drop=True)
        log_df['rank'] = range(1, len(log_df) + 1)
        return log_df
    return pd.DataFrame(columns=['query', 'rank'])

# Function to display movies in a grid with same-sized posters
def display_movies_grid(movie_names, movie_posters, num_cols=5):
    num_rows = (len(movie_names) + num_cols - 1) // num_cols
    for row in range(num_rows):
        cols = st.columns(num_cols)
        for col in range(num_cols):
            idx = row * num_cols + col
            if idx < len(movie_names):
                with cols[col]:
                    st.image(movie_posters[idx], use_column_width=True)
                    st.caption(movie_names[idx])
            else:
                with cols[col]:
                    st.write("")  # Empty cell

# Streamlit app
st.title("Movie Recommendation System")

# Load the search log
search_log = load_search_log()

# Create dropdowns for predefined movie titles and search history
st.sidebar.subheader("Search Options")

# Predefined movie titles dropdown
predefined_movies = df["title"].tolist()
selected_movie = st.sidebar.selectbox("Select a movie title:", predefined_movies, index=0)

# Search history dropdown
search_history = search_log['query'].tolist() if not search_log.empty else ["No recent searches"]
selected_history_query = st.sidebar.selectbox("Select a search history query:", search_history, index=0)

# Text input for custom search
query = st.text_input("Search for movies:", "")

# Automatically update query based on dropdown selection or text input
if selected_movie and selected_movie != "Select a movie title:":
    query = selected_movie
elif selected_history_query and selected_history_query != "No recent searches":
    query = selected_history_query

# Perform search if query is available
if query:
    recommended_movie_names, recommended_movie_posters = recommend_movies(query)
    save_search_log(query)
    st.subheader("Recommended Movies:")
    display_movies_grid(recommended_movie_names, recommended_movie_posters, num_cols=5)
else:
    # Display movies from log based on [8, 6, 5, 3, 3] pattern
    recent_queries_with_ranks = search_log['query'].unique()[-5:]
    search_counts = [8, 6, 5, 3, 3]

    st.subheader("Top 25 Movies from Recent Searches:")
    top_25_movies = []
    top_25_posters = []

    for i, query in enumerate(recent_queries_with_ranks):
        movie_names, movie_posters = recommend_movies(query)
        num_to_add = min(len(movie_names), len(movie_posters), search_counts[i])
        top_25_movies.extend(movie_names[:num_to_add])
        top_25_posters.extend(movie_posters[:num_to_add])

    top_25_movies = top_25_movies[:25]
    top_25_posters = top_25_posters[:25]

    combined_top_25 = list(zip(top_25_movies, top_25_posters))
    random.shuffle(combined_top_25)
    shuffled_top_25_movies, shuffled_top_25_posters = zip(*combined_top_25)
    if len(shuffled_top_25_movies) == len(shuffled_top_25_posters):
        display_movies_grid(shuffled_top_25_movies, shuffled_top_25_posters, num_cols=5)
