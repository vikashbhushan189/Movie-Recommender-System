import pandas as pd
import numpy as np
import re
from ast import literal_eval
import pickle # For saving/loading models and data

# NLTK imports and dummies 
try:
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    _stop_words_ = set(stopwords.words('english')) 
    _lemmatizer_ = WordNetLemmatizer() # Renamed
    print("NLTK successfully imported and initialized.")
except ImportError:
    print("NLTK not found. Text preprocessing (lemmatization/stopwords) will be skipped.")
    print("Please install it: pip install nltk && python -c 'import nltk; nltk.download(\"punkt\"); nltk.download(\"wordnet\"); nltk.download(\"stopwords\")'")
    def word_tokenize(text): return str(text).split()
    class DummyLemmatizer:
        def lemmatize(self, word): return word
    _lemmatizer_ = DummyLemmatizer()
    class DummyStopwords:
        def words(self, lang): return set()
    _stopwords_instance_ = DummyStopwords()
    _stop_words_ = _stopwords_instance_.words('english') # Assign empty set

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate, train_test_split

class MovieRecommender:
    def __init__(self, data_path='data/'):
        print(f"\n--- Initializing MovieRecommender from {data_path} ---")
        self.data_path = data_path
        self.df = None
        self.ratings = None
        self.links = None
        
        self.title_to_index = None
        self.tfidf_plot = None
        self.tfidf_matrix_plot = None
        self.count_metadata = None
        self.cv_matrix_metadata = None
        self.svd_model = None # SVD model for collaborative filtering

        self._load_data()
        self._preprocess_data()
        self._initialize_recommenders()

    def _load_data(self):
        print("Loading data files...")
        try:
            self.metadata = pd.read_csv(f'{self.data_path}movies_metadata.csv', low_memory=False)
            self.credits = pd.read_csv(f'{self.data_path}credits.csv')
            self.keywords = pd.read_csv(f'{self.data_path}keywords.csv')
            self.ratings = pd.read_csv(f'{self.data_path}ratings_small.csv')
            self.links = pd.read_csv(f'{self.data_path}links.csv')
            print("Data files loaded successfully.")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Missing data file. Ensure '{self.data_path}' contains all required CSVs. Error: {e}")

    def _preprocess_data(self):
        print("Preprocessing data...")
        # Phase 1: Merging, ID alignment, and Weighted Rating
        self.metadata = self.metadata[pd.to_numeric(self.metadata['id'], errors='coerce').notnull()]
        self.metadata['id'] = self.metadata['id'].astype('int')
        self.credits['id'] = self.credits['id'].astype('int')
        self.keywords['id'] = self.keywords['id'].astype('int')

        self.df = self.metadata.merge(self.credits, on='id', how='inner')
        self.df = self.df.merge(self.keywords, on='id', how='inner')
        self.df.drop_duplicates(subset=['id'], inplace=True)

        # ID mapping using links.csv
        self.links = self.links[pd.to_numeric(self.links['tmdbId'], errors='coerce').notnull()]
        self.links['tmdbId'] = self.links['tmdbId'].astype('int')
        tmdbId_to_movieId = {v: k for k, v in self.links.set_index('movieId')['tmdbId'].to_dict().items()}
        self.df['movieId'] = self.df['id'].map(tmdbId_to_movieId)
        self.df.dropna(subset=['movieId'], inplace=True)
        self.df['movieId'] = self.df['movieId'].astype('int')

        self.df['vote_count'] = pd.to_numeric(self.df['vote_count'], errors='coerce').fillna(0)
        self.df['vote_average'] = pd.to_numeric(self.df['vote_average'], errors='coerce')
        C = self.df['vote_average'].mean()
        self.df['vote_average'] = self.df['vote_average'].fillna(C)
        m = self.df['vote_count'].quantile(0.90)

        def weighted_rating_func(x): # Renamed to avoid collision, also using class's m/C
            v = x['vote_count']
            R = x['vote_average']
            if v == 0: return C
            return (v / (v + m)) * R + (m / (m + v)) * C
        self.df['score'] = self.df.apply(weighted_rating_func, axis=1)

        # Phase 2: Feature Extraction and Text Preprocessing
        def safe_literal_eval(val):
            if isinstance(val, str):
                try: return literal_eval(val)
                except (ValueError, SyntaxError): return []
            return []

        features_to_parse = ['cast', 'crew', 'keywords', 'genres']
        for feature in features_to_parse:
            self.df[feature] = self.df[feature].apply(safe_literal_eval)

        def get_director(crew_list):
            for member in crew_list:
                if isinstance(member, dict) and member.get('job') == 'Director':
                    return member.get('name', '')
            return ''
        def get_top_n_names(list_of_dicts, n=3):
            names = []
            if isinstance(list_of_dicts, list):
                for item in list_of_dicts:
                    if isinstance(item, dict) and 'name' in item:
                        names.append(item['name'])
            return names[:n]

        self.df['director'] = self.df['crew'].apply(get_director)
        self.df['cast'] = self.df['cast'].apply(get_top_n_names)
        self.df['keywords'] = self.df['keywords'].apply(get_top_n_names)
        self.df['genres'] = self.df['genres'].apply(get_top_n_names)
        self.df['overview'] = self.df['overview'].fillna('')

        def clean_and_lemmatize_text(text): # Using global NLTK variables defined outside
            if not isinstance(text, str): return ""
            text = re.sub(r'<.*?>', '', text)
            text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
            tokens = word_tokenize(text) # Uses global word_tokenize
            processed_tokens = []
            for word in tokens:
                if word not in _stop_words_: # Uses global _stop_words_
                    processed_tokens.append(_lemmatizer_.lemmatize(word)) # Uses global _lemmatizer_
            return " ".join(processed_tokens)
        self.df['clean_overview'] = self.df['overview'].apply(clean_and_lemmatize_text)
        
        def clean_list_of_strings(lst):
            if isinstance(lst, list):
                return [clean_and_lemmatize_text(item) for item in lst]
            return []
        self.df['processed_keywords'] = self.df['keywords'].apply(clean_list_of_strings)
        self.df['processed_cast'] = self.df['cast'].apply(clean_list_of_strings)
        self.df['processed_genres'] = self.df['genres'].apply(clean_list_of_strings)
        self.df['processed_director'] = self.df['director'].apply(clean_and_lemmatize_text)

        def create_processed_soup(x):
            keywords_str = ' '.join(x['processed_keywords'])
            cast_str = ' '.join(x['processed_cast'])
            genres_str = ' '.join(x['processed_genres'])
            director_str = x['processed_director']
            soup = f"{keywords_str} {cast_str} {director_str} {genres_str}"
            return ' '.join(soup.split())
        self.df['processed_soup'] = self.df.apply(create_processed_soup, axis=1)
        
        # Prepare df_rated_movies for CF alignment
        self.df_rated_movies = self.df[self.df['movieId'].notna()].copy()

        print("Data preprocessing complete.")

    def _initialize_recommenders(self):
        print("Initializing recommenders...")
        self.title_to_index = pd.Series(self.df.index.values, index=self.df['title']).drop_duplicates().to_dict()

        # TF-IDF for plot
        self.tfidf_plot = TfidfVectorizer(stop_words='english', max_df=0.8, min_df=5)
        self.tfidf_matrix_plot = self.tfidf_plot.fit_transform(self.df['clean_overview'])
        print(f"TF-IDF Matrix (Plot) Shape: {self.tfidf_matrix_plot.shape}")

        # CountVectorizer for metadata
        self.count_metadata = CountVectorizer(stop_words='english')
        self.cv_matrix_metadata = self.count_metadata.fit_transform(self.df['processed_soup'])
        print(f"CountVectorizer Matrix (Metadata) Shape: {self.cv_matrix_metadata.shape}")

        # SVD model for collaborative filtering
        if not self.ratings.empty:
            reader = Reader(rating_scale=(0.5, 5))
            data = Dataset.load_from_df(self.ratings[['userId', 'movieId', 'rating']], reader)
            self.svd_model = SVD()
            print("Training SVD model (might take a moment)...")
            trainset = data.build_full_trainset()
            self.svd_model.fit(trainset)
            print("SVD model trained.")
        else:
            print("Ratings data not available, SVD model will not be trained.")
            self.svd_model = None

        print("Recommender initialization complete.")

    def get_top_popular_movies(self, num_recommendations=10):
        """Returns the top N popular movies based on weighted rating."""
        if self.df is None:
            raise RuntimeError("Recommender not initialized. Run _initialize_recommenders() first.")
        
        # Filter for qualified movies first (optional, as `score` is already on all df movies)
        # q_movies = self.df.copy().loc[self.df['vote_count'] >= self.df['vote_count'].quantile(0.90)]
        # top_movies = q_movies.sort_values('score', ascending=False)
        top_movies = self.df.sort_values('score', ascending=False) # `score` is already based on a cutoff `m`
        
        return top_movies[['title', 'vote_count', 'vote_average', 'score']].head(num_recommendations)

    def get_content_based_recommendations(self, movie_title, content_source='plot', num_recommendations=10):
        """
        Generates content-based recommendations for a given movie.
        content_source: 'plot' or 'metadata'
        """
        if movie_title not in self.title_to_index:
            return pd.DataFrame([{"title": "Movie not found in our database.", "similarity_score": np.nan}])

        idx = self.title_to_index[movie_title]
        
        if content_source == 'plot' and self.tfidf_matrix_plot is not None:
            matrix = self.tfidf_matrix_plot
        elif content_source == 'metadata' and self.cv_matrix_metadata is not None:
            matrix = self.cv_matrix_metadata
        else:
            return pd.DataFrame([{"title": "Invalid content source or model not initialized.", "similarity_score": np.nan}])

        movie_vector = matrix[idx]
        sim_scores = cosine_similarity(movie_vector, matrix)
        sim_scores = list(enumerate(sim_scores[0]))

        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = [score for score in sim_scores if score[0] != idx] # Exclude self
        sim_scores = sim_scores[:num_recommendations]

        movie_indices = [i[0] for i in sim_scores]
        recommended_movies = self.df.iloc[movie_indices].copy()
        recommended_movies['similarity_score'] = [s[1] for s in sim_scores]
        return recommended_movies[['title', 'similarity_score']]

    def get_hybrid_recommendations(self, movie_title, content_source='plot', num_recommendations=10):
        """
        Generates hybrid recommendations by combining content similarity with weighted rating score.
        content_source: 'plot' or 'metadata'
        """
        if movie_title not in self.title_to_index:
            return pd.DataFrame([{"title": "Movie not found in our database.", "hybrid_ranking_score": np.nan}])

        idx = self.title_to_index[movie_title]

        if content_source == 'plot' and self.tfidf_matrix_plot is not None:
            matrix = self.tfidf_matrix_plot
        elif content_source == 'metadata' and self.cv_matrix_metadata is not None:
            matrix = self.cv_matrix_metadata
        else:
            return pd.DataFrame([{"title": "Invalid content source or model not initialized.", "hybrid_ranking_score": np.nan}])

        movie_vector = matrix[idx]
        sim_scores_raw = cosine_similarity(movie_vector, matrix)[0]

        candidate_scores = [(i, score) for i, score in enumerate(sim_scores_raw) if i != idx]
        candidate_scores = sorted(candidate_scores, key=lambda x: x[1], reverse=True)[:50] # Take more candidates for re-ranking

        candidate_movie_indices = [i[0] for i in candidate_scores]
        candidate_df = self.df.iloc[candidate_movie_indices].copy()
        
        candidate_similarity_map = {original_idx: score for original_idx, score in candidate_scores}
        candidate_df['content_similarity'] = candidate_df.index.map(candidate_similarity_map)
        
        candidate_df['score'] = pd.to_numeric(candidate_df['score'], errors='coerce')
        max_score = candidate_df['score'].max()
        candidate_df['normalized_score'] = candidate_df['score'] / max_score if max_score > 0 else 0

        content_weight = 0.7
        popularity_weight = 0.3

        candidate_df['hybrid_ranking_score'] = (candidate_df['content_similarity'] * content_weight) + \
                                             (candidate_df['normalized_score'] * popularity_weight)

        final_recommendations = candidate_df.sort_values(by='hybrid_ranking_score', ascending=False)
        return final_recommendations[['title', 'hybrid_ranking_score', 'content_similarity', 'score']].head(num_recommendations)

    def get_collaborative_recommendations(self, user_id, num_recommendations=10):
        """
        Generates collaborative filtering recommendations for a given user ID.
        """
        if self.svd_model is None:
            return pd.DataFrame([{"title": "Collaborative filtering model not initialized (no ratings data).", "predicted_rating": np.nan}])

        user_train_rated_movie_ids = set()
        if user_id in self.svd_model.trainset._raw2inner_id_users:
            inner_user_id = self.svd_model.trainset.to_inner_uid(user_id)
            user_train_rated_movie_ids_inner = {iid for iid, _ in self.svd_model.trainset.ur[inner_user_id]}
            user_train_rated_movie_ids = {self.svd_model.trainset.to_raw_iid(inner_iid) for inner_iid in user_train_rated_movie_ids_inner}

        predictions_list = []
        all_known_raw_iids_from_model = {self.svd_model.trainset.to_raw_iid(inner_id) for inner_id in self.svd_model.trainset.all_items()}
        
        movies_to_predict = self.df_rated_movies[self.df_rated_movies['movieId'].isin(list(all_known_raw_iids_from_model))]
        
        for _, row in movies_to_predict.iterrows():
            movie_id = row['movieId']
            if movie_id not in user_train_rated_movie_ids:
                prediction = self.svd_model.predict(user_id, movie_id)
                predictions_list.append({'movieId': movie_id, 'predicted_rating': prediction.est})

        if not predictions_list:
            return pd.DataFrame([{"title": f"User {user_id} has no unrated movies or not in dataset.", "predicted_rating": np.nan}])

        predictions_df = pd.DataFrame(predictions_list)
        predictions_df = predictions_df.sort_values(by='predicted_rating', ascending=False)

        top_recommendations = pd.merge(
            predictions_df,
            self.df_rated_movies[['movieId', 'title', 'vote_average', 'vote_count', 'score']],
            on='movieId',
            how='left'
        )
        top_recommendations.drop_duplicates(subset=['title'], inplace=True)
        return top_recommendations[['title', 'predicted_rating', 'vote_average', 'vote_count', 'score']].head(num_recommendations)

    def get_recommendations(self, user_id=None, movie_title=None, content_source='metadata', num_recommendations=10):
        """
        Unified recommendation function based on available input.
        - If no user_id or movie_title: Returns popular movies.
        - If movie_title: Returns content-based recommendations.
        - If user_id: Returns collaborative filtering recommendations.
        - (Hybrid logic is currently tied to content-based, can be expanded)
        """
        if user_id is None and movie_title is None:
            print("No specific user or movie provided. Returning top popular movies.")
            return self.get_top_popular_movies(num_recommendations)
        elif user_id is None and movie_title is not None:
            print(f"Movie title '{movie_title}' provided. Returning content-based recommendations.")
            # Decide whether to use simple content-based or hybrid for specific movie
            # For simplicity, let's offer content-based here. User can explicitly call hybrid if needed.
            # OR we could embed a smarter rule here.
            return self.get_content_based_recommendations(movie_title, content_source=content_source, num_recommendations=num_recommendations)
        elif user_id is not None and movie_title is None:
            print(f"User ID {user_id} provided. Returning collaborative filtering recommendations.")
            return self.get_collaborative_recommendations(user_id, num_recommendations)
        else: # Both user_id and movie_title provided - consider a true hybrid approach here
            print(f"User ID {user_id} and movie title '{movie_title}' provided. Employing advanced hybrid (user preference + content).")
            # This is where a more sophisticated hybrid combining user interests (CF) AND item content comes in.
            # For now, let's call CF (as it's personalized to user). You could integrate hybrid item-item similarity.
            # A common approach: Get CF recs, then re-rank/filter by content similarity to `movie_title` or past liked movies.
            
            # Simple demonstration: use CF as the primary personalization
            # Or you could route to your get_hybrid_recommendations if that fits this context (e.g., user is viewing a movie)
            return self.get_collaborative_recommendations(user_id, num_recommendations)
            # Future advanced hybrid here:
            # 1. Get CF recommendations (potential movies user might like)
            # 2. Get content recommendations based on `movie_title`
            # 3. Combine/rerank: give higher scores to movies appearing in both lists, or a weighted average of CF predicted rating and content similarity.


    def save_models(self, path='saved_models/'):
        """Saves trained models and processed data to disk."""
        import os
        os.makedirs(path, exist_ok=True)
        print(f"Saving models and data to '{path}'...")
        with open(f'{path}df_recommender.pkl', 'wb') as f:
            pickle.dump(self.df, f)
        with open(f'{path}tfidf_plot.pkl', 'wb') as f:
            pickle.dump(self.tfidf_plot, f)
        with open(f'{path}tfidf_matrix_plot.pkl', 'wb') as f:
            pickle.dump(self.tfidf_matrix_plot, f)
        with open(f'{path}count_metadata.pkl', 'wb') as f:
            pickle.dump(self.count_metadata, f)
        with open(f'{path}cv_matrix_metadata.pkl', 'wb') as f:
            pickle.dump(self.cv_matrix_metadata, f)
        if self.svd_model:
            # Surprise models have their own save/load methods or can be pickled
            # from surprise import dump
            # dump.dump(f'{path}svd_model.pkl', self.svd_model) # More robust save for surprise models
            with open(f'{path}svd_model.pkl', 'wb') as f:
                 pickle.dump(self.svd_model, f)
        with open(f'{path}title_to_index.pkl', 'wb') as f:
            pickle.dump(self.title_to_index, f)
        with open(f'{path}df_rated_movies.pkl', 'wb') as f:
            pickle.dump(self.df_rated_movies, f) # Also save the filtered df for CF
        
        print("Models and data saved successfully.")

    @classmethod
    def load_models(cls, path='saved_models/'):
        """Loads trained models and processed data from disk into a new Recommender instance."""
        import os
        if not os.path.exists(f'{path}df_recommender.pkl'):
            raise FileNotFoundError(f"Saved models not found in '{path}'. Please train and save first.")
        
        print(f"Loading models and data from '{path}'...")
        instance = cls.__new__(cls) # Create a new instance without calling __init__
        with open(f'{path}df_recommender.pkl', 'rb') as f:
            instance.df = pickle.load(f)
        with open(f'{path}tfidf_plot.pkl', 'rb') as f:
            instance.tfidf_plot = pickle.load(f)
        with open(f'{path}tfidf_matrix_plot.pkl', 'rb') as f:
            instance.tfidf_matrix_plot = pickle.load(f)
        with open(f'{path}count_metadata.pkl', 'rb') as f:
            instance.count_metadata = pickle.load(f)
        with open(f'{path}cv_matrix_metadata.pkl', 'rb') as f:
            instance.cv_matrix_metadata = pickle.load(f)
        if os.path.exists(f'{path}svd_model.pkl'):
            with open(f'{path}svd_model.pkl', 'rb') as f:
                instance.svd_model = pickle.load(f)
        else:
            instance.svd_model = None # If SVD was not saved
        with open(f'{path}title_to_index.pkl', 'rb') as f:
            instance.title_to_index = pickle.load(f)
        with open(f'{path}df_rated_movies.pkl', 'rb') as f:
            instance.df_rated_movies = pickle.load(f)

        # Set other attributes to None if they're not explicitly loaded or not needed for loaded state
        instance.data_path = path # Adjust data_path to the loading path
        instance.ratings = None # Ratings only needed for initial training
        instance.metadata = None
        instance.credits = None
        instance.keywords = None
        instance.links = None

        print("Models and data loaded successfully.")
        return instance


# --- Example Usage (Replaced original test calls) ---
if __name__ == '__main__':
    # Initialize and train the recommender
    # This simulates your current script's full execution
    recommender = MovieRecommender(data_path='data/')

    print("\n--- Testing Unified Recommender Function ---")

    print("\n1. Popular Movies:")
    print(recommender.get_recommendations(num_recommendations=5).to_string())

    print("\n2. Content-Based (Plot) for 'Interstellar':")
    print(recommender.get_recommendations(movie_title='Interstellar', content_source='plot', num_recommendations=5).to_string())

    print("\n3. Content-Based (Metadata) for 'Mean Girls':")
    print(recommender.get_recommendations(movie_title='Mean Girls', content_source='metadata', num_recommendations=5).to_string())

    print("\n4. Collaborative Filtering for User ID 1:")
    print(recommender.get_recommendations(user_id=1, num_recommendations=5).to_string())

    print("\n5. Testing cold start with non-existent movie:")
    print(recommender.get_recommendations(movie_title='A Movie That Does Not Exist 123', num_recommendations=3).to_string())


    # --- Test Saving and Loading ---
    print("\n--- Testing Saving and Loading Models ---")
    recommender.save_models('saved_models_deploy/')

    # Simulate restarting the application by loading saved models
    print("\n--- Loading Recommender from Saved Models ---")
    loaded_recommender = MovieRecommender.load_models('saved_models_deploy/')

    print("\n--- Testing Loaded Recommender (should be fast) ---")
    print("\n1. Popular Movies from loaded model:")
    print(loaded_recommender.get_recommendations(num_recommendations=3).to_string())

    print("\n2. CF for User ID 2 (from loaded model):")
    print(loaded_recommender.get_recommendations(user_id=2, num_recommendations=3).to_string())

    # Example: you could run evaluation using the loaded_recommender too
    # The `eval_results` and evaluation functions from your previous code
    # could be adapted to run against `loaded_recommender.get_recommendations` or its specific methods.

    print("\n--- Deployment Setup Complete ---")