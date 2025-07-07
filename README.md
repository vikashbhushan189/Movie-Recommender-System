# 🎬 Advanced Movie Recommender System

[![GitHub last commit](https://img.shields.io/github/last-commit/your-username/your-repo-name)](https://github.com/your-username/your-repo-name)
[![GitHub stars](https://img.shields.io/github/stars/your-username/your-repo-name?style=social)](https://github.com/your-username/your-repo-name)

This repository contains an advanced movie recommendation system developed in Python. It's designed to be modular, efficient, and interactive via a Streamlit web application.

## ✨ Features

-   **Popularity-Based Chart:** Recommends top-rated movies using a robust weighted rating formula, preventing bias from movies with few ratings.
-   **Content-Based Filtering:**
    -   **Plot Similarity:** Uses TF-IDF (Term Frequency-Inverse Document Frequency) on movie plot overviews.
    -   **Metadata Similarity:** Leverages CountVectorizer on combined features like cast, director, keywords, and genres.
    -   Incorporates advanced text preprocessing (lemmatization, stopwords removal) for higher quality textual features.
-   **Collaborative Filtering:** Employs Singular Value Decomposition (SVD) on user-item rating data for personalized recommendations.
-   **Hybrid Recommendation:** Combines collaborative filtering with content-based similarity to offer more comprehensive recommendations, adapting to user preferences and movie features.
-   **Modular Design:** Core logic is encapsulated in the `MovieRecommender` class (`api.py`), enabling clean separation of concerns.
-   **Efficient Model Loading:** Utilizes `pickle` to serialize and deserialize preprocessed data and trained models, ensuring rapid startup for the Streamlit app.
-   **Interactive Web Application (Streamlit):** A user-friendly interface allows real-time interaction with different recommendation types.

## 📊 Key Results & Performance (Metrics)

Below are some aggregated performance metrics from the evaluation phase, showcasing the effectiveness of different recommendation approaches. Metrics like Precision@K, Recall@K, and F1-Score@K indicate the relevance of the top `K` recommendations.

### Top Popular Movies (Weighted Rating)

A snippet of the highest-rated movies according to the weighted rating formula, ensuring fairness by accounting for vote counts.

**Example Chart:**

| Title                     | Vote Count | Vote Average | Score   |
| :------------------------ | :--------- | :----------- | :------ |
| The Shawshank Redemption  | 8358       | 8.5          | 8.445874|
| The Godfather             | 6024       | 8.5          | 8.425445|
| Dilwale Dulhania Le Jayenge | 661        | 9.1          | 8.421501|
| The Dark Knight           | 12269      | 8.3          | 8.265480|
| ...                       | ...        | ...          | ...     |



### Recommendation Model Evaluation Summary (K=10)

This table shows the average Precision, Recall, and F1-Score at K=10 for various recommendation methods, averaged across multiple test users and seed movies.

| Movie                      | Method              | Precision | Recall | F1_Score | Avg_Num_Liked |
| :------------------------- | :------------------ | :-------- | :----- | :--------- | :-------------- |
| Finding Nemo               | Content (Metadata)  | 0.0200    | 0.0005 | 0.0010     | 116.2           |
| Finding Nemo               | Content (Plot)      | 0.0000    | 0.0000 | 0.0000     | 116.2           |
| Finding Nemo               | Hybrid (Plot+WR)    | 0.0000    | 0.0000 | 0.0000     | 116.2           |
| Pulp Fiction               | Content (Metadata)  | 0.0400    | 0.0058 | 0.0101     | 64.0            |
| Pulp Fiction               | Content (Plot)      | 0.0000    | 0.0000 | 0.0000     | 64.0            |
| Pulp Fiction               | Hybrid (Plot+WR)    | 0.0000    | 0.0000 | 0.0000     | 64.0            |
| The Shawshank Redemption   | Content (Metadata)  | 0.0200    | 0.0029 | 0.0051     | 37.2            |
| The Shawshank Redemption   | Content (Plot)      | 0.0000    | 0.0000 | 0.0000     | 37.2            |
| The Shawshank Redemption   | Hybrid (Plot+WR)    | 0.0000    | 0.0000 | 0.0000     | 37.2            |
| Toy Story                  | Content (Metadata)  | 0.0200    | 0.0071 | 0.0105     | 138.8           |
| Toy Story                  | Content (Plot)      | 0.0400    | 0.0081 | 0.0124     | 138.8           |
| Toy Story                  | Hybrid (Plot+WR)    | 0.0400    | 0.0081 | 0.0124     | 138.8           |

### Collaborative Filtering (SVD) Evaluation

Evaluation of the SVD model shows its accuracy in predicting ratings and its recall for specific users.

**RMSE and MAE from Cross-Validation:**

| Metric | Value  |
| :----- | :----- |
| RMSE   | ~0.87 |
| MAE    | ~0.67  |


**Precision/Recall for Example User (K=10):**

| User ID | Precision@10 | Recall@10 | F1-Score@10 | Total Liked Movies |
| :------ | :----------- | :-------- | :---------- | :----------------- |
| 1       | 0.0000       | 0.0000    | 0.0000      | 3                  |




## ⚙️ Setup and Installation

To get this project running locally, follow these steps:

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/your-movie-recommender.git
    cd your_movie_recommender
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install pandas numpy scikit-learn scikit-surprise nltk streamlit
    ```

4.  **Download NLTK Data:**
    NLTK requires some data for text preprocessing.
    ```bash
    python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('stopwords')"
    ```

5.  **Download Datasets:**
    This project uses subsets of the [MovieLens Latest Small Dataset (ml-latest-small.zip)](https://grouplens.org/datasets/movielens/) and [The Movies Dataset from Kaggle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset). Please download the following CSV files and place them into a `./data/` directory within your project root:
    -   `movies_metadata.csv` (from Kaggle)
    -   `credits.csv` (from Kaggle)
    -   `keywords.csv` (from Kaggle)
    -   `ratings_small.csv` (from MovieLens Latest Small, inside ml-latest-small.zip)
    -   `links.csv` (from MovieLens Latest Small, inside ml-latest-small.zip)

    Your `data/` directory should look like this:
    ```
    your_movie_recommender/
    ├── data/
    │   ├── movies_metadata.csv
    │   ├── credits.csv
    │   ├── keywords.csv
    │   ├── ratings_small.csv
    │   └── links.csv
    └── ...
    ```

## ▶️ How to Run

1.  **Generate/Update Saved Models:**
    The `MovieRecommender` class will automatically attempt to load pre-processed data and trained models from `./saved_models_deploy/`. If these files don't exist or are outdated, the system will re-run the full preprocessing and training pipeline (which takes several minutes) and save the new models.

    You can explicitly trigger this or ensure models are saved by running `api.py` once:
    ```bash
    python api.py
    ```
    This command initializes the `MovieRecommender` class, processes the data, trains the models (TF-IDF, CountVectorizer, SVD), and saves the results to `./saved_models_deploy/`.

2.  **Start the Streamlit Application:**
    Once the models are saved (or after the `api.py` execution completes its initial setup), you can launch the interactive recommender web application:
    ```bash
    streamlit run streamlit_app.py
    ```
    This command will open the Streamlit app in your default web browser (usually at `http://localhost:8501`).

## 📁 Repository Structure
```
your_movie_recommender/
├── data/ # Placeholder for raw datasets (download separately)
│ └── .gitkeep
├── saved_models_deploy/ # Stores serialized models and preprocessed data (generated by api.py)
│ └── .gitkeep
├── notebooks/
│ └── Movie_Recommender_Development.ipynb # Your complete Jupyter notebook with code and analysis
├── api.py # Core MovieRecommender class and logic
├── streamlit_app.py # Streamlit web application interface
└── README.md # This file!

```
## 🤝 Contribution

Feel free to fork this repository, explore the code, open issues for bugs, or suggest improvements!