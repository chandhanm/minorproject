# ============================================================
# PHASE 5
# HYBRID MOVIE RECOMMENDATION SYSTEM
#
# Combines:
# 1. Content-Based Filtering
# 2. Collaborative Filtering using SVD
# ============================================================

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise.model_selection import train_test_split


# ============================================================
# HYBRID RECOMMENDER CLASS
# ============================================================

class HybridRecommender:

    def __init__(
        self,
        movies_path="data/movies.csv",
        ratings_path="data/ratings.csv",
        content_weight=0.4,
        collaborative_weight=0.6,
        n_factors=100,
        n_epochs=20,
        lr_all=0.005,
        reg_all=0.02,
        random_state=42
    ):

        self.movies_path = movies_path
        self.ratings_path = ratings_path

        self.content_weight = content_weight
        self.collaborative_weight = collaborative_weight

        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr_all = lr_all
        self.reg_all = reg_all
        self.random_state = random_state

        self.movies = None
        self.ratings = None

        self.tfidf_matrix = None
        self.content_similarity = None
        self.title_to_index = None

        self.data = None
        self.trainset = None
        self.testset = None
        self.svd_model = None

        self.prepare_movies()
        self.prepare_ratings()
        self.build_content_model()
        self.build_collaborative_model()


    # ========================================================
    # 1. LOAD MOVIES
    # ========================================================

    def prepare_movies(self):

        print("\nLoading movies dataset...")

        self.movies = pd.read_csv(
            self.movies_path
        )

        self.movies = self.movies[
            [
                "movieId",
                "title",
                "genres"
            ]
        ].copy()

        # Handle missing values
        self.movies["title"] = (
            self.movies["title"]
            .fillna("Unknown Movie")
        )

        self.movies["genres"] = (
            self.movies["genres"]
            .fillna("Unknown")
        )

        # Convert movie ID to integer
        self.movies["movieId"] = (
            self.movies["movieId"].astype(int)
        )

        # Convert | into spaces
        self.movies["genres_text"] = (
            self.movies["genres"]
            .str.replace("|", " ", regex=False)
        )

        print(
            "Movies loaded:",
            len(self.movies)
        )


    # ========================================================
    # 2. LOAD RATINGS
    # ========================================================

    def prepare_ratings(self):

        print("\nLoading ratings dataset...")

        self.ratings = pd.read_csv(
            self.ratings_path
        )

        self.ratings = self.ratings[
            [
                "userId",
                "movieId",
                "rating"
            ]
        ].copy()

        # Remove missing values
        self.ratings.dropna(
            subset=[
                "userId",
                "movieId",
                "rating"
            ],
            inplace=True
        )

        # Remove duplicates
        self.ratings.drop_duplicates(
            inplace=True
        )

        # Correct data types
        self.ratings["userId"] = (
            self.ratings["userId"].astype(int)
        )

        self.ratings["movieId"] = (
            self.ratings["movieId"].astype(int)
        )

        self.ratings["rating"] = (
            self.ratings["rating"].astype(float)
        )

        print(
            "Ratings loaded:",
            len(self.ratings)
        )

        print(
            "Users:",
            self.ratings["userId"].nunique()
        )

        print(
            "Movies with ratings:",
            self.ratings["movieId"].nunique()
        )


    # ========================================================
    # 3. BUILD CONTENT-BASED MODEL
    # ========================================================

    def build_content_model(self):

        print("\nBuilding content-based model...")

        vectorizer = TfidfVectorizer(
            stop_words="english"
        )

        self.tfidf_matrix = vectorizer.fit_transform(
            self.movies["genres_text"]
        )

        self.content_similarity = cosine_similarity(
            self.tfidf_matrix
        )

        self.title_to_index = pd.Series(
            self.movies.index,
            index=self.movies["title"]
        ).drop_duplicates()

        print(
            "Content-based model built successfully."
        )


    # ========================================================
    # 4. BUILD COLLABORATIVE MODEL
    # ========================================================

    def build_collaborative_model(self):

        print(
            "\nBuilding collaborative filtering model..."
        )

        min_rating = self.ratings["rating"].min()
        max_rating = self.ratings["rating"].max()

        reader = Reader(
            rating_scale=(
                min_rating,
                max_rating
            )
        )

        self.data = Dataset.load_from_df(
            self.ratings[
                [
                    "userId",
                    "movieId",
                    "rating"
                ]
            ],
            reader
        )

        # Train/test split
        self.trainset, self.testset = train_test_split(
            self.data,
            test_size=0.20,
            random_state=self.random_state
        )

        # Create SVD model
        self.svd_model = SVD(
            n_factors=self.n_factors,
            n_epochs=self.n_epochs,
            lr_all=self.lr_all,
            reg_all=self.reg_all,
            random_state=self.random_state
        )

        # Train
        self.svd_model.fit(
            self.trainset
        )

        print(
            "Collaborative filtering model built successfully."
        )


    # ========================================================
    # 5. CONTENT SCORE
    # ========================================================

    def get_content_scores(
        self,
        movie_title
    ):

        if movie_title not in self.title_to_index:

            raise ValueError(
                f"Movie '{movie_title}' was not found."
            )

        movie_index = self.title_to_index[
            movie_title
        ]

        similarity_scores = (
            self.content_similarity[movie_index]
        )

        return similarity_scores


    # ========================================================
    # 6. COLLABORATIVE SCORE
    # ========================================================

    def get_collaborative_scores(
        self,
        user_id
    ):

        # Get all movie IDs
        movie_ids = self.movies[
            "movieId"
        ].tolist()

        predicted_ratings = []

        for movie_id in movie_ids:

            prediction = self.svd_model.predict(
                user_id,
                movie_id
            )

            predicted_ratings.append(
                prediction.est
            )

        predicted_ratings = np.array(
            predicted_ratings
        )

        # Normalize predictions between 0 and 1
        min_pred = predicted_ratings.min()
        max_pred = predicted_ratings.max()

        if max_pred == min_pred:

            normalized_scores = np.ones(
                len(predicted_ratings)
            )

        else:

            normalized_scores = (
                predicted_ratings - min_pred
            ) / (
                max_pred - min_pred
            )

        return normalized_scores


    # ========================================================
    # 7. GET USER RATED MOVIES
    # ========================================================

    def get_user_rated_movies(
        self,
        user_id
    ):

        user_ratings = self.ratings[
            self.ratings["userId"] == user_id
        ]

        return set(
            user_ratings["movieId"].tolist()
        )


    # ========================================================
    # 8. HYBRID RECOMMENDATION
    # ========================================================

    def recommend(
        self,
        user_id,
        movie_title,
        n=10
    ):

        # ----------------------------------------------------
        # Validate movie title
        # ----------------------------------------------------

        if movie_title not in self.title_to_index:

            raise ValueError(
                f"Movie '{movie_title}' was not found."
            )


        # ----------------------------------------------------
        # Content scores
        # ----------------------------------------------------

        content_scores = self.get_content_scores(
            movie_title
        )


        # ----------------------------------------------------
        # Collaborative scores
        # ----------------------------------------------------

        collaborative_scores = (
            self.get_collaborative_scores(
                user_id
            )
        )


        # ----------------------------------------------------
        # Hybrid score
        # ----------------------------------------------------

        hybrid_scores = (
            self.content_weight * content_scores
            +
            self.collaborative_weight
            * collaborative_scores
        )


        # ----------------------------------------------------
        # Create recommendation dataframe
        # ----------------------------------------------------

        recommendations = self.movies[
            [
                "movieId",
                "title",
                "genres"
            ]
        ].copy()

        recommendations[
            "content_score"
        ] = content_scores

        recommendations[
            "collaborative_score"
        ] = collaborative_scores

        recommendations[
            "hybrid_score"
        ] = hybrid_scores


        # ----------------------------------------------------
        # Remove selected movie
        # ----------------------------------------------------

        recommendations = recommendations[
            recommendations["title"] != movie_title
        ]


        # ----------------------------------------------------
        # Remove movies already rated by user
        # ----------------------------------------------------

        rated_movies = self.get_user_rated_movies(
            user_id
        )

        recommendations = recommendations[
            ~recommendations["movieId"].isin(
                rated_movies
            )
        ]


        # ----------------------------------------------------
        # Sort by hybrid score
        # ----------------------------------------------------

        recommendations = (
            recommendations
            .sort_values(
                by="hybrid_score",
                ascending=False
            )
            .head(n)
            .copy()
        )


        # ----------------------------------------------------
        # Add ranking
        # ----------------------------------------------------

        recommendations.insert(
            0,
            "rank",
            range(
                1,
                len(recommendations) + 1
            )
        )


        # ----------------------------------------------------
        # Round scores
        # ----------------------------------------------------

        recommendations[
            "content_score"
        ] = recommendations[
            "content_score"
        ].round(4)

        recommendations[
            "collaborative_score"
        ] = recommendations[
            "collaborative_score"
        ].round(4)

        recommendations[
            "hybrid_score"
        ] = recommendations[
            "hybrid_score"
        ].round(4)


        # ----------------------------------------------------
        # Reset index
        # ----------------------------------------------------

        recommendations.reset_index(
            drop=True,
            inplace=True
        )

        return recommendations


    # ========================================================
    # 9. USER-ONLY HYBRID RECOMMENDATIONS
    # ========================================================

    def recommend_for_user(
        self,
        user_id,
        n=10
    ):

        """
        Generate personalized recommendations for a user
        using collaborative filtering only as the preference
        signal and movie similarity as an additional signal.
        """

        rated_movies = self.get_user_rated_movies(
            user_id
        )

        # Get predicted ratings
        predicted_ratings = []

        for movie_id in self.movies[
            "movieId"
        ]:

            prediction = self.svd_model.predict(
                user_id,
                movie_id
            )

            predicted_ratings.append(
                prediction.est
            )

        candidates = self.movies.copy()

        candidates[
            "predicted_rating"
        ] = predicted_ratings

        # Remove already rated movies
        candidates = candidates[
            ~candidates["movieId"].isin(
                rated_movies
            )
        ]

        # Sort
        candidates = (
            candidates
            .sort_values(
                by="predicted_rating",
                ascending=False
            )
            .head(n)
            .copy()
        )

        candidates[
            "predicted_rating"
        ] = candidates[
            "predicted_rating"
        ].round(2)

        candidates.insert(
            0,
            "rank",
            range(
                1,
                len(candidates) + 1
            )
        )

        candidates.reset_index(
            drop=True,
            inplace=True
        )

        return candidates


# ============================================================
# 10. MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 65)
    print("PHASE 5 - HYBRID MOVIE RECOMMENDATION SYSTEM")
    print("=" * 65)


    # --------------------------------------------------------
    # Create Hybrid Recommender
    # --------------------------------------------------------

    recommender = HybridRecommender(
        movies_path="data/movies.csv",
        ratings_path="data/ratings.csv",
        content_weight=0.4,
        collaborative_weight=0.6,
        n_factors=100,
        n_epochs=20,
        lr_all=0.005,
        reg_all=0.02,
        random_state=42
    )


    # --------------------------------------------------------
    # Test User and Movie
    # --------------------------------------------------------

    test_user_id = 1

    test_movie = "Toy Story (1995)"


    # --------------------------------------------------------
    # Generate Hybrid Recommendations
    # --------------------------------------------------------

    recommendations = recommender.recommend(
        user_id=test_user_id,
        movie_title=test_movie,
        n=10
    )


    # --------------------------------------------------------
    # Display Results
    # --------------------------------------------------------

    print("\n")
    print("=" * 65)
    print(
        f"HYBRID RECOMMENDATIONS FOR USER {test_user_id}"
    )
    print(
        f"Based on: {test_movie}"
    )
    print("=" * 65)

    print(
        recommendations.to_string(
            index=False
        )
    )


    # --------------------------------------------------------
    # Completion Message
    # --------------------------------------------------------

    print("\n")
    print("=" * 65)
    print("PHASE 5 COMPLETED SUCCESSFULLY")
    print("=" * 65)

    print(
        "\nContent-Based Weight:",
        recommender.content_weight
    )

    print(
        "Collaborative Weight:",
        recommender.collaborative_weight
    )