# ============================================================
# PHASE 6
# MODEL EVALUATION
# MOVIE RECOMMENDATION SYSTEM
#
# Evaluates:
# 1. Collaborative Filtering - SVD
# 2. Content-Based Filtering
# 3. Hybrid Recommendation
#
# Metrics:
# - RMSE
# - MAE
# - Precision@K
# - Recall@K
# - Coverage
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise import accuracy
from surprise.model_selection import train_test_split


# ============================================================
# 2. RECOMMENDATION EVALUATOR CLASS
# ============================================================

class RecommendationEvaluator:

    def __init__(
        self,
        movies_path="data/movies.csv",
        ratings_path="data/ratings.csv",
        n_factors=100,
        n_epochs=20,
        lr_all=0.005,
        reg_all=0.02,
        random_state=42
    ):

        # ----------------------------------------------------
        # File paths
        # ----------------------------------------------------

        self.movies_path = movies_path
        self.ratings_path = ratings_path

        # ----------------------------------------------------
        # SVD parameters
        # ----------------------------------------------------

        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr_all = lr_all
        self.reg_all = reg_all
        self.random_state = random_state

        # ----------------------------------------------------
        # Data variables
        # ----------------------------------------------------

        self.movies = None
        self.ratings = None

        self.train_ratings = None
        self.test_ratings = None

        # ----------------------------------------------------
        # Surprise variables
        # ----------------------------------------------------

        self.data = None
        self.trainset = None
        self.testset = None

        self.svd_model = None

        # ----------------------------------------------------
        # Content-based variables
        # ----------------------------------------------------

        self.vectorizer = None
        self.tfidf_matrix = None

        self.movie_id_to_index = None

        # ----------------------------------------------------
        # Load and prepare data
        # ----------------------------------------------------

        self.load_data()

        self.prepare_content_model()

        self.prepare_collaborative_model()


    # ========================================================
    # 3. LOAD MOVIES AND RATINGS
    # ========================================================

    def load_data(self):

        print("\n")
        print("=" * 65)
        print("LOADING DATA")
        print("=" * 65)

        # ----------------------------------------------------
        # Load movies
        # ----------------------------------------------------

        print("\nLoading movies.csv...")

        self.movies = pd.read_csv(
            self.movies_path
        )

        # Required columns
        self.movies = self.movies[
            [
                "movieId",
                "title",
                "genres"
            ]
        ].copy()

        # Missing values
        self.movies["title"] = (
            self.movies["title"]
            .fillna("Unknown Movie")
        )

        self.movies["genres"] = (
            self.movies["genres"]
            .fillna("Unknown")
        )

        # Data types
        self.movies["movieId"] = (
            self.movies["movieId"]
            .astype(int)
        )

        # Remove duplicate movie IDs
        self.movies = (
            self.movies
            .drop_duplicates(
                subset=["movieId"]
            )
            .reset_index(drop=True)
        )

        print(
            "Movies loaded:",
            len(self.movies)
        )

        # ----------------------------------------------------
        # Load ratings
        # ----------------------------------------------------

        print("\nLoading ratings.csv...")

        self.ratings = pd.read_csv(
            self.ratings_path
        )

        # Required columns
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

        # Remove duplicate rows
        self.ratings.drop_duplicates(
            inplace=True
        )

        # Data types
        self.ratings["userId"] = (
            self.ratings["userId"]
            .astype(int)
        )

        self.ratings["movieId"] = (
            self.ratings["movieId"]
            .astype(int)
        )

        self.ratings["rating"] = (
            self.ratings["rating"]
            .astype(float)
        )

        self.ratings.reset_index(
            drop=True,
            inplace=True
        )

        print(
            "Ratings loaded:",
            len(self.ratings)
        )

        print(
            "Number of users:",
            self.ratings["userId"].nunique()
        )

        print(
            "Number of rated movies:",
            self.ratings["movieId"].nunique()
        )

        print(
            "Rating range:",
            self.ratings["rating"].min(),
            "to",
            self.ratings["rating"].max()
        )


    # ========================================================
    # 4. BUILD CONTENT-BASED MODEL
    # ========================================================

    def prepare_content_model(self):

        print("\n")
        print("=" * 65)
        print("BUILDING CONTENT-BASED MODEL")
        print("=" * 65)

        # ----------------------------------------------------
        # Convert genre format
        #
        # Adventure|Animation|Children
        #
        # becomes
        #
        # Adventure Animation Children
        # ----------------------------------------------------

        genres_text = (
            self.movies["genres"]
            .str.replace(
                "|",
                " ",
                regex=False
            )
        )

        # ----------------------------------------------------
        # TF-IDF
        # ----------------------------------------------------

        self.vectorizer = TfidfVectorizer(
            stop_words="english"
        )

        self.tfidf_matrix = (
            self.vectorizer.fit_transform(
                genres_text
            )
        )

        # ----------------------------------------------------
        # Movie ID -> matrix index
        # ----------------------------------------------------

        self.movie_id_to_index = pd.Series(
            self.movies.index,
            index=self.movies["movieId"]
        )

        print(
            "TF-IDF matrix shape:",
            self.tfidf_matrix.shape
        )

        print(
            "Content-based model ready."
        )


    # ========================================================
    # 5. BUILD COLLABORATIVE FILTERING MODEL
    # ========================================================

    def prepare_collaborative_model(self):

        print("\n")
        print("=" * 65)
        print("BUILDING COLLABORATIVE FILTERING MODEL")
        print("=" * 65)

        # ----------------------------------------------------
        # Rating scale
        # ----------------------------------------------------

        min_rating = (
            self.ratings["rating"].min()
        )

        max_rating = (
            self.ratings["rating"].max()
        )

        reader = Reader(
            rating_scale=(
                min_rating,
                max_rating
            )
        )

        # ----------------------------------------------------
        # Create Surprise dataset
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Train/test split
        # ----------------------------------------------------

        self.trainset, self.testset = (
            train_test_split(
                self.data,
                test_size=0.20,
                random_state=self.random_state
            )
        )

        print(
            "Training ratings:",
            self.trainset.n_ratings
        )

        print(
            "Testing ratings:",
            len(self.testset)
        )

        # ----------------------------------------------------
        # Convert training set back to normal IDs
        # ----------------------------------------------------

        train_pairs = []

        for inner_user_id in (
            self.trainset.all_users()
        ):

            raw_user_id = (
                self.trainset.to_raw_uid(
                    inner_user_id
                )
            )

            for (
                inner_item_id,
                rating
            ) in self.trainset.ur[
                inner_user_id
            ]:

                raw_movie_id = (
                    self.trainset.to_raw_iid(
                        inner_item_id
                    )
                )

                train_pairs.append(
                    [
                        int(raw_user_id),
                        int(raw_movie_id),
                        float(rating)
                    ]
                )

        self.train_ratings = pd.DataFrame(
            train_pairs,
            columns=[
                "userId",
                "movieId",
                "rating"
            ]
        )

        # ----------------------------------------------------
        # Convert test set to DataFrame
        # ----------------------------------------------------

        self.test_ratings = pd.DataFrame(
            self.testset,
            columns=[
                "userId",
                "movieId",
                "rating"
            ]
        )

        self.test_ratings["userId"] = (
            self.test_ratings["userId"]
            .astype(int)
        )

        self.test_ratings["movieId"] = (
            self.test_ratings["movieId"]
            .astype(int)
        )

        self.test_ratings["rating"] = (
            self.test_ratings["rating"]
            .astype(float)
        )

        # ----------------------------------------------------
        # Create SVD model
        # ----------------------------------------------------

        self.svd_model = SVD(
            n_factors=self.n_factors,
            n_epochs=self.n_epochs,
            lr_all=self.lr_all,
            reg_all=self.reg_all,
            random_state=self.random_state
        )

        print("\nTraining SVD model...")

        self.svd_model.fit(
            self.trainset
        )

        print(
            "SVD model training completed."
        )


    # ========================================================
    # 6. EVALUATE RMSE AND MAE
    # ========================================================

    def evaluate_svd_error(self):

        print("\n")
        print("=" * 65)
        print("COLLABORATIVE FILTERING ERROR METRICS")
        print("=" * 65)

        # ----------------------------------------------------
        # Generate predictions
        # ----------------------------------------------------

        predictions = (
            self.svd_model.test(
                self.testset
            )
        )

        # ----------------------------------------------------
        # RMSE
        # ----------------------------------------------------

        print("\nCalculating RMSE...")

        rmse = accuracy.rmse(
            predictions,
            verbose=True
        )

        # ----------------------------------------------------
        # MAE
        # ----------------------------------------------------

        print("\nCalculating MAE...")

        mae = accuracy.mae(
            predictions,
            verbose=True
        )

        print("\n")
        print("-" * 65)
        print(
            f"RMSE: {rmse:.4f}"
        )
        print(
            f"MAE : {mae:.4f}"
        )
        print("-" * 65)

        return rmse, mae


    # ========================================================
    # 7. GET MOVIES RATED BY USER IN TRAINING DATA
    # ========================================================

    def get_train_rated_movies(
        self,
        user_id
    ):

        user_data = self.train_ratings[
            self.train_ratings["userId"] == user_id
        ]

        return set(
            user_data["movieId"].tolist()
        )


    # ========================================================
    # 8. GET RELEVANT TEST MOVIES
    # ========================================================

    def get_test_relevant_movies(
        self,
        user_id,
        threshold=4.0
    ):

        user_test_data = self.test_ratings[
            self.test_ratings["userId"] == user_id
        ]

        relevant_movies = user_test_data[
            user_test_data["rating"] >= threshold
        ]

        return set(
            relevant_movies["movieId"].tolist()
        )


    # ========================================================
    # 9. COLLABORATIVE TOP-K RECOMMENDATIONS
    # ========================================================

    def collaborative_recommend(
        self,
        user_id,
        k=10
    ):

        # ----------------------------------------------------
        # Movies already rated in training data
        # ----------------------------------------------------

        rated_movies = (
            self.get_train_rated_movies(
                user_id
            )
        )

        # ----------------------------------------------------
        # Candidate movies
        # ----------------------------------------------------

        candidates = self.movies[
            ~self.movies["movieId"].isin(
                rated_movies
            )
        ].copy()

        # ----------------------------------------------------
        # Predict ratings
        # ----------------------------------------------------

        predictions = []

        for movie_id in candidates[
            "movieId"
        ]:

            prediction = (
                self.svd_model.predict(
                    user_id,
                    movie_id
                )
            )

            predictions.append(
                prediction.est
            )

        candidates["score"] = (
            predictions
        )

        # ----------------------------------------------------
        # Top K
        # ----------------------------------------------------

        recommendations = (
            candidates
            .sort_values(
                by="score",
                ascending=False
            )
            .head(k)
            .copy()
        )

        return recommendations


    # ========================================================
    # 10. CONTENT SCORES FOR USER
    # ========================================================

    def content_scores_for_user(
        self,
        user_id
    ):

        # ----------------------------------------------------
        # Get user's training ratings
        # ----------------------------------------------------

        user_train = self.train_ratings[
            self.train_ratings["userId"] == user_id
        ]

        # ----------------------------------------------------
        # Movies liked by user
        # Rating >= 4.0
        # ----------------------------------------------------

        liked_movies = user_train[
            user_train["rating"] >= 4.0
        ]["movieId"].tolist()

        # ----------------------------------------------------
        # If user has no liked movies
        # ----------------------------------------------------

        if len(liked_movies) == 0:

            return np.zeros(
                len(self.movies)
            )

        # ----------------------------------------------------
        # Convert movie IDs to matrix indexes
        # ----------------------------------------------------

        liked_indexes = []

        for movie_id in liked_movies:

            if movie_id in self.movie_id_to_index:

                movie_index = (
                    self.movie_id_to_index[
                        movie_id
                    ]
                )

                liked_indexes.append(
                    int(movie_index)
                )

        # ----------------------------------------------------
        # If no valid indexes
        # ----------------------------------------------------

        if len(liked_indexes) == 0:

            return np.zeros(
                len(self.movies)
            )

        # ----------------------------------------------------
        # Calculate similarity only between:
        #
        # liked movies
        #
        # and
        #
        # all movies
        #
        # This avoids creating a huge
        # 10329 x 10329 matrix.
        # ----------------------------------------------------

        similarity_matrix = linear_kernel(
            self.tfidf_matrix[
                liked_indexes
            ],
            self.tfidf_matrix
        )

        # ----------------------------------------------------
        # For each candidate movie, use its highest
        # similarity to any movie liked by the user.
        # ----------------------------------------------------

        content_scores = (
            similarity_matrix.max(
                axis=0
            )
        )

        return np.asarray(
            content_scores
        )


    # ========================================================
    # 11. CONTENT-BASED TOP-K RECOMMENDATIONS
    # ========================================================

    def content_recommend(
        self,
        user_id,
        k=10
    ):

        # ----------------------------------------------------
        # Get already rated movies
        # ----------------------------------------------------

        rated_movies = (
            self.get_train_rated_movies(
                user_id
            )
        )

        # ----------------------------------------------------
        # Calculate content scores
        # ----------------------------------------------------

        scores = (
            self.content_scores_for_user(
                user_id
            )
        )

        # ----------------------------------------------------
        # Create candidates
        # ----------------------------------------------------

        candidates = self.movies.copy()

        candidates["score"] = (
            scores
        )

        # ----------------------------------------------------
        # Remove already rated movies
        # ----------------------------------------------------

        candidates = candidates[
            ~candidates["movieId"].isin(
                rated_movies
            )
        ]

        # ----------------------------------------------------
        # Sort and select Top K
        # ----------------------------------------------------

        recommendations = (
            candidates
            .sort_values(
                by="score",
                ascending=False
            )
            .head(k)
            .copy()
        )

        return recommendations


    # ========================================================
    # 12. NORMALIZE SCORES
    # ========================================================

    @staticmethod
    def normalize_scores(
        scores
    ):

        scores = np.asarray(
            scores,
            dtype=float
        )

        if len(scores) == 0:

            return scores

        min_score = scores.min()
        max_score = scores.max()

        # Avoid division by zero
        if max_score == min_score:

            return np.zeros(
                len(scores)
            )

        normalized = (
            scores - min_score
        ) / (
            max_score - min_score
        )

        return normalized


    # ========================================================
    # 13. HYBRID TOP-K RECOMMENDATIONS
    # ========================================================

    def hybrid_recommend(
        self,
        user_id,
        k=10,
        content_weight=0.4,
        collaborative_weight=0.6
    ):

        # ----------------------------------------------------
        # Movies already rated
        # ----------------------------------------------------

        rated_movies = (
            self.get_train_rated_movies(
                user_id
            )
        )

        # ----------------------------------------------------
        # Candidate movies
        # ----------------------------------------------------

        candidates = self.movies[
            ~self.movies["movieId"].isin(
                rated_movies
            )
        ].copy()

        # ----------------------------------------------------
        # Content scores for all movies
        # ----------------------------------------------------

        all_content_scores = (
            self.content_scores_for_user(
                user_id
            )
        )

        # ----------------------------------------------------
        # Collaborative predictions
        # ----------------------------------------------------

        collaborative_scores = []

        for movie_id in candidates[
            "movieId"
        ]:

            prediction = (
                self.svd_model.predict(
                    user_id,
                    movie_id
                )
            )

            collaborative_scores.append(
                prediction.est
            )

        collaborative_scores = np.asarray(
            collaborative_scores
        )

        # ----------------------------------------------------
        # Extract content scores for candidates
        # ----------------------------------------------------

        content_scores = []

        for movie_id in candidates[
            "movieId"
        ]:

            movie_index = (
                self.movie_id_to_index[
                    movie_id
                ]
            )

            content_scores.append(
                all_content_scores[
                    int(movie_index)
                ]
            )

        content_scores = np.asarray(
            content_scores
        )

        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        normalized_content = (
            self.normalize_scores(
                content_scores
            )
        )

        normalized_collaborative = (
            self.normalize_scores(
                collaborative_scores
            )
        )

        # ----------------------------------------------------
        # Calculate hybrid score
        # ----------------------------------------------------

        hybrid_scores = (
            content_weight
            * normalized_content
            +
            collaborative_weight
            * normalized_collaborative
        )

        candidates["score"] = (
            hybrid_scores
        )

        # ----------------------------------------------------
        # Sort
        # ----------------------------------------------------

        recommendations = (
            candidates
            .sort_values(
                by="score",
                ascending=False
            )
            .head(k)
            .copy()
        )

        return recommendations


    # ========================================================
    # 14. PRECISION@K
    # ========================================================

    @staticmethod
    def precision_at_k(
        recommended_movies,
        relevant_movies,
        k
    ):

        recommended_movies = (
            recommended_movies[:k]
        )

        if len(recommended_movies) == 0:

            return 0.0

        hits = len(
            set(recommended_movies)
            .intersection(
                relevant_movies
            )
        )

        precision = (
            hits
            /
            len(recommended_movies)
        )

        return precision


    # ========================================================
    # 15. RECALL@K
    # ========================================================

    @staticmethod
    def recall_at_k(
        recommended_movies,
        relevant_movies,
        k
    ):

        if len(relevant_movies) == 0:

            return 0.0

        recommended_movies = (
            recommended_movies[:k]
        )

        hits = len(
            set(recommended_movies)
            .intersection(
                relevant_movies
            )
        )

        recall = (
            hits
            /
            len(relevant_movies)
        )

        return recall


    # ========================================================
    # 16. EVALUATE ONE RECOMMENDATION MODEL
    # ========================================================

    def evaluate_recommendation_model(
        self,
        model_name,
        k=10,
        max_users=100
    ):

        print("\n")
        print("=" * 65)
        print(
            f"EVALUATING {model_name.upper()}"
        )
        print("=" * 65)

        # ----------------------------------------------------
        # Find users having at least one relevant
        # movie in the test set
        # ----------------------------------------------------

        eligible_users = []

        unique_users = (
            self.test_ratings[
                "userId"
            ].unique()
        )

        for user_id in unique_users:

            relevant_movies = (
                self.get_test_relevant_movies(
                    user_id
                )
            )

            if len(relevant_movies) > 0:

                eligible_users.append(
                    user_id
                )

        # ----------------------------------------------------
        # Limit number of users
        # ----------------------------------------------------

        if max_users is not None:

            eligible_users = (
                eligible_users[:max_users]
            )

        print(
            "Users selected:",
            len(eligible_users)
        )

        # ----------------------------------------------------
        # Metric lists
        # ----------------------------------------------------

        precisions = []
        recalls = []

        recommended_movies_all = set()

        # ----------------------------------------------------
        # Evaluate each user
        # ----------------------------------------------------

        for count, user_id in enumerate(
            eligible_users,
            start=1
        ):

            # ------------------------------------------------
            # Relevant movies
            # ------------------------------------------------

            relevant_movies = (
                self.get_test_relevant_movies(
                    user_id
                )
            )

            # ------------------------------------------------
            # Generate recommendations
            # ------------------------------------------------

            if model_name == "Collaborative":

                recommendations = (
                    self.collaborative_recommend(
                        user_id=user_id,
                        k=k
                    )
                )

            elif model_name == "Content-Based":

                recommendations = (
                    self.content_recommend(
                        user_id=user_id,
                        k=k
                    )
                )

            elif model_name == "Hybrid":

                recommendations = (
                    self.hybrid_recommend(
                        user_id=user_id,
                        k=k,
                        content_weight=0.4,
                        collaborative_weight=0.6
                    )
                )

            else:

                raise ValueError(
                    "Invalid model name."
                )

            # ------------------------------------------------
            # Get movie IDs
            # ------------------------------------------------

            recommended_ids = (
                recommendations[
                    "movieId"
                ].tolist()
            )

            # ------------------------------------------------
            # Precision
            # ------------------------------------------------

            precision = (
                self.precision_at_k(
                    recommended_ids,
                    relevant_movies,
                    k
                )
            )

            # ------------------------------------------------
            # Recall
            # ------------------------------------------------

            recall = (
                self.recall_at_k(
                    recommended_ids,
                    relevant_movies,
                    k
                )
            )

            precisions.append(
                precision
            )

            recalls.append(
                recall
            )

            recommended_movies_all.update(
                recommended_ids
            )

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if count % 10 == 0:

                print(
                    f"Evaluated {count} users..."
                )

        # ----------------------------------------------------
        # Handle no users
        # ----------------------------------------------------

        if len(precisions) == 0:

            return {
                "Model": model_name,
                f"Precision@{k}": 0.0,
                f"Recall@{k}": 0.0,
                "Coverage": 0.0
            }

        # ----------------------------------------------------
        # Average precision
        # ----------------------------------------------------

        precision_avg = np.mean(
            precisions
        )

        # ----------------------------------------------------
        # Average recall
        # ----------------------------------------------------

        recall_avg = np.mean(
            recalls
        )

        # ----------------------------------------------------
        # Catalog coverage
        # ----------------------------------------------------

        coverage = (
            len(recommended_movies_all)
            /
            len(self.movies)
        )

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        print("\n")
        print(
            f"Model: {model_name}"
        )

        print(
            f"Precision@{k}: "
            f"{precision_avg:.4f}"
        )

        print(
            f"Recall@{k}: "
            f"{recall_avg:.4f}"
        )

        print(
            f"Coverage: "
            f"{coverage:.4f}"
        )

        # ----------------------------------------------------
        # Return
        # ----------------------------------------------------

        return {
            "Model": model_name,
            f"Precision@{k}": precision_avg,
            f"Recall@{k}": recall_avg,
            "Coverage": coverage
        }


    # ========================================================
    # 17. EVALUATE ALL MODELS
    # ========================================================

    def evaluate_all_models(
        self,
        k=10,
        max_users=100
    ):

        print("\n")
        print("=" * 70)
        print("STARTING COMPLETE MODEL EVALUATION")
        print("=" * 70)

        # ----------------------------------------------------
        # RMSE / MAE for SVD
        # ----------------------------------------------------

        rmse, mae = (
            self.evaluate_svd_error()
        )

        # ----------------------------------------------------
        # Collaborative
        # ----------------------------------------------------

        collaborative_results = (
            self.evaluate_recommendation_model(
                model_name="Collaborative",
                k=k,
                max_users=max_users
            )
        )

        # ----------------------------------------------------
        # Content-Based
        # ----------------------------------------------------

        content_results = (
            self.evaluate_recommendation_model(
                model_name="Content-Based",
                k=k,
                max_users=max_users
            )
        )

        # ----------------------------------------------------
        # Hybrid
        # ----------------------------------------------------

        hybrid_results = (
            self.evaluate_recommendation_model(
                model_name="Hybrid",
                k=k,
                max_users=max_users
            )
        )

        # ----------------------------------------------------
        # Add RMSE / MAE
        # ----------------------------------------------------

        collaborative_results["RMSE"] = (
            rmse
        )

        collaborative_results["MAE"] = (
            mae
        )

        # RMSE/MAE are not directly calculated
        # for content-based ranking in this phase.

        content_results["RMSE"] = np.nan
        content_results["MAE"] = np.nan

        # The hybrid system is evaluated using
        # ranking metrics.

        hybrid_results["RMSE"] = np.nan
        hybrid_results["MAE"] = np.nan

        # ----------------------------------------------------
        # Create final DataFrame
        # ----------------------------------------------------

        results = pd.DataFrame(
            [
                collaborative_results,
                content_results,
                hybrid_results
            ]
        )

        # ----------------------------------------------------
        # Arrange columns
        # ----------------------------------------------------

        results = results[
            [
                "Model",
                "RMSE",
                "MAE",
                f"Precision@{k}",
                f"Recall@{k}",
                "Coverage"
            ]
        ]

        return results


# ============================================================
# 18. MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 70)
    print("PHASE 6 - MODEL EVALUATION")
    print("MOVIE RECOMMENDATION SYSTEM")
    print("=" * 70)


    # ========================================================
    # CREATE EVALUATOR
    # ========================================================

    evaluator = RecommendationEvaluator(
        movies_path="data/movies.csv",
        ratings_path="data/ratings.csv",

        # SVD parameters
        n_factors=100,
        n_epochs=20,
        lr_all=0.005,
        reg_all=0.02,

        random_state=42
    )


    # ========================================================
    # EVALUATE ALL MODELS
    # ========================================================

    results = (
        evaluator.evaluate_all_models(
            k=10,

            # Start with 100 users.
            # After successful testing, you can
            # increase this to 668.
            max_users=100
        )
    )


    # ========================================================
    # DISPLAY FINAL RESULTS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("FINAL MODEL COMPARISON")
    print("=" * 70)

    print(
        results.to_string(
            index=False
        )
    )


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    output_file = (
        "evaluation_results.csv"
    )

    results.to_csv(
        output_file,
        index=False
    )

    print("\n")
    print(
        "Evaluation results saved to:"
    )

    print(
        output_file
    )


    # ========================================================
    # FIND BEST MODEL
    # ========================================================

    precision_column = (
        "Precision@10"
    )

    best_model_index = (
        results[
            precision_column
        ].idxmax()
    )

    best_model = (
        results.loc[
            best_model_index,
            "Model"
        ]
    )

    best_precision = (
        results.loc[
            best_model_index,
            precision_column
        ]
    )

    print("\n")
    print("=" * 70)
    print("BEST MODEL")
    print("=" * 70)

    print(
        f"Best model based on "
        f"Precision@10: {best_model}"
    )

    print(
        f"Precision@10: "
        f"{best_precision:.4f}"
    )


    # ========================================================
    # COMPLETION
    # ========================================================

    print("\n")
    print("=" * 70)
    print("PHASE 6 COMPLETED SUCCESSFULLY")
    print("=" * 70)