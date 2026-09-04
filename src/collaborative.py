import pandas as pd
import numpy as np

from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise import accuracy
from surprise.model_selection import train_test_split





class CollaborativeRecommender:

    def __init__(
        self,
        ratings_path="data/ratings.csv",
        n_factors=100,
        n_epochs=20,
        lr_all=0.005,
        reg_all=0.02,
        random_state=42
    ):

        self.ratings_path = ratings_path

        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr_all = lr_all
        self.reg_all = reg_all
        self.random_state = random_state

        self.ratings = None
        self.data = None
        self.trainset = None
        self.testset = None
        self.model = None

        self.rmse = None
        self.mae = None

        self.load_data()
        self.prepare_data()


   

    def load_data(self):

        print("\nLoading ratings dataset...")

        self.ratings = pd.read_csv(
            self.ratings_path
        )

        # Keep required columns
        self.ratings = self.ratings[
            ["userId", "movieId", "rating"]
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

        print("Ratings dataset loaded.")

        print(
            "Number of ratings:",
            len(self.ratings)
        )

        print(
            "Number of users:",
            self.ratings["userId"].nunique()
        )

        print(
            "Number of movies:",
            self.ratings["movieId"].nunique()
        )


    

    def prepare_data(self):

        min_rating = self.ratings["rating"].min()
        max_rating = self.ratings["rating"].max()

        print(
            "Rating scale:",
            min_rating,
            "to",
            max_rating
        )

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


   

    def split_data(self):

        print("\nSplitting data...")

        # IMPORTANT:
        # train_test_split is imported from
        # surprise.model_selection

        self.trainset, self.testset = train_test_split(
            self.data,
            test_size=0.20,
            random_state=self.random_state
        )

        print(
            "Training ratings:",
            self.trainset.n_ratings
        )

        print(
            "Testing ratings:",
            len(self.testset)
        )


    

    def create_model(self):

        print("\nCreating SVD model...")

        self.model = SVD(
            n_factors=self.n_factors,
            n_epochs=self.n_epochs,
            lr_all=self.lr_all,
            reg_all=self.reg_all,
            random_state=self.random_state
        )

        print("SVD model created.")


    

    def train(self):

        # Split data first
        if self.trainset is None:
            self.split_data()

        # Create model
        if self.model is None:
            self.create_model()

        print("\nTraining SVD model...")

        self.model.fit(
            self.trainset
        )

        print(
            "SVD model training completed."
        )


    

    def evaluate(self):

        if self.model is None:
            raise ValueError(
                "Model has not been trained."
            )

        if self.testset is None:
            raise ValueError(
                "Test dataset is not available."
            )

        print("\nEvaluating model...")

        predictions = self.model.test(
            self.testset
        )

        print("\nRMSE:")

        self.rmse = accuracy.rmse(
            predictions,
            verbose=True
        )

        print("\nMAE:")

        self.mae = accuracy.mae(
            predictions,
            verbose=True
        )

        print("\n")
        print("=" * 60)
        print("MODEL EVALUATION")
        print("=" * 60)

        print(
            f"RMSE: {self.rmse:.4f}"
        )

        print(
            f"MAE : {self.mae:.4f}"
        )

        print("=" * 60)

        return self.rmse, self.mae


    

    def predict_rating(
        self,
        user_id,
        movie_id
    ):

        if self.model is None:
            raise ValueError(
                "Model has not been trained."
            )

        prediction = self.model.predict(
            uid=user_id,
            iid=movie_id
        )

        return prediction.est


    

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


   

    def recommend(
        self,
        user_id,
        movies_path="data/movies.csv",
        n=10
    ):

        if self.model is None:
            raise ValueError(
                "Model has not been trained."
            )

       

        movies = pd.read_csv(
            movies_path
        )

        movies = movies[
            [
                "movieId",
                "title",
                "genres"
            ]
        ].copy()

        movies["movieId"] = (
            movies["movieId"].astype(int)
        )

        
        rated_movies = (
            self.get_user_rated_movies(user_id)
        )

       

        candidates = movies[
            ~movies["movieId"].isin(
                rated_movies
            )
        ].copy()

        print(
            f"\nMovies already rated by "
            f"user {user_id}: "
            f"{len(rated_movies)}"
        )

        print(
            f"Candidate movies: "
            f"{len(candidates)}"
        )

        

        predicted_ratings = []

        for movie_id in candidates[
            "movieId"
        ]:

            prediction = self.model.predict(
                user_id,
                movie_id
            )

            predicted_ratings.append(
                prediction.est
            )

        candidates[
            "predicted_rating"
        ] = predicted_ratings

        

        recommendations = (
            candidates
            .sort_values(
                by="predicted_rating",
                ascending=False
            )
            .head(n)
            .copy()
        )

       

        recommendations[
            "predicted_rating"
        ] = recommendations[
            "predicted_rating"
        ].round(2)

        

        recommendations.insert(
            0,
            "rank",
            range(
                1,
                len(recommendations) + 1
            )
        )

        recommendations.reset_index(
            drop=True,
            inplace=True
        )

        return recommendations




if __name__ == "__main__":

    print("\n")
    print("=" * 60)
    print("COLLABORATIVE FILTERING MOVIE RECOMMENDER")
    print("=" * 60)


 

    recommender = CollaborativeRecommender(
        ratings_path="data/ratings.csv",
        n_factors=100,
        n_epochs=20,
        lr_all=0.005,
        reg_all=0.02,
        random_state=42
    )


   

    recommender.train()


  

    rmse, mae = recommender.evaluate()




    test_user_id = 1
    test_movie_id = 1

    predicted_rating = (
        recommender.predict_rating(
            test_user_id,
            test_movie_id
        )
    )

    print("\n")
    print("=" * 60)
    print("RATING PREDICTION TEST")
    print("=" * 60)

    print(
        f"User ID: {test_user_id}"
    )

    print(
        f"Movie ID: {test_movie_id}"
    )

    print(
        f"Predicted Rating: "
        f"{predicted_rating:.2f}"
    )


   

    recommendations = (
        recommender.recommend(
            user_id=test_user_id,
            movies_path="data/movies.csv",
            n=10
        )
    )


   

    print("\n")
    print("=" * 60)
    print(
        f"TOP 10 MOVIE RECOMMENDATIONS "
        f"FOR USER {test_user_id}"
    )
    print("=" * 60)

    print(
        recommendations.to_string(
            index=False
        )
    )



    print("\n")
    print("=" * 60)
    print("PHASE 4 COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(
        f"RMSE: {rmse:.4f}"
    )

    print(
        f"MAE : {mae:.4f}"
    )