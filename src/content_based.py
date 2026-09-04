import importlib


pd = importlib.import_module("pandas")
TfidfVectorizer = importlib.import_module("sklearn.feature_extraction.text").TfidfVectorizer
cosine_similarity = importlib.import_module("sklearn.metrics.pairwise").cosine_similarity


class ContentBasedRecommender:

    def __init__(self, movies_path):

        # --------------------------------------------------
        # 1. Load movie dataset
        # --------------------------------------------------
        self.movies = pd.read_csv(movies_path)

        # --------------------------------------------------
        # 2. Handle missing values
        # --------------------------------------------------
        self.movies["genres"] = (
            self.movies["genres"]
            .fillna("")
        )

        self.movies["title"] = (
            self.movies["title"]
            .fillna("")
        )

        # --------------------------------------------------
        # 3. Convert genre separator "|" into spaces
        # --------------------------------------------------
        self.movies["genres"] = (
            self.movies["genres"]
            .str.replace("|", " ", regex=False)
        )

        # --------------------------------------------------
        # 4. Create TF-IDF Vectorizer
        # --------------------------------------------------
        self.vectorizer = TfidfVectorizer(
            stop_words="english"
        )

        # --------------------------------------------------
        # 5. Convert movie genres into TF-IDF vectors
        # --------------------------------------------------
        self.tfidf_matrix = (
            self.vectorizer.fit_transform(
                self.movies["genres"]
            )
        )

        # --------------------------------------------------
        # 6. Calculate Cosine Similarity
        # --------------------------------------------------
        self.cosine_similarity = cosine_similarity(
            self.tfidf_matrix
        )

        # --------------------------------------------------
        # 7. Create movie title → index mapping
        # --------------------------------------------------
        self.movie_indices = pd.Series(
            self.movies.index,
            index=self.movies["title"]
        ).drop_duplicates()


    # ======================================================
    # SEARCH MOVIES
    # ======================================================

    def search_movies(self, query, limit=10):

        """
        Search movies by title.

        Parameters:
            query  : movie title or partial title
            limit  : maximum number of results

        Returns:
            DataFrame containing matching movies
        """

        if not query:
            return pd.DataFrame(
                columns=[
                    "movieId",
                    "title",
                    "genres"
                ]
            )

        results = self.movies[
            self.movies["title"]
            .str.contains(
                query,
                case=False,
                na=False
            )
        ]

        return results[
            [
                "movieId",
                "title",
                "genres"
            ]
        ].head(limit)


    # ======================================================
    # RECOMMEND MOVIES
    # ======================================================

    def recommend(self, movie_title, n=10):

        """
        Recommend movies similar to the selected movie.

        Parameters:
            movie_title : exact movie title
            n           : number of recommendations

        Returns:
            DataFrame containing recommended movies
        """

        # --------------------------------------------------
        # Check whether movie exists
        # --------------------------------------------------
        if movie_title not in self.movie_indices:

            print(
                f"Movie '{movie_title}' "
                "was not found in the dataset."
            )

            return pd.DataFrame(
                columns=[
                    "movieId",
                    "title",
                    "genres",
                    "similarity"
                ]
            )

        # --------------------------------------------------
        # Get index of selected movie
        # --------------------------------------------------
        movie_index = self.movie_indices[
            movie_title
        ]

        # --------------------------------------------------
        # Get similarity scores
        # --------------------------------------------------
        similarity_scores = list(
            enumerate(
                self.cosine_similarity[
                    movie_index
                ]
            )
        )

        # --------------------------------------------------
        # Sort movies by similarity
        # --------------------------------------------------
        similarity_scores = sorted(
            similarity_scores,
            key=lambda x: x[1],
            reverse=True
        )

        # --------------------------------------------------
        # Remove the selected movie itself
        # --------------------------------------------------
        similarity_scores = (
            similarity_scores[1:n + 1]
        )

        # --------------------------------------------------
        # Extract movie indices
        # --------------------------------------------------
        movie_indices = [
            index
            for index, score
            in similarity_scores
        ]

        # --------------------------------------------------
        # Extract similarity scores
        # --------------------------------------------------
        scores = [
            score
            for index, score
            in similarity_scores
        ]

        # --------------------------------------------------
        # Create recommendation DataFrame
        # --------------------------------------------------
        recommendations = (
            self.movies.iloc[
                movie_indices
            ][
                [
                    "movieId",
                    "title",
                    "genres"
                ]
            ].copy()
        )

        # --------------------------------------------------
        # Add similarity score
        # --------------------------------------------------
        recommendations[
            "similarity"
        ] = scores

        # --------------------------------------------------
        # Round similarity score
        # --------------------------------------------------
        recommendations[
            "similarity"
        ] = recommendations[
            "similarity"
        ].round(4)

        # --------------------------------------------------
        # Reset index
        # --------------------------------------------------
        recommendations = (
            recommendations
            .reset_index(drop=True)
        )

        return recommendations


# ==========================================================
# TEST THE MODEL
# ==========================================================

if __name__ == "__main__":

    print("=" * 70)
    print("CONTENT-BASED MOVIE RECOMMENDATION SYSTEM")
    print("=" * 70)

    # ------------------------------------------------------
    # Load model
    # ------------------------------------------------------

    model = ContentBasedRecommender(
        "data/movies.csv"
    )

    print("\nModel loaded successfully!")

    print(
        "\nNumber of movies:",
        len(model.movies)
    )

    print(
        "TF-IDF matrix shape:",
        model.tfidf_matrix.shape
    )

    print(
        "Similarity matrix shape:",
        model.cosine_similarity.shape
    )

    # ------------------------------------------------------
    # Search example
    # ------------------------------------------------------

    print("\n" + "=" * 70)
    print("MOVIE SEARCH")
    print("=" * 70)

    search_result = model.search_movies(
        "Toy",
        limit=10
    )

    print(
        search_result.to_string(
            index=False
        )
    )

    # ------------------------------------------------------
    # Recommendation example
    # ------------------------------------------------------

    movie_title = "Toy Story (1995)"

    print("\n" + "=" * 70)
    print(
        f"RECOMMENDATIONS FOR: {movie_title}"
    )
    print("=" * 70)

    recommendations = model.recommend(
        movie_title,
        n=10
    )

    if len(recommendations) > 0:

        print(
            recommendations.to_string(
                index=False
            )
        )

    else:

        print(
            "No recommendations found."
        )