from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SRC_DIR = ROOT / "src"
APP_FILE = ROOT / "app.py"

MOVIES_FILE = DATA_DIR / "movies.csv"
RATINGS_FILE = DATA_DIR / "ratings.csv"

REPORT_FILE = ROOT / "phase8_report.txt"

PASS_COUNT = 0
FAIL_COUNT = 0
WARN_COUNT = 0
REPORT_LINES = []


# ============================================================
# REPORT HELPERS
# ============================================================

def log(message=""):
    print(message)
    REPORT_LINES.append(str(message))


def check(condition, name, success_message="", failure_message=""):
    global PASS_COUNT, FAIL_COUNT

    if condition:
        PASS_COUNT += 1
        message = f"PASS  | {name}"
        if success_message:
            message += f" — {success_message}"
        log(message)
        return True

    FAIL_COUNT += 1
    message = f"FAIL  | {name}"
    if failure_message:
        message += f" — {failure_message}"
    log(message)
    return False


def warning(name, message):
    global WARN_COUNT
    WARN_COUNT += 1
    log(f"WARN  | {name} — {message}")


def section(title):
    log("")
    log("=" * 72)
    log(title)
    log("=" * 72)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    section("1. DATA LOADING")

    check(
        MOVIES_FILE.exists(),
        "movies.csv exists",
        failure_message=str(MOVIES_FILE),
    )

    check(
        RATINGS_FILE.exists(),
        "ratings.csv exists",
        failure_message=str(RATINGS_FILE),
    )

    if not MOVIES_FILE.exists() or not RATINGS_FILE.exists():
        raise FileNotFoundError("Required dataset files were not found.")

    movies = pd.read_csv(MOVIES_FILE)
    ratings = pd.read_csv(RATINGS_FILE)

    log(f"Movies shape : {movies.shape}")
    log(f"Ratings shape: {ratings.shape}")

    return movies, ratings


# ============================================================
# DATASET TESTS
# ============================================================

def test_dataset(movies, ratings):
    section("2. DATASET INTEGRITY TESTS")

    required_movie_cols = {"movieId", "title", "genres"}
    required_rating_cols = {"userId", "movieId", "rating", "timestamp"}

    check(
        required_movie_cols.issubset(movies.columns),
        "Movie columns",
        success_message=str(sorted(required_movie_cols)),
        failure_message=f"Found {list(movies.columns)}",
    )

    check(
        required_rating_cols.issubset(ratings.columns),
        "Rating columns",
        success_message=str(sorted(required_rating_cols)),
        failure_message=f"Found {list(ratings.columns)}",
    )

    check(
        movies["movieId"].notna().all(),
        "Movie IDs are not missing",
    )

    check(
        ratings[["userId", "movieId", "rating"]].notna().all().all(),
        "Required rating values are not missing",
    )

    check(
        movies["movieId"].is_unique,
        "Movie IDs are unique",
        failure_message="Duplicate movie IDs detected.",
    )

    check(
        ratings["rating"].between(
            ratings["rating"].min(),
            ratings["rating"].max()
        ).all(),
        "Ratings are inside dataset rating scale",
    )

    log(f"Users         : {ratings['userId'].nunique():,}")
    log(f"Movies        : {movies['movieId'].nunique():,}")
    log(f"Ratings       : {len(ratings):,}")
    log(f"Rating range  : {ratings['rating'].min()} - {ratings['rating'].max()}")
    log(f"Average rating: {ratings['rating'].mean():.3f}")

    duplicate_rows = ratings.duplicated().sum()
    if duplicate_rows:
        warning("Duplicate rating rows", f"{duplicate_rows:,} duplicate rows found.")
    else:
        check(True, "No duplicate rating rows")


# ============================================================
# CONTENT-BASED TEST
# ============================================================

def test_content_based(movies):
    section("3. CONTENT-BASED MODEL TESTS")

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    model_movies = movies.copy()
    model_movies["title"] = model_movies["title"].fillna("Unknown")
    model_movies["genres"] = (
        model_movies["genres"]
        .fillna("Unknown")
        .astype(str)
        .str.replace("|", " ", regex=False)
    )

    start = time.perf_counter()

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
    )

    tfidf_matrix = vectorizer.fit_transform(model_movies["genres"])
    similarity_matrix = cosine_similarity(tfidf_matrix)

    build_time = time.perf_counter() - start

    check(
        tfidf_matrix.shape[0] == len(model_movies),
        "TF-IDF row count",
    )

    check(
        similarity_matrix.shape == (len(model_movies), len(model_movies)),
        "Content similarity matrix shape",
    )

    log(f"TF-IDF matrix : {tfidf_matrix.shape}")
    log(f"Build time    : {build_time:.3f} seconds")

    # Search for Toy Story.
    query = "Toy Story"
    matches = model_movies[
        model_movies["title"].str.contains(query, case=False, na=False)
    ]

    check(
        len(matches) > 0,
        "Movie search for 'Toy Story'",
        success_message=f"{len(matches)} match(es)",
    )

    if len(matches) == 0:
        return

    selected_index = matches.index[0]

    scores = similarity_matrix[selected_index].copy()
    scores[selected_index] = -1

    top_indices = np.argsort(scores)[::-1][:10]

    recommendations = model_movies.iloc[top_indices].copy()
    recommendations["similarity"] = scores[top_indices]

    check(
        len(recommendations) == 10,
        "Content-based returns 10 recommendations",
    )

    check(
        not recommendations["movieId"].duplicated().any(),
        "Content-based recommendations have unique movies",
    )

    check(
        (recommendations["similarity"] >= 0).all()
        and (recommendations["similarity"] <= 1).all(),
        "Content similarity is in [0, 1]",
    )

    check(
        selected_index not in recommendations.index,
        "Selected movie is excluded",
    )

    log("")
    log("Top content-based recommendations:")
    for _, row in recommendations.head(5).iterrows():
        log(
            f"  {row['title']} | "
            f"similarity={row['similarity']:.3f}"
        )

    # Optimization note.
    warning(
        "Content model memory",
        "A full cosine similarity matrix is O(N²). "
        "For larger datasets, use cosine_similarity(tfidf_matrix[-query]) "
        "against the matrix or NearestNeighbors instead of storing all pairs."
    )


# ============================================================
# COLLABORATIVE FILTERING TEST
# ============================================================

def test_collaborative(ratings, movies):
    section("4. COLLABORATIVE FILTERING TESTS")

    try:
        from surprise import Dataset, Reader, SVD
        from surprise.model_selection import train_test_split
        from surprise import accuracy
    except Exception as exc:
        check(
            False,
            "Surprise/scikit-surprise import",
            failure_message=str(exc),
        )
        return None

    reader = Reader(
        rating_scale=(
            float(ratings["rating"].min()),
            float(ratings["rating"].max()),
        )
    )

    surprise_data = Dataset.load_from_df(
        ratings[["userId", "movieId", "rating"]],
        reader,
    )

    trainset, testset = train_test_split(
        surprise_data,
        test_size=0.20,
        random_state=42,
    )

    model = SVD(
        n_factors=100,
        n_epochs=20,
        lr_all=0.005,
        reg_all=0.02,
        random_state=42,
    )

    start = time.perf_counter()
    model.fit(trainset)
    train_time = time.perf_counter() - start

    predictions = model.test(testset)

    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)

    log(f"Training time: {train_time:.3f} seconds")
    log(f"RMSE         : {rmse:.4f}")
    log(f"MAE          : {mae:.4f}")

    check(
        np.isfinite(rmse) and rmse >= 0,
        "RMSE is valid",
    )

    check(
        np.isfinite(mae) and mae >= 0,
        "MAE is valid",
    )

    # Test a known user/movie pair.
    user_id = int(ratings["userId"].iloc[0])
    movie_id = int(ratings["movieId"].iloc[0])

    prediction = model.predict(user_id, movie_id)

    log(
        f"Sample prediction: user={user_id}, "
        f"movie={movie_id}, estimate={prediction.est:.3f}"
    )

    check(
        np.isfinite(prediction.est),
        "SVD prediction is numeric",
    )

    # Generate top-N recommendations for one user.
    rated_movies = set(
        ratings.loc[
            ratings["userId"] == user_id,
            "movieId"
        ].astype(int)
    )

    candidates = movies[
        ~movies["movieId"].isin(rated_movies)
    ].copy()

    # Use a moderate candidate sample for benchmarking.
    candidates = candidates.head(1000)

    start = time.perf_counter()

    candidate_scores = [
        model.predict(user_id, int(movie_id)).est
        for movie_id in candidates["movieId"]
    ]

    prediction_time = time.perf_counter() - start

    candidates["predicted_rating"] = candidate_scores
    recommendations = candidates.sort_values(
        "predicted_rating",
        ascending=False
    ).head(10)

    log(
        f"Prediction benchmark: {len(candidates)} candidates "
        f"in {prediction_time:.3f} seconds"
    )

    check(
        len(recommendations) == 10,
        "Collaborative returns 10 recommendations",
    )

    check(
        not recommendations["movieId"].duplicated().any(),
        "Collaborative recommendations are unique",
    )

    check(
        np.isfinite(recommendations["predicted_rating"]).all(),
        "Collaborative scores are numeric",
    )

    check(
        not recommendations["movieId"].isin(rated_movies).any(),
        "Already-rated movies are excluded",
    )

    log("")
    log("Top collaborative recommendations:")
    for _, row in recommendations.head(5).iterrows():
        log(
            f"  {row['title']} | "
            f"predicted={row['predicted_rating']:.3f}"
        )

    warning(
        "Collaborative runtime",
        "Do not train SVD on every Streamlit rerun. "
        "Cache the trained model with @st.cache_resource."
    )

    return model


# ============================================================
# HYBRID MODEL TEST
# ============================================================

def test_hybrid(movies, ratings, collaborative_model):
    section("5. HYBRID MODEL TESTS")

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    model_movies = movies.copy()
    model_movies["title"] = model_movies["title"].fillna("Unknown")
    model_movies["genres"] = (
        model_movies["genres"]
        .fillna("Unknown")
        .astype(str)
        .str.replace("|", " ", regex=False)
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
    )

    tfidf = vectorizer.fit_transform(model_movies["genres"])

    # Choose a real movie from the dataset.
    selected_movie = "Toy Story (1995)"
    matching = model_movies.index[
        model_movies["title"].str.lower() == selected_movie.lower()
    ]

    if len(matching) == 0:
        matching = model_movies.index[
            model_movies["title"].str.contains(
                "Toy Story",
                case=False,
                na=False,
            )
        ]

    if len(matching) == 0:
        warning(
            "Hybrid seed movie",
            "Toy Story was not found; hybrid test skipped."
        )
        return

    movie_index = matching[0]
    user_id = int(ratings["userId"].iloc[0])

    content_scores = cosine_similarity(
        tfidf[movie_index],
        tfidf
    ).flatten()

    candidates = model_movies.copy()
    candidates["content_score"] = content_scores

    rated_movies = set(
        ratings.loc[
            ratings["userId"] == user_id,
            "movieId"
        ].astype(int)
    )

    candidates = candidates[
        ~candidates["movieId"].isin(rated_movies)
    ].copy()

    # Exclude selected seed movie.
    candidates = candidates[
        candidates["movieId"] != model_movies.loc[movie_index, "movieId"]
    ].copy()

    start = time.perf_counter()

    candidates["predicted_rating"] = [
        collaborative_model.predict(
            user_id,
            int(movie_id)
        ).est
        for movie_id in candidates["movieId"]
    ]

    # Normalize both components.
    def minmax(series):
        low = float(series.min())
        high = float(series.max())

        if high == low:
            return pd.Series(
                np.zeros(len(series)),
                index=series.index
            )

        return (series - low) / (high - low)

    candidates["content_norm"] = minmax(
        candidates["content_score"]
    )

    candidates["rating_norm"] = minmax(
        candidates["predicted_rating"]
    )

    candidates["hybrid_score"] = (
        0.50 * candidates["content_norm"]
        + 0.50 * candidates["rating_norm"]
    )

    recommendations = candidates.sort_values(
        "hybrid_score",
        ascending=False
    ).head(10)

    hybrid_time = time.perf_counter() - start

    log(f"Hybrid recommendation time: {hybrid_time:.3f} seconds")

    check(
        len(recommendations) == 10,
        "Hybrid returns 10 recommendations",
    )

    check(
        not recommendations["movieId"].duplicated().any(),
        "Hybrid recommendations are unique",
    )

    check(
        recommendations["hybrid_score"].between(0, 1).all(),
        "Hybrid score is in [0, 1]",
    )

    check(
        not recommendations["movieId"].isin(rated_movies).any(),
        "Hybrid excludes already-rated movies",
    )

    log("")
    log("Top hybrid recommendations:")
    for _, row in recommendations.head(5).iterrows():
        log(
            f"  {row['title']} | "
            f"content={row['content_score']:.3f} | "
            f"rating={row['predicted_rating']:.3f} | "
            f"hybrid={row['hybrid_score']:.3f}"
        )


# ============================================================
# APP CODE QUALITY TEST
# ============================================================

def test_app_file():
    section("6. STREAMLIT APP CODE QUALITY")

    if not APP_FILE.exists():
        warning(
            "app.py",
            "app.py was not found at the project root."
        )
        return

    text = APP_FILE.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    check(
        "import streamlit as st" in text,
        "Streamlit import",
    )

    check(
        "st.set_page_config" in text,
        "Streamlit page configuration",
    )

    check(
        "st.cache_resource" in text,
        "Model caching is present",
        failure_message="Add @st.cache_resource around model training.",
    )

    # The previous UI bug displayed raw HTML because the movie card
    # markup was being rendered incorrectly. We only flag likely card
    # markup in the recommendation function, not CSS.
    if "def render_movie_card" in text:
        function_text = text[text.index("def render_movie_card"):]

        if "unsafe_allow_html=True" in function_text:
            warning(
                "Movie card HTML",
                "render_movie_card contains unsafe_allow_html=True. "
                "Use native Streamlit components for the card body."
            )
        else:
            check(
                True,
                "Movie card does not explicitly enable unsafe HTML",
            )


# ============================================================
# OPTIMIZATION CHECKLIST
# ============================================================

def optimization_checklist():
    section("7. PHASE 8 OPTIMIZATION CHECKLIST")

    items = [
        "Cache datasets with @st.cache_data.",
        "Cache trained models with @st.cache_resource.",
        "Do not train SVD on every button click.",
        "Avoid building a full N×N content similarity matrix for larger datasets.",
        "Exclude already-rated movies before collaborative ranking.",
        "Limit candidate movies before expensive prediction operations when appropriate.",
        "Use deterministic random_state values for reproducible experiments.",
        "Keep recommendation count configurable.",
        "Validate recommendation outputs before rendering them.",
        "Keep the UI separate from model-training code.",
    ]

    for index, item in enumerate(items, start=1):
        log(f"{index:02d}. {item}")


# ============================================================
# MAIN
# ============================================================

def main():
    section("PHASE 8 — TESTING & OPTIMIZATION")
    log("Movie Recommendation System")
    log(f"Project root: {ROOT}")

    try:
        movies, ratings = load_data()
        test_dataset(movies, ratings)
        test_content_based(movies)

        collaborative_model = test_collaborative(
            ratings,
            movies,
        )

        if collaborative_model is not None:
            test_hybrid(
                movies,
                ratings,
                collaborative_model,
            )

        test_app_file()
        optimization_checklist()

    except Exception as exc:
        check(
            False,
            "Phase 8 execution",
            failure_message=str(exc),
        )
        log("")
        log("Detailed traceback:")
        log(traceback.format_exc())

    section("PHASE 8 FINAL RESULT")

    log(f"Passed: {PASS_COUNT}")
    log(f"Failed: {FAIL_COUNT}")
    log(f"Warnings: {WARN_COUNT}")

    if FAIL_COUNT == 0:
        log("")
        log("STATUS: PHASE 8 TESTING PASSED")
    else:
        log("")
        log("STATUS: PHASE 8 NEEDS FIXES")

    REPORT_FILE.write_text(
        "\n".join(REPORT_LINES),
        encoding="utf-8",
    )

    log(f"Report saved to: {REPORT_FILE}")


if __name__ == "__main__":
    main()
