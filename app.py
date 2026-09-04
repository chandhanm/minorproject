
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split


# ============================================================
# PHASE 7D - FINAL POLISHED STREAMLIT UI
# Movie Recommendation System
# Dataset: MovieLens / GroupLens
# ============================================================

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
        /* ---------- Global ---------- */
        .stApp {
            background:
                radial-gradient(circle at 10% 0%, rgba(110, 78, 160, 0.18), transparent 28%),
                radial-gradient(circle at 90% 10%, rgba(255, 76, 87, 0.12), transparent 25%),
                #0b0d12;
        }

        .main .block-container {
            max-width: 1450px;
            padding-top: 2rem;
            padding-bottom: 4rem;
        }

        /* ---------- Sidebar ---------- */
        section[data-testid="stSidebar"] {
            background: #11141c;
            border-right: 1px solid rgba(255,255,255,0.08);
        }

        section[data-testid="stSidebar"] .block-container {
            padding-top: 2rem;
        }

        /* ---------- Header ---------- */
        .hero {
            padding: 2.2rem 2.4rem;
            border-radius: 28px;
            background:
                linear-gradient(135deg,
                    rgba(91, 61, 128, 0.55),
                    rgba(33, 38, 54, 0.88));
            border: 1px solid rgba(255,255,255,0.10);
            box-shadow: 0 20px 60px rgba(0,0,0,0.25);
            margin-bottom: 1.5rem;
        }

        .hero-badge {
            display: inline-block;
            padding: 0.35rem 0.75rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.10);
            color: #d8dbe7;
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            margin-bottom: 0.9rem;
        }

        .hero h1 {
            margin: 0;
            color: #ffffff;
            font-size: 3.1rem;
            line-height: 1.05;
            font-weight: 800;
        }

        .hero p {
            color: #c7cad5;
            font-size: 1.08rem;
            margin-top: 0.9rem;
            max-width: 820px;
        }

        /* ---------- Metric cards ---------- */
        .metric-card {
            background: rgba(25, 28, 38, 0.86);
            border: 1px solid rgba(255,255,255,0.09);
            border-radius: 20px;
            padding: 1.25rem;
            min-height: 125px;
            box-shadow: 0 10px 35px rgba(0,0,0,0.15);
        }

        .metric-icon {
            font-size: 1.35rem;
        }

        .metric-value {
            color: #ffffff;
            font-size: 1.85rem;
            font-weight: 800;
            margin-top: 0.35rem;
        }

        .metric-label {
            color: #aeb3c1;
            font-size: 0.88rem;
            margin-top: 0.15rem;
        }

        /* ---------- Section titles ---------- */
        .section-title {
            color: #ffffff;
            font-size: 1.65rem;
            font-weight: 800;
            margin: 1.8rem 0 0.9rem 0;
        }

        .section-subtitle {
            color: #9ea4b3;
            margin-top: -0.5rem;
            margin-bottom: 1.2rem;
        }

        /* ---------- Movie cards ---------- */
        .movie-card {
            background:
                linear-gradient(145deg, rgba(27,30,41,0.98), rgba(18,21,29,0.98));
            border: 1px solid rgba(255,255,255,0.09);
            border-radius: 22px;
            padding: 1.2rem;
            margin: 0.6rem 0;
            min-height: 185px;
            transition: transform 0.18s ease, border-color 0.18s ease;
            box-shadow: 0 12px 35px rgba(0,0,0,0.16);
        }

        .movie-card:hover {
            transform: translateY(-3px);
            border-color: rgba(255,255,255,0.20);
        }

        .movie-rank {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 36px;
            height: 36px;
            border-radius: 12px;
            background: rgba(255,255,255,0.08);
            color: #ffffff;
            font-weight: 800;
            margin-bottom: 0.7rem;
        }

        .movie-title {
            color: #ffffff;
            font-size: 1.15rem;
            font-weight: 800;
            line-height: 1.35;
            margin-bottom: 0.7rem;
        }

        .movie-meta {
            color: #aeb3c1;
            font-size: 0.91rem;
            line-height: 1.55;
            margin: 0.28rem 0;
        }

        .movie-score {
            display: inline-block;
            margin-top: 0.65rem;
            padding: 0.38rem 0.7rem;
            border-radius: 999px;
            background: rgba(255, 193, 7, 0.10);
            border: 1px solid rgba(255,193,7,0.18);
            color: #f5d36d;
            font-weight: 700;
            font-size: 0.86rem;
        }

        /* ---------- Info panel ---------- */
        .info-panel {
            background: rgba(20,23,31,0.92);
            border: 1px solid rgba(255,255,255,0.09);
            border-radius: 22px;
            padding: 1.4rem;
            margin: 0.8rem 0;
        }

        .info-panel h3 {
            color: #ffffff;
            margin-top: 0;
        }

        .info-panel p {
            color: #aeb3c1;
            line-height: 1.65;
        }

        /* ---------- Recommendation mode ---------- */
        .mode-pill {
            display: inline-block;
            padding: 0.35rem 0.75rem;
            border-radius: 999px;
            background: rgba(124, 92, 170, 0.18);
            border: 1px solid rgba(124, 92, 170, 0.32);
            color: #d9c8ef;
            font-size: 0.82rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }

        /* ---------- Footer ---------- */
        .footer {
            text-align: center;
            color: #747a88;
            padding: 2.5rem 0 0.5rem 0;
            font-size: 0.85rem;
        }

        /* ---------- Buttons ---------- */
        div.stButton > button {
            border-radius: 13px;
            font-weight: 700;
            min-height: 2.7rem;
        }

        /* ---------- Hide Streamlit branding ---------- */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_data():
    movies = pd.read_csv("data/movies.csv")
    ratings = pd.read_csv("data/ratings.csv")

    movies["title"] = movies["title"].fillna("Unknown Movie").astype(str)
    movies["genres"] = movies["genres"].fillna("Unknown").astype(str)

    ratings = ratings[["userId", "movieId", "rating"]].copy()
    ratings = ratings.dropna(subset=["userId", "movieId", "rating"])

    ratings["userId"] = ratings["userId"].astype(int)
    ratings["movieId"] = ratings["movieId"].astype(int)
    ratings["rating"] = ratings["rating"].astype(float)

    return movies, ratings


# ============================================================
# CONTENT MODEL
# ============================================================

@st.cache_resource
def build_content_model(movies):
    model_movies = movies.copy()
    genre_text = (
        model_movies["genres"]
        .fillna("Unknown")
        .astype(str)
        .str.replace("|", " ", regex=False)
    )

    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(genre_text)

    similarity = cosine_similarity(matrix)

    title_to_index = pd.Series(
        model_movies.index,
        index=model_movies["title"]
    ).drop_duplicates()

    return vectorizer, similarity, title_to_index


# ============================================================
# COLLABORATIVE MODEL
# ============================================================

@st.cache_resource
def build_collaborative_model(ratings):
    reader = Reader(
        rating_scale=(
            float(ratings["rating"].min()),
            float(ratings["rating"].max())
        )
    )

    data = Dataset.load_from_df(
        ratings[["userId", "movieId", "rating"]],
        reader
    )

    trainset, testset = train_test_split(
        data,
        test_size=0.20,
        random_state=42
    )

    model = SVD(
        n_factors=100,
        n_epochs=20,
        lr_all=0.005,
        reg_all=0.02,
        random_state=42
    )

    model.fit(trainset)

    return model, trainset, testset


# ============================================================
# METRICS / DATA HELPERS
# ============================================================

@st.cache_data
def movie_statistics(movies, ratings):
    stats = (
        ratings.groupby("movieId")
        .agg(
            average_rating=("rating", "mean"),
            rating_count=("rating", "count")
        )
        .reset_index()
    )

    return movies.merge(stats, on="movieId", how="left").fillna(
        {"average_rating": 0, "rating_count": 0}
    )


@st.cache_data
def user_history(user_id, ratings, movies):
    history = ratings[ratings["userId"] == user_id].copy()

    history = history.merge(
        movies[["movieId", "title", "genres"]],
        on="movieId",
        how="left"
    )

    return history.sort_values(
        ["rating", "title"],
        ascending=[False, True]
    )


# ============================================================
# RECOMMENDATION FUNCTIONS
# ============================================================

def content_recommendations(movie_title, n, movies, similarity, title_to_index):
    if movie_title not in title_to_index.index:
        return pd.DataFrame()

    idx = int(title_to_index[movie_title])

    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    rows = []

    for movie_index, score in scores:
        if movie_index == idx:
            continue

        row = movies.iloc[movie_index]

        rows.append(
            {
                "movieId": int(row["movieId"]),
                "title": row["title"],
                "genres": row["genres"],
                "score": float(score),
            }
        )

        if len(rows) >= n:
            break

    return pd.DataFrame(rows)


def collaborative_recommendations(
    user_id,
    n,
    movies,
    ratings,
    model
):
    rated_ids = set(
        ratings.loc[
            ratings["userId"] == user_id,
            "movieId"
        ].astype(int)
    )

    candidates = movies[
        ~movies["movieId"].isin(rated_ids)
    ].copy()

    if candidates.empty:
        return pd.DataFrame()

    # Predict ratings for all unseen movies.
    candidates["predicted_rating"] = candidates["movieId"].apply(
        lambda movie_id: float(
            model.predict(int(user_id), int(movie_id)).est
        )
    )

    candidates = candidates.sort_values(
        "predicted_rating",
        ascending=False
    ).head(n)

    return candidates[
        ["movieId", "title", "genres", "predicted_rating"]
    ].reset_index(drop=True)


def hybrid_recommendations(
    user_id,
    movie_title,
    n,
    movies,
    ratings,
    similarity,
    title_to_index,
    model,
):
    if movie_title not in title_to_index.index:
        return pd.DataFrame()

    selected_index = int(title_to_index[movie_title])

    rated_ids = set(
        ratings.loc[
            ratings["userId"] == user_id,
            "movieId"
        ].astype(int)
    )

    candidates = movies[
        ~movies["movieId"].isin(rated_ids)
    ].copy()

    if candidates.empty:
        return pd.DataFrame()

    # Content score against the selected movie.
    content_scores = similarity[selected_index]

    candidate_indices = candidates.index.to_numpy()

    candidates["content_score"] = [
        float(content_scores[i])
        for i in candidate_indices
    ]

    # Collaborative prediction.
    candidates["predicted_rating"] = candidates["movieId"].apply(
        lambda movie_id: float(
            model.predict(int(user_id), int(movie_id)).est
        )
    )

    # Normalize both components to 0-1.
    content_min = candidates["content_score"].min()
    content_max = candidates["content_score"].max()

    rating_min = candidates["predicted_rating"].min()
    rating_max = candidates["predicted_rating"].max()

    if content_max > content_min:
        candidates["content_norm"] = (
            (candidates["content_score"] - content_min)
            / (content_max - content_min)
        )
    else:
        candidates["content_norm"] = 0.0

    if rating_max > rating_min:
        candidates["rating_norm"] = (
            (candidates["predicted_rating"] - rating_min)
            / (rating_max - rating_min)
        )
    else:
        candidates["rating_norm"] = 0.0

    # 50% content + 50% collaborative.
    candidates["hybrid_score"] = (
        0.50 * candidates["content_norm"]
        + 0.50 * candidates["rating_norm"]
    )

    candidates = candidates.sort_values(
        "hybrid_score",
        ascending=False
    ).head(n)

    return candidates[
        [
            "movieId",
            "title",
            "genres",
            "predicted_rating",
            "content_score",
            "hybrid_score",
        ]
    ].reset_index(drop=True)


# ============================================================
# UI HELPERS
# ============================================================

def render_movie_card(
    rank,
    title,
    genres,
    score_label=None,
    score_value=None,
    extra_label=None,
    extra_value=None,
):
    """
    Native Streamlit movie card.

    IMPORTANT:
    Do not use raw HTML here. Using Streamlit-native components prevents
    <div>, <span>, etc. from appearing as visible text in the browser.
    """

    with st.container(border=True):
        # Rank + movie title
        st.markdown(f"### {rank}. 🎬 {title}")

        # Genre information
        st.write(f"🎭 **Genres:** {genres}")

        # Scores
        if score_label is not None and score_value is not None:
            if extra_label is not None and extra_value is not None:
                score_col, hybrid_col = st.columns(2)

                with score_col:
                    st.metric(score_label, score_value)

                with hybrid_col:
                    st.metric(extra_label, extra_value)
            else:
                st.metric(score_label, score_value)

def render_empty_state(title, message):
    st.markdown(
        f"""
        <div class="info-panel">
            <h3>{title}</h3>
            <p>{message}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# LOAD EVERYTHING
# ============================================================

try:
    movies, ratings = load_data()

    _, similarity_matrix, title_to_index = build_content_model(movies)

    collaborative_model, trainset, testset = build_collaborative_model(
        ratings
    )

    stats = movie_statistics(movies, ratings)

except Exception as e:
    st.error("The application could not load the recommendation models.")
    st.exception(e)
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="font-size:2.1rem;font-weight:800;color:white;">
            🎬 MovieMatch
        </div>
        <div style="color:#9ea4b3;margin-bottom:1.5rem;">
            Intelligent Movie Recommendation System
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 🎯 Recommendation Mode")

    mode = st.radio(
        "Choose a method",
        [
            "Hybrid",
            "Content-Based",
            "Collaborative",
        ],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("---")

    st.markdown("### 👤 Personalization")

    user_ids = sorted(ratings["userId"].unique().tolist())

    selected_user = st.selectbox(
        "Select User",
        user_ids,
        index=0,
    )

    recommendation_count = st.slider(
        "Number of recommendations",
        min_value=5,
        max_value=20,
        value=10,
        step=1,
    )

    st.markdown("---")

    st.markdown("### 📊 Dataset")

    st.markdown(
        f"""
        <div style="color:#b5bac7;line-height:2;">
            🎬 Movies: <strong style="color:white;">{len(movies):,}</strong><br>
            👤 Users: <strong style="color:white;">{ratings["userId"].nunique():,}</strong><br>
            ⭐ Ratings: <strong style="color:white;">{len(ratings):,}</strong><br>
            📈 Avg Rating: <strong style="color:white;">{ratings["rating"].mean():.2f}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.caption("Phase 7D • Final UI")
    st.caption("Content + Collaborative + Hybrid")


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-badge">PHASE 7D • PRODUCTION-STYLE INTERFACE</div>
        <h1>🎬 Movie Recommendation System</h1>
        <p>
            Discover movies you'll love using content similarity,
            collaborative filtering, and a hybrid recommendation engine.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATASET METRICS
# ============================================================

st.markdown(
    '<div class="section-title">📊 Dataset Overview</div>',
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-icon">🎬</div>
            <div class="metric-value">{len(movies):,}</div>
            <div class="metric-label">Movies</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-icon">👤</div>
            <div class="metric-value">{ratings["userId"].nunique():,}</div>
            <div class="metric-label">Users</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-icon">⭐</div>
            <div class="metric-value">{len(ratings):,}</div>
            <div class="metric-label">Ratings</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-icon">📈</div>
            <div class="metric-value">{ratings["rating"].mean():.2f}</div>
            <div class="metric-label">Average Rating</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# MOVIE SEARCH
# ============================================================

st.markdown(
    '<div class="section-title">🔎 Find a Movie</div>',
    unsafe_allow_html=True,
)

search_query = st.text_input(
    "Search movie title",
    placeholder="Try: Toy Story, Batman, Star Wars...",
)

if search_query.strip():
    search_results = movies[
        movies["title"]
        .str.contains(
            search_query.strip(),
            case=False,
            na=False,
            regex=False,
        )
    ].copy()

    search_results = search_results.head(100)

    if search_results.empty:
        render_empty_state(
            "No movie found",
            "Try a different title or a shorter search term.",
        )
        selected_movie = None
    else:
        selected_movie = st.selectbox(
            "Choose a movie",
            search_results["title"].tolist(),
        )
else:
    popular_movies = (
        stats.sort_values(
            ["rating_count", "average_rating"],
            ascending=[False, False]
        )
        .head(100)
    )

    selected_movie = st.selectbox(
        "Choose a movie",
        popular_movies["title"].tolist(),
        index=0,
    )


# ============================================================
# SELECTED MOVIE INFO
# ============================================================

if selected_movie:
    selected_row = movies[
        movies["title"] == selected_movie
    ].iloc[0]

    selected_stats = stats[
        stats["movieId"] == selected_row["movieId"]
    ].iloc[0]

    st.markdown(
        '<div class="section-title">🎞️ Selected Movie</div>',
        unsafe_allow_html=True,
    )

    a, b, c = st.columns([2.3, 1, 1])

    with a:
        st.markdown(
            f"""
            <div class="info-panel">
                <h3>🎬 {selected_row["title"]}</h3>
                <p>🎭 <strong>Genres:</strong> {selected_row["genres"]}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with b:
        st.metric(
            "Average Rating",
            f'{selected_stats["average_rating"]:.2f}'
        )

    with c:
        st.metric(
            "Ratings",
            f'{int(selected_stats["rating_count"]):,}'
        )


# ============================================================
# GENERATE BUTTON
# ============================================================

st.markdown("")

generate = st.button(
    "✨ Generate Personalized Recommendations",
    use_container_width=True,
    type="primary",
)


# ============================================================
# RECOMMENDATION ENGINE
# ============================================================

if generate:

    with st.spinner("Analyzing movies and generating recommendations..."):

        if mode == "Content-Based":

            recommendations = content_recommendations(
                selected_movie,
                recommendation_count,
                movies,
                similarity_matrix,
                title_to_index,
            )

            st.session_state["recommendations"] = recommendations
            st.session_state["recommendation_mode"] = mode

        elif mode == "Collaborative":

            recommendations = collaborative_recommendations(
                selected_user,
                recommendation_count,
                movies,
                ratings,
                collaborative_model,
            )

            st.session_state["recommendations"] = recommendations
            st.session_state["recommendation_mode"] = mode

        else:

            recommendations = hybrid_recommendations(
                selected_user,
                selected_movie,
                recommendation_count,
                movies,
                ratings,
                similarity_matrix,
                title_to_index,
                collaborative_model,
            )

            st.session_state["recommendations"] = recommendations
            st.session_state["recommendation_mode"] = mode


# ============================================================
# DISPLAY RECOMMENDATIONS
# ============================================================

if "recommendations" in st.session_state:

    recommendations = st.session_state["recommendations"]
    result_mode = st.session_state.get(
        "recommendation_mode",
        mode
    )

    st.markdown(
        f"""
        <div class="section-title">🎯 Recommended Movies</div>
        <div class="section-subtitle">
            <span class="mode-pill">{result_mode}</span>
            Personalized results for User {selected_user}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if recommendations.empty:

        render_empty_state(
            "No recommendations available",
            "There are not enough candidate movies for this selection.",
        )

    else:

        # Explanation panel
        if result_mode == "Content-Based":
            explanation = (
                f"Recommendations are based on genre similarity to "
                f"<strong>{selected_movie}</strong>."
            )

        elif result_mode == "Collaborative":
            explanation = (
                f"Recommendations are based on the learned rating "
                f"patterns of User <strong>{selected_user}</strong>."
            )

        else:
            explanation = (
                f"Hybrid recommendations combine content similarity to "
                f"<strong>{selected_movie}</strong> with predicted preferences "
                f"for User <strong>{selected_user}</strong>."
            )

        st.markdown(
            f"""
            <div class="info-panel">
                <h3>💡 Why these movies?</h3>
                <p>{explanation}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Two-column recommendation layout.
        left_col, right_col = st.columns(2)

        for i, (_, movie) in enumerate(
            recommendations.iterrows(),
            start=1
        ):

            target_col = left_col if i % 2 == 1 else right_col

            with target_col:

                if result_mode == "Content-Based":

                    render_movie_card(
                        rank=i,
                        title=movie["title"],
                        genres=movie["genres"],
                        score_label="Similarity",
                        score_value=f'{movie["score"]:.3f}',
                    )

                elif result_mode == "Collaborative":

                    render_movie_card(
                        rank=i,
                        title=movie["title"],
                        genres=movie["genres"],
                        score_label="Predicted Rating",
                        score_value=f'{movie["predicted_rating"]:.2f} / 5',
                    )

                else:

                    render_movie_card(
                        rank=i,
                        title=movie["title"],
                        genres=movie["genres"],
                        score_label="Predicted Rating",
                        score_value=f'{movie["predicted_rating"]:.2f} / 5',
                        extra_label="Hybrid Score",
                        extra_value=f'{movie["hybrid_score"]:.3f}',
                    )


# ============================================================
# USER RATING HISTORY
# ============================================================

st.markdown("---")

with st.expander(
    f"📚 View User {selected_user} Rating History",
    expanded=False,
):

    history = user_history(
        selected_user,
        ratings,
        movies
    )

    if history.empty:
        st.info("This user has no rating history.")
    else:

        h1, h2, h3 = st.columns(3)

        with h1:
            st.metric(
                "Movies Rated",
                f"{len(history):,}"
            )

        with h2:
            st.metric(
                "Average Rating",
                f'{history["rating"].mean():.2f}'
            )

        with h3:
            st.metric(
                "Highest Rating",
                f'{history["rating"].max():.1f}'
            )

        display_history = history[
            ["title", "genres", "rating"]
        ].copy()

        display_history.columns = [
            "Movie",
            "Genres",
            "Rating",
        ]

        st.dataframe(
            display_history.head(50),
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# ABOUT THE SYSTEM
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">🧠 How the Recommendation System Works</div>',
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(
    [
        "🎭 Content-Based",
        "👥 Collaborative",
        "🔀 Hybrid",
    ]
)

with tab1:
    st.markdown(
        """
        <div class="info-panel">
            <h3>🎭 Content-Based Filtering</h3>
            <p>
                The system converts movie genres into TF-IDF vectors and
                calculates cosine similarity. Movies with similar genre
                profiles receive higher similarity scores.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with tab2:
    st.markdown(
        """
        <div class="info-panel">
            <h3>👥 Collaborative Filtering</h3>
            <p>
                An SVD model learns latent user-movie preference patterns
                from the rating matrix and predicts ratings for movies
                the selected user has not rated.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with tab3:
    st.markdown(
        """
        <div class="info-panel">
            <h3>🔀 Hybrid Recommendation</h3>
            <p>
                The hybrid engine combines content similarity and
                collaborative predicted ratings. This gives the system
                both movie-level similarity and user-level personalization.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        🎬 Movie Recommendation System • Phase 7D<br>
        Built with Python • Pandas • Scikit-learn • Surprise • Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
