from src.content_based import ContentBasedRecommender


# Create recommendation model
model = ContentBasedRecommender(
    "data/movies.csv"
)


# Select a movie
movie = "Jumanji (1995)"


# Generate recommendations
recommendations = model.recommend(
    movie,
    n=10
)


print("\nRecommendations for:", movie)
print("-" * 60)

print(
    recommendations[
        ["title", "genres", "similarity"]
    ].to_string(index=False)
)