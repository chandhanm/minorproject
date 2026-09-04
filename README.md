# Movie Recommendation System

## Internship Project

A Data Science based Movie Recommendation System that recommends movies to users using Content-Based Filtering, Collaborative Filtering, and a Hybrid Recommendation approach.

---

## 1. Project Overview

Movie recommendation systems help users discover movies according to their interests and previous preferences.

This project develops a complete recommendation system using the MovieLens dataset. The system analyzes movie metadata and user ratings to generate personalized movie recommendations.

Three recommendation approaches are implemented:

1. Content-Based Filtering
2. Collaborative Filtering
3. Hybrid Recommendation

The final system is deployed through an interactive Streamlit web application.

---

## 2. Objectives

The main objectives of this project are:

- Collect and understand movie rating data.
- Perform data preprocessing and exploratory data analysis.
- Develop a Content-Based Recommendation System.
- Develop a Collaborative Filtering Recommendation System.
- Develop a Hybrid Recommendation System.
- Evaluate recommendation performance.
- Optimize the recommendation pipeline.
- Build an interactive Streamlit application.
- Provide personalized movie recommendations.

---

## 3. Dataset

The project uses the MovieLens dataset provided by GroupLens.

The dataset contains:

### movies.csv

Movie information including:

- movieId
- title
- genres

### ratings.csv

User rating information including:

- userId
- movieId
- rating
- timestamp

The uploaded dataset contains:

- 10,329 movies
- 105,339 ratings

---

## 4. Technologies Used

### Programming Language

- Python

### Libraries

- Pandas
- NumPy
- Scikit-learn
- Scikit-Surprise
- Matplotlib
- Seaborn
- SciPy
- Streamlit

### Development Tools

- Jupyter Notebook
- Visual Studio Code
- Git
- GitHub

---

## 5. Project Workflow

The project follows the following workflow:

Data Collection
        ↓
Data Preprocessing
        ↓
Exploratory Data Analysis
        ↓
Content-Based Filtering
        ↓
Collaborative Filtering
        ↓
Hybrid Recommendation
        ↓
Model Evaluation
        ↓
Optimization
        ↓
Streamlit Application
        ↓
Final Recommendation System

---

## 6. Data Preprocessing

The following preprocessing operations were performed:

- Checked missing values.
- Removed invalid rating records.
- Checked duplicate records.
- Verified data types.
- Converted timestamps into datetime format.
- Checked rating distributions.
- Merged movie and rating information where required.
- Processed movie genres for recommendation modeling.

---

## 7. Exploratory Data Analysis

Exploratory Data Analysis was performed to understand:

- Number of movies.
- Number of ratings.
- Rating distribution.
- Most-rated movies.
- Average movie ratings.
- Genre distribution.
- User activity.
- Relationship between ratings and movie popularity.

Visualizations were created using Matplotlib and Seaborn.

---

# 8. Recommendation Methods

## 8.1 Content-Based Filtering

The Content-Based Recommendation System recommends movies similar to a movie selected by the user.

Movie genres are converted into numerical features using TF-IDF.

Cosine Similarity is then used to calculate similarity between movies.

### Process

Movie Genres
      ↓
Text Preprocessing
      ↓
TF-IDF Vectorization
      ↓
Cosine Similarity
      ↓
Similar Movies

The system returns movies with the highest similarity scores.

---

## 8.2 Collaborative Filtering

Collaborative Filtering uses user-rating behavior to generate recommendations.

The project uses Singular Value Decomposition (SVD) from the Surprise library.

The model learns relationships between:

- Users
- Movies
- Ratings

The model predicts how highly a user may rate movies they have not rated yet.

---

## 8.3 Hybrid Recommendation

The Hybrid Recommendation System combines Content-Based and Collaborative Filtering.

The system combines:

- Content similarity score
- Predicted user rating

The combined score is used to rank recommendations.

This approach attempts to provide recommendations that are both relevant to the selected movie and personalized to the user.

---

# 9. Model Evaluation

The Collaborative Filtering model is evaluated using:

### RMSE

Root Mean Squared Error measures the difference between predicted and actual ratings.

### MAE

Mean Absolute Error measures the average absolute prediction error.

Recommendation quality is also checked using:

- Recommendation uniqueness
- Exclusion of already-rated movies
- Score validity
- Recommendation generation time

---

# 10. Streamlit Application

The final system includes an interactive Streamlit application.

The application provides:

- Movie search
- User selection
- Recommendation mode selection
- Content-Based recommendations
- Collaborative recommendations
- Hybrid recommendations
- User rating history
- Movie statistics
- Recommendation explanations

Run the application using:

```bash
python -m streamlit run app.py