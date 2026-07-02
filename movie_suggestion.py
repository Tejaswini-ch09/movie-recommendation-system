import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Read dataset
movies = pd.read_csv("movies.csv")

# Convert genres into numbers
vectorizer = CountVectorizer()
genre_matrix = vectorizer.fit_transform(movies["genre"])

# Calculate similarity
similarity = cosine_similarity(genre_matrix)

# Ask user for a movie
movie_name = input("Enter a movie name: ")

# Check if movie exists
if movie_name in movies["title"].values:

    index = movies[movies["title"] == movie_name].index[0]

    scores = list(enumerate(similarity[index]))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    print("\nRecommended Movies:\n")

    for i in scores[1:6]:
        print(movies.iloc[i[0]]["title"])

else:
    print("Movie not found.")