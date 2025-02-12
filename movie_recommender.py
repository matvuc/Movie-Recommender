import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class MovieRecommender:

    def __init__(self, data_path):
        try:
            self.df = pd.read_csv(data_path)
            if self.df.empty:
                raise ValueError("The dataset is empty.")
        except FileNotFoundError:
            print(f"Error: The file '{data_path}' was not found. Try providing a valid path.")
            return
        except pd.errors.EmptyDataError:
            print(f"Error: The file '{data_path}' is empty.")
            return
        except Exception as e:
            print(f"An unexpected error occurred while loading the dataset: {e}")
            return
        self._prepare_data()
        self._compute_similarity()

    def _prepare_data(self):
        """Preprocessing data: handle missing columns, fill NaN,
        and apply Time Frequency - Inverse Document Frequency."""

        required_features = ["keywords", "cast", "genres", "director"]

        # Ensure all required columns exist
        missing_columns = [col for col in required_features if col not in self.df.columns]
        if missing_columns:
            print(f"Warning: Missing columns in dataset: {missing_columns}. Proceeding with available data.")

        # Fill NaN values and ensure all data is in string format
        for feature in required_features:
            if feature in self.df:
                self.df[feature] = self.df[feature].fillna("").astype(str)
            else:
                self.df[feature] = ""  # Create empty column if missing

        # combine relevant features into a single string
        self.df["combined_features"] = self.df.apply(lambda row: " ".join(row[required_features]), axis=1)

        # Apply TF-IDF Vectorization with error handling
        try:
            self.vectorizer = TfidfVectorizer(stop_words='english')
            self.count_matrix = self.vectorizer.fit_transform(self.df["combined_features"])
        except ValueError as e:
            print(f"Error: TF-IDF vectorization failed due to data issues: {e}")

    def _compute_similarity(self):
        """Compute cosine similarity between movies."""
        try:
            if not hasattr(self, "count_matrix") or self.count_matrix.shape[0] == 0:
                raise ValueError("TF-IDF matrix is empty. Ensure the dataset has valid data.")

            self.cosine_sim = cosine_similarity(self.count_matrix)

        except ValueError as e:
            print(f"Error: Unable to compute similarity. {e}")
            self.cosine_sim = None  # Set to None to indicate failure

    def _get_title_from_index(self, index):
        """Finding a movies index."""
        return self.df.iloc[index]["title"]

    def _get_index_from_title(self, title):
        """Finding a movies title."""
        return self.df[self.df.title == title].index[0]

    def recommend(self, movie_title, top_n=5):
        """Return top n recommended movies for a given movie title."""

        if not isinstance(movie_title, str) or not movie_title.strip():
            return ["Invalid movie title. Please enter a valid title."]

        # Check if the movie title exists in dataset
        movie_indices = self.df[self.df["title"] == movie_title].index.tolist()
        if not movie_indices:
            return ["Movie not found in dataset."]

        # Pick the first occurrence of it if multiple exist in dataset
        movie_index = movie_indices[0]

        # Compute similarity scores
        try:
            similar_movies = list(enumerate(self.cosine_sim[movie_index]))
            sorted_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)[1:top_n + 1]
        except Exception as e:
            return [f"Error computing recommendations: {e}"]

        return [self._get_title_from_index(movie[0]) for movie in sorted_movies]

if __name__ == "__main__":
    recommender = MovieRecommender("movie_dataset.csv")
    movie = input("Which movies would you like recommendations for? ")
    recommendations = recommender.recommend(movie)
    print(f"Top 5 recommendations for {movie}:")
    for i,rec in enumerate(recommendations,1):
        print(f"{i}. {rec}")
