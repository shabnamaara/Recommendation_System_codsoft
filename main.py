import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load MovieLens dataset for collaborative filtering
data = Dataset.load_builtin('ml-100k')
trainset, testset = train_test_split(data, test_size=0.25)

# Collaborative Filtering using SVD
algo = SVD()
algo.fit(trainset)
predictions = algo.test(testset)
accuracy.rmse(predictions)

# Load movie metadata for content-based filtering
# Assuming movies.csv contains columns 'movieId', 'title', 'genres'
movies = pd.read_csv('https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/books.csv', usecols=['book_id', 'title', 'authors', 'average_rating', 'ratings_count', 'image_url', 'small_image_url'])

# Create a TF-IDF matrix of the movie genres
tfidf = TfidfVectorizer(stop_words='english')
movies['authors'] = movies['authors'].fillna('')
tfidf_matrix = tfidf.fit_transform(movies['authors'])

# Calculate cosine similarity between all movies
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to get movie recommendations based on similarity score
def get_content_based_recommendations(title, cosine_sim=cosine_sim):
    idx = movies[movies['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Get the 5 most similar movies
    book_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[book_indices]

# Function to get collaborative filtering recommendations for a specific user
def get_collaborative_recommendations(user_id, algo=algo, trainset=trainset):
    user_rated_movies = [iid for (iid, _) in trainset.ur[int(user_id)]]
    all_movie_ids = [iid for iid in trainset.all_items()]
    
    predictions = []
    for movie_id in all_movie_ids:
        if movie_id not in user_rated_movies:
            predictions.append((movie_id, algo.predict(user_id, movie_id).est))
    
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    top_5_recommendations = predictions[:5]
    top_5_movie_ids = [iid for iid, _ in top_5_recommendations]
    
    return movies[movies['book_id'].isin(top_5_movie_ids)]['title']

# Example: Get content-based recommendations for a specific movie
print("Content-Based Recommendations for 'Harry Potter and the Philosopher's Stone (Harry Potter, #1)':")
print(get_content_based_recommendations("Harry Potter and the Philosopher's Stone (Harry Potter, #1)"))

# Example: Get collaborative filtering recommendations for a specific user
print("\nCollaborative Filtering Recommendations for User 196:")
print(get_collaborative_recommendations(196))
