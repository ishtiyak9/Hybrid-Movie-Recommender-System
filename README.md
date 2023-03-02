# Movie Recommendation System using Collaborative Filtering and Content-Based Filtering
This project is a movie recommendation system that combines two different techniques: collaborative filtering and content-based filtering. Collaborative filtering is based on the idea that if two users have similar preferences in the past, they are likely to have similar preferences in the future. On the other hand, content-based filtering is based on the idea that if two items have similar features, they are likely to be similar.

The dataset used in this project is the MovieLens dataset, which contains movie ratings from various users. The system uses this data to recommend movies to a user based on their previous ratings and other users with similar preferences.

The project is implemented in Python using the 
- numpy
- pandas
- scikit-learn
- surprise
- tensorflow
- keras
- pyspark.

The code preprocesses the data by normalizing the ratings and one-hot encoding the genres. It then splits the data into training and test sets and trains a Singular Value Decomposition (SVD) model using the training set.

In addition to collaborative filtering, the project also implements content-based filtering using the movie titles. It uses the TF-IDF vectorization technique to convert the titles into vectors, which are then used to compute cosine similarity between the movies.

The final step is to combine the two filtering techniques to create a hybrid system that recommends movies based on both collaborative and content-based filtering. The system computes similarity scores between the user's rated movies and all other movies, and then computes a weighted average of the top N movies.

To run the code, you need to have Python and the required libraries installed. You can download the MovieLens dataset from their website and place the CSV files in the same directory as the code. You can then run the code and specify a user ID for which you want to generate movie recommendations.

Overall, this project provides a simple and effective way to recommend movies to users based on their preferences and similar movies.
