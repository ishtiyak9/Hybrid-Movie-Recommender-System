import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from pyspark.ml.feature import HashingTF, IDF
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession
from pyspark.conf import SparkConf

conf = SparkConf().setAppName('my_app_name').setMaster('local[*]').set('spark.driver.host', 'localhost')
spark = SparkSession.builder.config(conf=conf).getOrCreate()


# Load and preprocess the data
movies_df = pd.read_csv('movies.csv')
ratings_df = pd.read_csv('ratings.csv')

# Merge the datasets
df = pd.merge(movies_df, ratings_df, on='movieId')

# One-hot encode the genres
mlb = MultiLabelBinarizer()
df = df.join(pd.DataFrame(mlb.fit_transform(df.pop('genres').str.split('|')),
                          columns=mlb.classes_,
                          index=df.index))

# Normalize the ratings
scaler = MinMaxScaler()
df['rating_norm'] = scaler.fit_transform(df[['rating']])

# define the Reader object
reader = Reader(rating_scale=(0.5, 5.0))

# load the dataframe into Surprise Dataset object
data = Dataset.load_from_df(df[['userId', 'movieId', 'rating_norm']], reader)

# Split the data into training and test sets
train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)

# Build the collaborative filtering model
algo = SVD()
algo.fit(train_df)

# Build the content-based filtering model
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_df['title'])
content_sim_matrix = cosine_similarity(tfidf_matrix)

# Convert data to PySpark format
spark_df = spark.createDataFrame(df[['userId', 'movieId', 'rating_norm']])

# Build the content-based filtering model using PySpark
hashing_tf = HashingTF(inputCol="title", outputCol="rawFeatures", numFeatures=10000)
tfidf = IDF(inputCol="rawFeatures", outputCol="features")
model = tfidf.fit(hashing_tf.transform(spark_df))
spark_df = model.transform(hashing_tf.transform(spark_df))
content_sim_matrix = model.transform(hashing_tf.transform(movies_df)).\
                        select(['movieId', 'features']).\
                        rdd.map(lambda x: (x[0], x[1])).collect()
content_sim_matrix = cosine_similarity([x[1].toArray() for x in content_sim_matrix])

# Combine the collaborative filtering and content-based filtering models
hybrid_sim_matrix = 0.5 * (algo.sim_all() + content_sim_matrix)

# Generate recommendations for a given user
user_id = 1
user_ratings = df[df['userId'] == user_id][['movieId', 'rating_norm']]
user_ratings = user_ratings.rename(columns={'movieId': 'id', 'rating_norm': 'rating'})

# Compute the similarity scores for the user's rated movies
sim_scores = hybrid_sim_matrix[user_ratings['id'].tolist(), :]

# Compute the weighted average of the top N movies
N = 10
weighted_scores = sim_scores.sum(axis=0) / sim_scores.sum()
weighted_scores = np.nan_to_num(weighted_scores)
recommendations = pd.DataFrame({'movieId': movies_df['movieId'], 'score': weighted_scores})
recommendations = pd.merge(recommendations, user_ratings, on='movieId')
recommendations = recommendations[~recommendations['id'].isin(user_ratings['id'])].sort_values('score', ascending=False).head(N)
print(recommendations)
