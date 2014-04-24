import pandas as pd
import numpy as np
from pandas import DataFrame

# Load the datas

path = '../data/ml-1m/'

unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
users = pd.read_table(path + 'users.dat', sep='::', header = None,names=unames)

rnames = ['user_id','movie_id', 'rating', 'timestamp']
ratings = pd.read_table(path+'ratings.dat', sep = '::', header = None, names = rnames)

mnames = ['movie_id', 'title', 'genres']
movies = pd.read_table(path+'movies.dat', sep ='::', header=None,names=mnames)

data = pd.merge(pd.merge(ratings, users) ,movies)
mean_ratings = data.pivot_table('rating', rows='title',cols='gender',aggfunc='mean')

mean_ratings_by_user = data.groupby('user_id')['rating'].mean()
mean_ratings_by_movie = data.groupby('movie_id')['rating'].mean()

# Create the bias b_i, b_u, and mu
mu = data.rating.mean()
b_u = mean_ratings_by_user-mu
b_i = mean_ratings_by_movie-mu

# Create the index of movie_id and user_id to make the link between the index
# and the actual names of users and movies

movie_id = mean_ratings_by_movie.index
user_id = mean_ratings_by_user.index

# Create the matrix that we have to complete
big_matrix = data.pivot_table('rating',rows='movie_id',cols='user_id')

#Create the tuple of index on which we want to optimize our objective function
list_triplet = DataFrame(data,columns=['user_id','movie_id','rating']).values








