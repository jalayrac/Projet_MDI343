import pandas as pd
from pandas import DataFrame, Series
import numpy as np
from stochastic_grad_descent import *
import random

def select_test_set(data,n=1):
    index_user = data.index 
    sel_ind = [random.randint(0,data[index_user[0]]-1)]
    for j in range(n-1):
            sel_ind = np.concatenate((sel_ind,[random.randint(0,data[index_user[0]]-1)]))
    
    for i in range(index_user.size-1):
        sub_ind = [random.randint(data[index_user[i]],data[index_user[i+1]]-1)]
        for j in range(n-1):
            sub_ind = np.concatenate((sub_ind,[random.randint(data[index_user[i]],data[index_user[i+1]]-1)]))
        
        sel_ind = np.concatenate((sel_ind,sub_ind))
           
    return sel_ind


if __name__=='__main__':
    # Load the datas
    path = '../data/ml-1m/'   
    # Users
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
    #
    #movie_id = mean_ratings_by_movie.index
    #user_id = mean_ratings_by_user.index
    
    # Create the matrix that we have to complete
    #big_matrix = data.pivot_table('rating',rows='movie_id',cols='user_id')
    
    #Create the tuple of index on which we want to optimize our objective function
    #Note : we sort the values by the user id in order to ease the creation of the 
    #test and train dataset
    triplet = DataFrame(data,columns=['user_id','movie_id','rating'])
    list_triplet = triplet.sort_index(by='user_id').values
    
    
    # Number of ratings by user
    ratings_by_user_cumsum = data.groupby('user_id').size().cumsum()
    
    # Size of the database (number of ratings)
    n_ratings = int(list_triplet.shape[0])
    
    
    
    # Creation of the map between index and movie and user id
    movie_id_val = data.movie_id.unique()
    movie_id_val.sort()
    movie_to_index = Series(np.arange(data.movie_id.unique().size),index=movie_id_val)
    user_id_val = data.user_id.unique()
    user_id_val.sort()
    user_to_index = Series(np.arange(data.user_id.unique().size),index=user_id_val)
    
    # Number of ratings by user
    ratings_by_user_cumsum = data.groupby('user_id').size().cumsum()
    # Creation of the training set and test set (a refaire mieux)
    
    #Create an index to choose a test set
    n_movie_by_user = 2
    ind_test = select_test_set(ratings_by_user_cumsum,n=n_movie_by_user)
    
    triplet_test = list_triplet[ind_test,:]
    triplet_train = np.delete(list_triplet,ind_test,0)
    
    
    alpha = 0.1
    gamma = 0.1
    
    # Toy Example
    #movie_to_index = np.arange(5)
    #user_to_index = np.arange(3)
    #list_triplet = np.array([[0,0,5],[0,2,3],[0,4,5],[1,1,4],[1,4,5],[2,2,5],[2,3,4],[2,4,1]])
    #mu = 4
    #dev_mean_u = np.array([0.333,0.5,-1])
    #b_u = Series(dev_mean_u,np.arange(3))
    #
    #dev_mean_i = np.array([1,0,0,0,-0.333])
    #b_i = Series(dev_mean_i,np.arange(5))
    
    #L,R = simple_sgd(movie_to_index,user_to_index,b_u,b_i,mu,list_triplet,alpha,gamma)















