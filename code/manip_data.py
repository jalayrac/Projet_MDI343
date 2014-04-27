import pandas as pd
from pandas import DataFrame, Series
import numpy as np
from stochastic_grad_descent import *
import random
import time

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
    
def evaluate_model(b_u,b_i,mu,L,R,triplet):
    RMSE = 0   
    R_tot = int(triplet.shape[0])    
    for r in range(R_tot):
        r_ui = triplet[r,2]
        r_hat_ui = mu+b_i[triplet[r,1]]+b_u[triplet[r,0]]+np.dot(L[triplet[r,0],:],R[triplet[r,1],:])
        RMSE = RMSE + pow((r_ui-r_hat_ui),2)
    #       RMSE = RMSE + abs(r_ui-r_hat_ui)
        
        
    RMSE = pow(RMSE,0.5)/R_tot
#    RMSE = RMSE/R_tot
    return RMSE

def prepare_dataset(size='1m'):
    
    if size=='1m':
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
        
        # Creation of the map between index and movie and user id
        # Movies
        movie_id_val = data.movie_id.unique()
        movie_id_val.sort()
        movie_to_index = Series(np.arange(data.movie_id.unique().size),index=movie_id_val)
        index_to_movie = Series(movie_id_val,index=np.arange(data.movie_id.unique().size))
        
        # Users
        user_id_val = data.user_id.unique()
        user_id_val.sort()
        user_to_index = Series(np.arange(data.user_id.unique().size),index=user_id_val)
        index_to_user = Series(user_id_val,index=np.arange(data.user_id.unique().size))
        
        # Now we modify the data in order to remap the index to the linear one
        data['user_id'] = data['user_id'].map(lambda x: user_to_index[x])
        data['movie_id'] = data['movie_id'].map(lambda x: movie_to_index[x])
    
        mean_ratings_by_user = data.groupby('user_id')['rating'].mean()
        mean_ratings_by_movie = data.groupby('movie_id')['rating'].mean()
        
        # Create the bias b_i, b_u, and mu
        mu = data.rating.mean()
        b_u = mean_ratings_by_user-mu
        b_i = mean_ratings_by_movie-mu
    
    return data,movies,users,index_to_user,index_to_movie,b_u,b_i,mu
    

if __name__=='__main__':
    
    data,movies,users,index_to_user,index_to_movie,b_u,b_i,mu = prepare_dataset(size='1m')
     
    #Create the tuple of index on which we want to optimize our objective function
    #Note : we sort the values by the user id in order to ease the creation of the 
    #test and train dataset
    triplet = DataFrame(data,columns=['user_id','movie_id','rating'])
    list_triplet = triplet.sort_index(by='user_id').values
       
    # Number of ratings by user
    ratings_by_user_cumsum = data.groupby('user_id').size().cumsum()  
    # Size of the database (number of ratings)
    
    #Learning Part
    
    #Create an index to choose a test set
    n_movie_by_user = 2
    ind_test = select_test_set(ratings_by_user_cumsum,n=n_movie_by_user)   
    triplet_test = list_triplet[ind_test,:]
    triplet_train = np.delete(list_triplet,ind_test,0)
    
    # Parameters for the strochastic gradient descent    
    alpha = 0.1
    gamma = 0.1
       
    temp_D = time.clock()
    L,R = simple_sgd(b_u,b_i,mu,triplet_train,alpha,gamma)
    temp_total = time.clock()-temp_D
    RMSE = evaluate_model(b_u,b_i,mu,L,R,triplet_test)
    
#R_tot = int(triplet_test.shape[0]) 
#histo = np.array([0])
#for r in range(R_tot):
#    r_ui = triplet_test[r,2]
#    r_hat_ui = mu+b_i[triplet_test[r,1]]+b_u[triplet_test[r,0]]+np.dot(L[triplet_test[r,0],:],R[triplet_test[r,1],:])
#    #RMSE = RMSE + pow((r_ui-r_hat_ui),2)
#    histo = np.concatenate((histo,[abs(r_ui-r_hat_ui)]))
#    RMSE = RMSE + abs(r_ui-r_hat_ui)
#    
#    
##RMSE = pow(RMSE,0.5)/R_tot
#RMSE = RMSE/R_tot
#
#x = histo
#pylab.hist(x, bins=20)
#pylab.show()









