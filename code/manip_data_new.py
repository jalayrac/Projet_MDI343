import pandas as pd
from pandas import DataFrame, Series
import numpy as np
from stochastic_grad_descent import *
import random
import time
import test_jellyfish as jlf
import pylab
import matplotlib.pyplot as plt

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
       # RMSE = RMSE + pow((r_ui-r_hat_ui),2)
        RMSE = RMSE + abs(r_ui-r_hat_ui)
    #RMSE = pow(RMSE,0.5)/R_tot
    RMSE = RMSE/R_tot
    return RMSE

def prepare_dataset(size='1m'):
    
    if size=='10m':
        #Load the datas
        path = '../data/ml-10m/ml-10M100K/'
        rnames = ['user_id','movie_id', 'rating', 'timestamp']
        ratings = pd.read_table(path+'ratings.dat', sep = '::', header = None, names = rnames)
        mnames = ['movie_id', 'title', 'genres']
        movies = pd.read_table(path+'movies.dat', sep ='::', header=None,names=mnames)
        
        data = pd.merge(ratings,movies)
        movie_id_val = data.movie_id.unique()
        movie_id_val.sort()
        movie_to_index = Series(np.arange(data.movie_id.unique().size),index=movie_id_val)
        index_to_movie = Series(movie_id_val,index=np.arange(data.movie_id.unique().size))
        
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
        
        
    if size=='1m':
        # Load the datas
        path = '../data/ml-1m/'   
        # Users
#        unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
#        users = pd.read_table(path + 'users.dat', sep='::', header = None,names=unames)  
        rnames = ['user_id','movie_id', 'rating','timestamps']
        data = pd.read_table(path+'ratings.dat', sep = '::', header = None, names = rnames)   
        data = data.drop('timestamps',1)
        mnames = ['movie_id', 'title', 'genres']
        movies = pd.read_table(path+'movies.dat', sep ='::', header=None,names=mnames)
        
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
        
        data['rating']=data['rating']-mu-data['user_id'].map(lambda x: b_u[x])-data['movie_id'].map(lambda x: b_i[x])
    
    return data,movies,index_to_user,index_to_movie

def draw2DMovies(R,index_to_movie,movies,dim_x=0,dim_y=1):

    plt.figure()    
    plt.title('Representation des films dans notre espace')
    # Find the good movies   
    val_x = R[:,dim_x]
    val_y = R[:,dim_y]
    #val_x = 0.5*((val_x-min(val_x))/(max(val_x)-min(val_x))+0.5)
    #val_y = 0.5*(val_y-min(val_y))/(max(val_y)-min(val_y)+0.5)
    plt.xlim(min(val_x),max(val_x)+0.8)
    plt.ylim(min(val_y),max(val_y))
    
    i_ur = argmax(val_x+val_y)
    print(i_ur)
    i_bl = argmin(val_x+val_y)
    print(i_bl)
    i_ul = argmax(val_y-val_x)
    print(i_ul)
    i_br = argmax(val_x-val_y)
    print(i_br)
    i_z = argmin(abs(val_x)+abs(val_y))
    print(i_z)
    #Upper right
    plt.text(val_x[i_ur],val_y[i_ur],movies.title[movies.movie_id==i_ur].values[0])
    #Bottom Left
    plt.text(val_x[i_bl],val_y[i_bl],movies.title[movies.movie_id==i_bl].values[0])
    #Upper left
    plt.text(val_x[i_ul],val_y[i_ul],movies.title[movies.movie_id==i_ul].values[0])
    #Bottom right
    plt.text(val_x[i_br],val_y[i_br],movies.title[movies.movie_id==i_br].values[0])
    #Around zero
    plt.text(val_x[i_z],val_y[i_z],movies.title[movies.movie_id==i_z].values[0])
    
    plt.show()

def displayHisto(t_test,L,R):
    
    histo_z = []
    histo= []    
    R_tot = int(t_test.shape[0])   
    
    for r in range(R_tot):
        r_ui = triplet_test[r,2]
        r_hat_ui = np.dot(L[t_test[r,0],:],R[t_test[r,1],:])
        histo.append(abs(r_ui-r_hat_ui))
        histo_z.append(abs(r_ui))
    
    
    pylab.hist([histo,histo_z],bins=20,histtype='bar',label=['Estimateur avec optimisation','Estimateur par la moyenne'])
    pylab.legend()

if __name__=='__main__':
    
    temp_D = time.clock()
    triplet,movies,index_to_user,index_to_movie = prepare_dataset(size='1m')
    temp_F = time.clock()-temp_D
    
    print('Temps de preprocessing des donnees')    
    print(temp_F)
    #Create the tuple of index on which we want to optimize our objective function
    #Note : we sort the values by the user id in order to ease the creation of the 
    #test and train dataset      
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
    
#    # Parameters for the stochastic gradient descent    
    alpha = 0.1
    gamma = 0.1
#    
#    temp_D = time.clock()
#    print('Gradient descent...')
#    L,R = simple_sgd(triplet_train,alpha,gamma)
    n_u = index_to_user.shape[0]
    n_i = index_to_movie.shape[0]
    L_z = np.random.random([n_u,30])
    R_z = np.random.random([n_i,30])    
    
##    L,R=jlf.jellyfish(triplet_train,alpha,gamma,nb_epochs=13)
#    temp_total = time.clock()-temp_D
    displayHisto(triplet_test,L_z,R_z)
    draw2DMovies(R_z,index_to_movie,movies,dim_x=6,dim_y=9)
    


