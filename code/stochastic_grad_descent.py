# -*- coding: utf-8 -*-

import numpy as np
#from matplotlib.pyplot import pause
import random

def simple_sgd(index_i,index_u,b_u,b_i,mu,triplet,alpha,gamma,norm='att_paper',r=5):
    
    n_u = index_u.shape[0]
    n_i = index_i.shape[0]
    # Prepare the factorization matrix
    L = np.random.random([n_u,r])
    R = np.random.random([n_i,r])
    
    if norm=='att_paper':
        
        n_total = int(triplet.shape[0])
        #n_iter = int(n_total*np.log(n_total))    
        n_iter = 100000
        
        #val = compObjFunc(mu,b_u,b_i,index_u,index_i,L,R,triplet,alpha)
        #obj = np.array([val])
        
        for data in range(n_iter):
            
            ind = random.randint(0,n_total-1)
            u = index_u[triplet[ind,0]]
            i = index_i[triplet[ind,1]]
            e = triplet[ind,2]-mu-b_i[triplet[ind,1]]-b_u[triplet[ind,0]]-np.dot(L[u,:],R[i,:])
            
            L[u,:] = L[u,:]+gamma*(e*R[i,:]-alpha*L[u,:])
            R[i,:] = R[i,:]+gamma*(e*L[u,:]-alpha*R[i,:])
            
            #val = compObjFunc(mu,b_u,b_i,index_u,index_i,L,R,triplet,alpha)
            #obj = np.concatenate((obj,[val]))
            
        return L,R
        
    else:
        print('You must specified a valid norm!')
        
def compObjFunc(mu,b_u,b_i,index_u,index_i,L,R,triplet,alpha):
    
    n_total = int(triplet.shape[0])
    obj = 0    
   
    for ind in range(n_total):
        u = index_u[triplet[ind,0]]
        i = index_i[triplet[ind,1]]
        e = triplet[ind,2]-mu-b_i[triplet[ind,1]]-b_u[triplet[ind,0]]-np.dot(L[u,:],R[i,:])    
        obj = obj+e*e+alpha*(np.dot(L[u,:],L[u,:])+np.dot(R[i,:],R[i,:]))
    
    return obj
    
        
                  
