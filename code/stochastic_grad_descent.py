# -*- coding: utf-8 -*-

import numpy as np
#from matplotlib.pyplot import pause
import random

def simple_sgd(n_u,n_i,triplet,alpha,gamma,norm='att_paper',r=50):
    
    # Prepare the factorization matrix
    L = np.random.random([n_u,r])
    R = np.random.random([n_i,r])
    
    if norm=='att_paper':
        
        n_total = int(triplet.shape[0])
        n_iter = int(n_total*np.log(n_total))    
        
        #val = compObjFunc(mu,b_u,b_i,index_u,index_i,L,R,triplet,alpha)
        #obj = np.array([val])
        e = 0
        for data in range(n_iter):
            
            ind = random.randint(0,n_total-1)
            e = triplet[ind,2]-np.dot(L[triplet[ind,0],:],R[triplet[ind,1],:])           
            t = np.array(L[triplet[ind,0],:], copy=True)
            L[triplet[ind,0],:] = L[triplet[ind,0],:]+gamma*(e*R[triplet[ind,1],:]-alpha*L[triplet[ind,0],:])
            R[triplet[ind,1],:] = R[triplet[ind,1],:]+gamma*(e*t-alpha*R[triplet[ind,1],:])
            #val = compObjFunc(mu,b_u,b_i,index_u,index_i,L,R,triplet,alpha)
            #obj = np.concatenate((obj,[val]))
            
        return L,R
        
    else:
        print('You must specified a valid norm!')
    
        
def compObjFunc(L,R,triplet,alpha):
    
    n_total = int(triplet.shape[0])
    obj = 0    
   
    for ind in range(n_total):
        e = triplet[ind,2]-np.dot(L[triplet[ind,0],:],R[triplet[ind,1],:])    
        obj = obj+e*e+alpha*(np.dot(L[triplet[ind,0],:],L[triplet[ind,0],:])+np.dot(R[triplet[ind,1],:],R[triplet[ind,1],:]))
    
    return obj
    
        
                  
