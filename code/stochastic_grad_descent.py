# -*- coding: utf-8 -*-

import numpy as np
#from matplotlib.pyplot import pause
import random
import time

def simple_sgd(b_u,b_i,mu,triplet,alpha,gamma,norm='att_paper',r=50,compute_obj=False):
    
    n_u = b_u.shape[0]
    n_i = b_i.shape[0]
    # Prepare the factorization matrix
    L = np.random.random([n_u,r])
    R = np.random.random([n_i,r])
    obj=[]
    if norm=='att_paper':
        
        n_total = int(triplet.shape[0])
        n_iter = int(n_total*np.log(n_total))    
        
        if compute_obj:
            obj.append(compObjFunc(mu,b_u,b_i,L,R,triplet,alpha))

        e = 0
        tic=time.clock()
        for data in range(n_iter):
            
            ind = random.randint(0,n_total-1)
            e = triplet[ind,2]-mu-b_i[triplet[ind,1]]-b_u[triplet[ind,0]]-np.dot(L[triplet[ind,0],:],R[triplet[ind,1],:])           
            t = np.array(L[triplet[ind,0],:], copy=True)
            L[triplet[ind,0],:] = L[triplet[ind,0],:]+gamma*(e*R[triplet[ind,1],:]-alpha*L[triplet[ind,0],:])
            R[triplet[ind,1],:] = R[triplet[ind,1],:]+gamma*(e*t-alpha*R[triplet[ind,1],:])
            
            if (data%n_total==0):
                print('Epoch '+str(time.clock()-tic))
                tic=time.clock()
                if (compute_obj):
                    obj.append(compObjFunc(mu,b_u,b_i,L,R,triplet,alpha))
            
        return L,R,obj
        
    else:
        print('You must specified a valid norm!')
    
        
def compObjFunc(mu,b_u,b_i,L,R,triplet,alpha):
    
    n_total = int(triplet.shape[0])
    obj = 0    
   
    for ind in range(n_total):
        e = triplet[ind,2]-mu-b_i[triplet[ind,1]]-b_u[triplet[ind,0]]-np.dot(L[triplet[ind,0],:],R[triplet[ind,1],:])    
        obj = obj+e*e+alpha*(np.dot(L[triplet[ind,0],:],L[triplet[ind,0],:])+np.dot(R[triplet[ind,1],:],R[triplet[ind,1],:]))
    
    return obj
    
        
                  
