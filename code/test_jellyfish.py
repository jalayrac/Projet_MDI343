# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 17:39:11 2014

@author: nicolas
"""

import multiprocessing as mp
import multiprocessing.sharedctypes as sct
import time
import numpy as np
import ctypes as c
import itertools as itt

#TODO: to complete, add parameters
def par_stoch_grad(n_u,n_i,r,datau,datai,datar,mu,b_u,b_i,inds,L,R,alpha,gamma,norm='att_paper'):
    for (u,i,rat) in itt.izip(datau[inds[0]:inds[1]],datai[inds[0]:inds[1]],datar[inds[0]:inds[1]]):
        
        e = rat-mu-b_i[i]-b_u[u]-np.dot(L[(u*r):((u+1)*r)],R[(i*r):((i+1)*r)])           
        tL = np.array(L[(u*r):((u+1)*r)], copy=True)
        tR = np.array(R[(i*r):((i+1)*r)], copy=True)
        L[(u*r):((u+1)*r)] = tL+gamma*(e*tR-alpha*tL)
        R[(i*r):((i+1)*r)] = tR+gamma*(e*tL-alpha*tR)        
        

        
def permute_data(n_u,n_i,p,datau,datai,datar,cdatau,cdatai,cdatar,q):
    #permutations
    pirow=np.random.permutation(np.arange(n_u))
    picol=np.random.permutation(np.arange(n_i))
    
    #temp: faster ?
    temp_pr=p*1./n_u
    temp_pc=p*1./n_i
    
    #first pass: size of each chunk
    size_chunks=np.zeros((p,p))
    for (i,j) in itt.izip(datau,datai):
        a=int(temp_pr*(pirow[i]-1))
        b=int(temp_pc*(picol[j]-1))
        size_chunks[a,b]+=1
        
    #beginning and end of each chunks, iterators
    chunks_shapes=np.zeros((p,p,2),dtype=int)
    chunks_iter=np.zeros((p,p),dtype=int)
    prec_chunk=0
    for a in range(p):
        for b in range(p):
            end_chunk=prec_chunk+size_chunks[a,b]
            chunks_shapes[a,b]=[prec_chunk,end_chunk]
            chunks_iter[a,b]=prec_chunk
            prec_chunk=end_chunk
            
    #second pass, copy data into chunks TODO: faster ? how ?
    for (i,j,r) in itt.izip(datau,datai,datar):
        a=int(temp_pr*(pirow[i]-1))
        b=int(temp_pc*(picol[j]-1))
        loc=chunks_iter[a,b]
        cdatau[loc]=i
        cdatai[loc]=j
        cdatar[loc]=r
        chunks_iter[a,b]=loc+1
        
    #return shapes in queue
    q.put(chunks_shapes)
    
def jellyfish(b_u,b_i,mu,triplet,alpha,gamma,norm='att_paper',r=15,nb_epochs=None,p=3):
    n_u = b_u.shape[0]
    n_i = b_i.shape[0]
    n_total = int(triplet.shape[0])
    if (nb_epochs is None):
        nb_epochs=np.log(n_total)
    
    L=sct.RawArray(c.c_double,np.random.random(n_u*r))
    R=sct.RawArray(c.c_double,np.random.random(n_i*r))
    datau=sct.RawArray(c.c_int,triplet[:,0].astype(int))
    datai=sct.RawArray(c.c_int,triplet[:,1].astype(int))
    datar=sct.RawArray(c.c_double,triplet[:,2])
    cdatau=sct.RawArray(c.c_int,n_total)
    cdatai=sct.RawArray(c.c_int,n_total)
    cdatar=sct.RawArray(c.c_double,n_total)
    b_u=sct.RawArray(c.c_double,b_u)
    b_i=sct.RawArray(c.c_double,b_i)
    q=mp.Queue()
        
    #main algo
    #first permutation
    permutp=mp.Process(target=permute_data,args=(n_u,n_i,p,datau,datai,datar,cdatau,cdatai,cdatar,q,))
    permutp.start()
    permutp.join()
    cs=q.get()
    
    for ep in range(nb_epochs):
        tic=time.clock()
        #read on cdata, write on data
        permutp=mp.Process(target=permute_data,args=(n_u,n_i,p,cdatau,cdatai,cdatar,datau,datai,datar,q,))
        permutp.start()
        #gradient: read on cdata, write on L and R (safely)
        for l in range(p):
            pool=[]
            for a in range(p):
                proc=mp.Process(target=par_stoch_grad,args=(n_u,n_i,r,cdatau,cdatai,cdatar,mu,b_u,b_i,cs[a,(a+l)%p],L,R,alpha,gamma,))
                proc.start()
                pool.append(proc)
            for a in range(p):
                pool[a].join()
        permutp.join()
        #swap data
        tempi=datau
        tempj=datai
        tempr=datar
        datau=cdatau
        datai=cdatai
        datar=cdatar
        cdatau=tempi
        cdatai=tempj
        cdatar=tempr
        
        print('Epoch '+str(time.clock()-tic))
        
    nL=np.zeros((n_u,r))
    nR=np.zeros((n_i,r))
    
    for u in range(n_u):
        nL[u,:]=L[(u*r):((u+1)*r)]
        
    for i in range(n_i):
        nR[i,:]=R[(i*r):((i+1)*r)]
        
    return nL,nR

if __name__ == '__main__':
    #toy example: approx 100k modif
    n_u=1000
    n_i=1000
    p=3 #p+1 cores, one for permutation
    r=10
    nb_epochs=3
    triplet=[]
    mu=0
    b_u=np.zeros(n_u)
    b_i=np.zeros(n_i)  
    alpha=0.1
    gamma=0.1
    
    #generate toy data
    for i in range(n_u):
        for j in range(n_i):
            if (np.random.rand(1)[0]<0.1):
                triplet.append([i,j,int(5*np.random.rand(1))])
                
    triplet=np.array(triplet)
                
    L,R=jellyfish(b_u,b_i,mu,triplet,alpha,gamma,nb_epochs=nb_epochs)

    
    