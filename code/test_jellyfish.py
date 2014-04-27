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
def stoch_grad(nb_users,nb_films,r,datai,dataj,datar,inds,L,R):
    for (i,j,rat) in itt.izip(datai[inds[0]:inds[1]],dataj[inds[0]:inds[1]],datar[inds[0]:inds[1]]):
        L[(i*r):((i+1)*r)]=np.ones(r)
        R[(j*r):((j+1)*r)]=np.ones(r)
        
def permute_data(nb_users,nb_films,p,datai,dataj,datar,cdatai,cdataj,cdatar,q):
    #permutations
    pirow=np.random.permutation(np.arange(nb_users))
    picol=np.random.permutation(np.arange(nb_films))
    
    #temp: faster ?
    temp_pr=p*1./nb_users
    temp_pc=p*1./nb_films
    
    #first pass: size of each chunk
    size_chunks=np.zeros((p,p))
    for (i,j) in itt.izip(datai,dataj):
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
    for (i,j,r) in itt.izip(datai,dataj,datar):
        a=int(temp_pr*(pirow[i]-1))
        b=int(temp_pc*(picol[j]-1))
        loc=chunks_iter[a,b]
        cdatai[loc]=i
        cdataj[loc]=j
        cdatar[loc]=r
        chunks_iter[a,b]=loc+1
        
    #return shapes in queue
    q.put(chunks_shapes)

if __name__ == '__main__':
    #toy example: approx 100k modif
    nb_users=1000
    nb_films=1000
    p=3 #p+1 cores, one for permutation
    r=10
    nb_epochs=3
    datai=[]
    dataj=[]
    datar=[]
    
    #generate toy data
    for i in range(nb_users):
        for j in range(nb_films):
            if (np.random.rand(1)[0]<0.1):
                datai.append(i)
                dataj.append(j)
                datar.append(int(10*np.random.rand(1)[0]))
                
    #shared memory
    L=sct.RawArray(c.c_double,nb_users*r)
    R=sct.RawArray(c.c_double,nb_films*r)
    datai=sct.RawArray(c.c_int,datai)
    dataj=sct.RawArray(c.c_int,dataj)
    datar=sct.RawArray(c.c_int,datar)
    cdatai=sct.RawArray(c.c_int,datai)
    cdataj=sct.RawArray(c.c_int,dataj)
    cdatar=sct.RawArray(c.c_int,datar)
    q=mp.Queue()
    
    #main algo
    #first permutation
    permutp=mp.Process(target=permute_data,args=(nb_users,nb_films,p,datai,dataj,datar,cdatai,cdataj,cdatar,q,))
    permutp.start()
    permutp.join()
    cs=q.get()
    
    for ep in range(nb_epochs):
        tic=time.clock()
        #read on cdata, write on data
        permutp=mp.Process(target=permute_data,args=(nb_users,nb_films,p,cdatai,cdataj,cdatar,datai,dataj,datar,q,))
        permutp.start()
        #gradient: read on cdata, write on L and R (safely)
        for l in range(p):
            pool=[]
            for a in range(p):
                proc=mp.Process(target=stoch_grad,args=(nb_users,nb_films,r,cdatai,cdataj,cdatar,cs[a,(a+l)%p],L,R,))
                proc.start()
                pool.append(proc)
            for a in range(p):
                pool[a].join()
        permutp.join()
        #swap data
        tempi=datai
        tempj=dataj
        tempr=datar
        datai=cdatai
        dataj=cdataj
        datar=cdatar
        cdatai=tempi
        cdataj=tempj
        cdatar=tempr
        
        print('Epoch '+str(time.clock()-tic))
    
    