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

def f(mp_arr,mp_it,mp_it2,mp_val,num,N,rr):
    for (i,j,r) in itt.izip(mp_it[(num*rr):((num+1)*rr)],mp_it2[(num*rr):((num+1)*rr)],mp_val[(num*rr):((num+1)*rr)]):
        mp_arr[i*N+j]=r
        


if __name__ == '__main__':
    N=6000
    nbModif=N*200
    nbThreads=4
    rr=nbModif/nbThreads
    lock = mp.Lock()
    pool=[]
    mp_arr=sct.Array(c.c_double,N*N,lock=False)
    mp_it=sct.Array(c.c_int,nbModif,lock=False)
    mp_it2=sct.Array(c.c_int,nbModif,lock=False)
    mp_val=sct.Array(c.c_double,nbModif,lock=False)
    listi=np.random.permutation(np.arange(N))
    listj=np.random.permutation(np.arange(N))
    
    # just random things to assign
    a=[]
    for num in np.arange(nbModif):
        n=num%N
        a.append((listi[n],listj[n],n))
        mp_it[num]=listi[n]
        mp_it2[num]=listj[n]
        mp_val[num]=n
    
    tic=time.clock()
    for num in range(nbThreads):
        p=mp.Process(target=f, args=(mp_arr,mp_it,mp_it2,mp_val,num,N,rr,))
        pool.append(p)
        p.start()
    for num in range(nbThreads):
        pool[num].join()
        
    print ('Parallel time ' +str(time.clock()-tic))
    
    ar=np.zeros(N*N)
    tic=time.clock()
    for (i,j,v) in a:
        ar[i*N+j]=v
    print('Normal time '+str(time.clock()-tic))
    
    #test  
    print('Test...')
    testar=np.zeros(N*N)
    for (i,j,v) in a:
        testar[i*N+j]=mp_arr[i*N+j]
    print('Norm diff '+str(np.linalg.norm(testar-ar)))
