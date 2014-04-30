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
               
        tL = np.array(L[(u*r):((u+1)*r)], copy=True)
        tR = np.array(R[(i*r):((i+1)*r)], copy=True)
        e = rat-mu-b_i[i]-b_u[u]-np.dot(tL,tR) 
        L[(u*r):((u+1)*r)] = tL+gamma*(e*tR-alpha*tL)
        R[(i*r):((i+1)*r)] = tR+gamma*(e*tL-alpha*tR)        
        
def partial_permute1(n_u,n_i,p,datau,datai,datar,q,prow,pcol):
#    print('begin')
#     #permutations
    pirow=np.random.permutation(np.arange(n_u))
    picol=np.random.permutation(np.arange(n_i))
#    print(p)
    #temp: faster ?
    temp_pr=p*1./n_u
    temp_pc=p*1./n_i
    
    #first pass: size of each chunk
    size_chunks=np.zeros((p,p))
    for (i,j) in itt.izip(datau,datai):
        a=int(temp_pr*(pirow[i]-1))
        b=int(temp_pc*(picol[j]-1))
        size_chunks[a,b]+=1
#    print('bla')
    #beginning and end of each chunks, iterators
    chunks_shapes=np.zeros((p,p,2),dtype=int)
    prec_chunk=0
    for a in range(p):
        for b in range(p):
            end_chunk=prec_chunk+size_chunks[a,b]
            chunks_shapes[a,b]=[prec_chunk,end_chunk]
            prec_chunk=end_chunk
#    print('blabla')
    #return shapes and permutations in queue
    q.put(chunks_shapes)
    prow=pirow.astype(int)
    pcol=picol.astype(int)
    
def partial_permute2(n_u,n_i,p,datau,datai,datar,cdatau,cdatai,cdatar,q_prec,prow,pcol,q_out):
    temp_pr=p*1./n_u
    temp_pc=p*1./n_i
    cs=q_prec.get()
    pirow=np.array(prow[:],dtype=int)
    picol=np.array(pcol[:],dtype=int)
    chunks_iter=np.zeros((p,p),dtype=int)
    for a in range(p):
        for b in range(p):
            chunks_iter[a,b]=cs[a,b][0]
            
    for (i,j,r) in itt.izip(datau,datai,datar):
        a=int(temp_pr*(pirow[i]-1))
        b=int(temp_pc*(picol[j]-1))
        loc=chunks_iter[a,b]
        cdatau[loc]=i
        cdatai[loc]=j
        cdatar[loc]=r
        chunks_iter[a,b]=loc+1
    q_out.put(cs)
    
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
            
    #second pass, copy data into chunks
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
    
def objFunc(mu,b_u,b_i,r,L,R,datau,datai,datar,alpha):
    obj=0
    for (u,i,rat) in itt.izip(datau,datai,datar):
        e = rat-mu-b_i[i]-b_u[u]-np.dot(L[(u*r):((u+1)*r)],R[(i*r):((i+1)*r)])
        obj+=e*e+alpha*(np.dot(L[(u*r):((u+1)*r)],L[(u*r):((u+1)*r)])+np.dot(R[(i*r):((i+1)*r)],R[(i*r):((i+1)*r)]))
    return obj
    
    
def jellyfish(b_u,b_i,mu,triplet,alpha,gamma,norm='att_paper',r=50,nb_epochs=None,p=3,double_permut=False,compute_obj=False):
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
    obj=[]
    
    if double_permut:
        pirow1=sct.RawArray(c.c_int,n_u)
        picol1=sct.RawArray(c.c_int,n_i)
        pirow2=sct.RawArray(c.c_int,n_u)
        picol2=sct.RawArray(c.c_int,n_i)
        q_temp=mp.Queue()
        q=mp.Queue()
        #main algo
        #first permutation
        permut1=mp.Process(target=partial_permute1,args=(n_u,n_i,p,datau,datai,datar,q_temp,pirow1,picol1,))
        permut1.start()
        permut1.join()
        permut1=mp.Process(target=partial_permute1,args=(n_u,n_i,p,datau,datai,datar,q_temp,pirow2,picol2,))
        permut1.start()
        permut2=mp.Process(target=partial_permute2,args=(n_u,n_i,p,datau,datai,datar,cdatau,cdatai,cdatar,q_temp,pirow1,picol1,q,))
        permut2.start()
        permut2.join()
        permut1.join()
        #swap
        pirow1,pirow2=pirow2,pirow1
        picol1,picol2=picol2,picol1
        #chunks shapes
        cs=q.get()
        
        for ep in range(nb_epochs):
            tic=time.clock()
            #read on cdata, write on data, read pirow1, write pirow2
            permut1=mp.Process(target=partial_permute1,args=(n_u,n_i,p,cdatau,cdatai,cdatar,q_temp,pirow2,picol2,))
            permut2=mp.Process(target=partial_permute2,args=(n_u,n_i,p,cdatau,cdatai,cdatar,datau,datai,datar,q_temp,pirow1,picol1,q,))
            permut1.start()
            permut2.start()
            #gradient: read on cdata, write on L and R (safely)
            for l in range(p):
                pool=[]
                for a in range(p):
                    proc=mp.Process(target=par_stoch_grad,args=(n_u,n_i,r,cdatau,cdatai,cdatar,mu,b_u,b_i,cs[a,(a+l)%p],L,R,alpha,gamma,))
                    proc.start()
                    pool.append(proc)
                for a in range(p):
                    pool[a].join()
            permut1.join()
            permut2.join()
            cs=q.get()
            #swap data
            datau,cdatau=cdatau,datau
            datai,cdatai=cdatai,datai
            datar,cdatar=cdatar,datar
            pirow1,pirow2=pirow2,pirow1
            picol1,picol2=picol2,picol1

            # obj function
            if compute_obj: #SLOW
                obj.append(objFunc(mu,b_u,b_i,r,L,R,datau,datai,datar,alpha))
                
            print('Epoch '+str(time.clock()-tic))
        
    else:
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
            cs=q.get()
            #swap data
            datau,cdatau=cdatau,datau
            datai,cdatai=cdatai,datai
            datar,cdatar=cdatar,datar
            
            # obj function
            if compute_obj: #SLOW
                obj.append(objFunc(mu,b_u,b_i,r,L,R,datau,datai,datar,alpha))
        
            print('Epoch '+str(time.clock()-tic))
        
    nL=np.zeros((n_u,r))
    nR=np.zeros((n_i,r))
    
    for u in range(n_u):
        nL[u,:]=L[(u*r):((u+1)*r)]
        
    for i in range(n_i):
        nR[i,:]=R[(i*r):((i+1)*r)]
        
    return nL,nR,obj

if __name__ == '__main__':
    print(0)

    
    