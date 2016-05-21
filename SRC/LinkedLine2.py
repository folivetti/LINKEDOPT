# coding: utf-8
from __future__ import print_function
# numerical
import numpy as np
from scipy.optimize import minimize_scalar, minimize
from scipy.spatial.distance import euclidean, pdist, cdist, squareform
from numpy import array
from operator import itemgetter

# niching benchmarking
from lib.niching_func import niching_func
from lib.count_goptima import count_goptima

from numba import autojit

import Startup as St
# Distance metric
@autojit
def LineSimple( x1, y1, x2, y2 ):
    eps = 1e-20
    xm = 0.5*(x2+x1)
    
    sumx = ((x2-x1)**2).sum()
    sumy = (y2-y1)**2
    sumdy = (0.5*(y1+y2) - f(xm))**2
    return np.sqrt(sumdy*sumx/(sumy+eps))

@autojit
def feaseable(x):
    D = x.shape[0] 
    return np.all(x >= f.lb[:D]) and np.all(x <= f.ub[:D])

@autojit        
def genDir(dim):
    dplus = np.random.uniform(low=-1.0,high=1.0,size=dim)
    return dplus/np.sqrt((dplus**2).sum())

def optNode(X, Y, Adj, lastidx, nodes, step, thr):

    nevals = 0
    for i in np.arange(nodes.shape[0]):
        x = np.copy(X[nodes[i],:])
        y = np.copy(Y[nodes[i]])
        dim = x.shape[0]
        d = genDir(dim)
        maxf = lambda alpha: -LineSimple(x,y,x+alpha*d, f(x+alpha*d))
        rl, ru = (f.lb[:dim] - x)/d, (f.ub[:dim] - x)/d
        if rl[rl>=0].shape[0] == 0 and ru[ru>=0].shape[0] == 0:
            continue
        r = np.append(rl[rl>=0], ru[ru>=0]).min()
        if r == 0:
            continue
        n = int(1 + (np.log(r) - np.log(step)) / np.log(2))
        rs = step*(2.0**np.arange(n-1,-1,-1))
        rs2 = np.minimum(rs*2.0, r*np.ones(rs.shape[0]))

        for j in np.arange(rs.shape[0]):
            ri = rs[j]
            ri2 = rs2[j]
            if ri2 < ri:
                continue
            alpha = minimize_scalar(maxf, bounds=(ri,ri2), method='bounded')
            nevals += alpha.nfev
            alpha = alpha.x
            x2 = x + alpha*d
            xm = (x+x2)/2.0
            y2 = f(x2)
            ym = f(xm)
            if ym > y2 and ym > y:
                X[nodes[i],:] = np.copy(xm)
                Y[nodes[i]] = np.copy(ym)
                if dim>1:
                    d = np.zeros(dim)
                    idx1, idx2 = np.random.choice(np.arange(dim),2,replace=False)
                    d[idx1],d[idx2] = (x2-xm)[idx1], (xm-x2)[idx2]
                    d = d/np.sqrt( (d**2).sum() )

            elif ym< y2 and ym < y:
                dist = cdist(np.array([xm]), X[:lastidx,:], 'euclidean')[0]
                closest = np.argmin(dist)
                if dist[closest] <= thr and y2 > y:
                    X[nodes[i],:] = np.copy(x2)
                    Y[nodes[i]] = np.copy(y2)
                elif lastidx < 1000:
                    X[lastidx,:] = np.copy(x2)
                    Y[lastidx] = np.copy(y2)
                    Adj[closest,lastidx] = 1
                    Adj[lastidx,closest] = 1
                    lastidx += 1
            elif y2 > ym and y2 > y:
                X[nodes[i],:] = np.copy(x2)
                Y[nodes[i]] = np.copy(y2)

    return lastidx, nevals 

@autojit
def candidateNodes(X, Y, Adj, lastidx, npop):
    if lastidx < npop:
        return np.arange(lastidx)

    degree = lastidx - Adj[:lastidx,:lastidx].sum(axis=0) + 1.0
    degree = degree[:lastidx]
    candnodes = np.random.choice(np.arange(lastidx), npop,
            p=degree/degree.sum())
    return candnodes

@autojit
def Supress(X, Y, Adj, lastidx, thr, thrL, ls):
    tabu = np.zeros(lastidx)
    flag = np.zeros(lastidx)
    idx = 0
    dist = squareform(pdist(X[:lastidx,:], 'euclidean'))
    ldist = np.zeros((lastidx,lastidx))

    for i in np.arange(lastidx-1):
        for j in np.arange(i+1,lastidx):
            ldist[i,j] = LineSimple(X[i],Y[i],X[j],Y[j])
            ldist[j,i] = ldist[i,j]
    for i in np.arange(lastidx):
        if tabu[i]==0:
            x, y = X[i], Y[i]
            idx = np.where(np.logical_or(dist[i]<=thr,ldist[i]<=thrL))[0]
            maxI = idx[np.argmax(Y[idx])]
            flag[maxI]=1
            tabu[idx]=1
    idx = np.where(flag==1)[0]
    lastidx=idx.shape[0]
    X[:lastidx] = np.copy(X[idx])
    Y[:lastidx] = np.copy(Y[idx])
    X[:lastidx], Y[:lastidx], nv = St.SciOpt(X[:lastidx], Y[:lastidx])


    Adj = np.zeros((1000,1000))
    if lastidx>1:
        for i in np.arange(lastidx):
            idx = np.argsort(cdist(np.array([X[i,:]]),X[:lastidx,:],
                'euclidean'))[0][1]
            Adj[i,idx] = 1
            Adj[idx,i] = 1
    #nv = 0
    #if ls:
    #    X[:lastidx], Y[:lastidx], nv = St.CMAOpt(X[:lastidx], Y[:lastidx], Adj)

    return lastidx, nv

@autojit
def LinkedLineOpt(maxit, dim, npop, step, thr, thrL,mute):

    maxpop = 1000
    X = np.zeros((maxpop,dim))
    Y = np.zeros(maxpop)
    Adj = np.zeros((maxpop,maxpop))
    lastidx = 1

    X[0,:] = f.lb[:dim] + (f.ub[:dim]-f.lb[:dim]) * 0.5 * np.ones(dim)
    Y[0] = f(X[0])
    nevals = 0
    
    for it in range(maxit):
        nodes = candidateNodes(X, Y, Adj, lastidx, npop)
        lastidx, nevals2 = optNode(X, Y, Adj, lastidx, nodes, step, thr)
        nevals += nevals2
        if it%20 == 0 and lastidx > 50:
            lastidx, nv = Supress(X, Y, Adj, lastidx, thr, thrL, True)
            nevals += nv
           
        if it % 5 == 0 and not mute:
            cg = cgopt1(X[:lastidx])
            print(lastidx, cg, nevals)
            if cg >= nopt*0.5:
                break
    lastidx, nv = Supress(X, Y, Adj, lastidx, thr, thrL, True)
    nevals += nv
    if not mute:
        print("end")
    print(lastidx, cgopt1(X[:lastidx]), nevals)
    return X[:lastidx,:], Y[:lastidx], nevals
