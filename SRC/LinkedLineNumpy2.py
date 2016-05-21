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
#@autojit
def LineSimple( x1, y1, x2, y2, xm, ym ):
    eps = 1e-20
    #xm = ratio*(x2+x1)
    #ym = f(xm)
    
    sumx = ((x2-x1)**2).sum()
    sumy = (y2-y1)**2
    sumdy = (0.5*(y1+y2) - ym)**2
    return np.sqrt(sumdy*sumx/(sumy+eps))

#@autojit
def feaseable(x):
    D = x.shape[0] 
    return np.all(x >= f.lb[:D]) and np.all(x <= f.ub[:D])

#@autojit        
def genDir(dim):
    dplus = np.random.uniform(low=-1.0,high=1.0,size=dim)
    return dplus/np.sqrt((dplus**2).sum())

def orthogonal(d):
    do = d.copy()
    do[0], do[1] = do[1], -do[0]
    do[2:] = 0.0
    do = do #- d
    print(do.dot(d))
    return do

def replaceNode(X, Y, x, y, idx, lastidx, thrl):
    eucdist = cdist(np.array([x]), X[:lastidx,:], 'euclidean')[0]
    closest = eucdist.argmin()
    xc, yc = X[closest,:], Y[closest]
    xcm = 0.5*(xc+x)
    ycm = f(xcm)
    dist2 = LineSimple( xc, yc, x, y, xcm, ycm )
    
    if dist2 > thrl:                    
        X[idx,:] = x.copy()
        Y[idx] = y
        return idx
    elif dist2 <= thrl and y > Y[closest]:
        X[closest,:] = x.copy()
        Y[closest] = y
        return closest
    return idx

def replaceOrInsert(X, Y, Adj, D, x, y, idx, lastidx, thrl):
    eucdist = cdist(np.array([x]), X[:lastidx,:], 'euclidean')[0]
    closest = eucdist.argmin()
    closest2 = eucdist.argsort()[1] if eucdist.shape[0] > 1 else closest
    xc, yc = X[closest,:], Y[closest]
    xc2, yc2 = X[closest2,:], Y[closest2]
    
    xcm = 0.5*(xc+x)
    ycm = f(xcm)
    dist2 = LineSimple( xc, yc, x, y, xcm, ycm )
    
    xcm2 = 0.5*(xc2+x)
    ycm2 = f(xcm2)
    dist3 = LineSimple( xc2, yc2, x, y, xcm2, ycm2 )
       
    if dist2 <= thrl and y > Y[closest]:
        X[closest,:] = x.copy()
        Y[closest] = y
        return closest, lastidx
    elif dist2 > thrl and lastidx < X.shape[0] and dist3 > thrl:
        X[lastidx,:] = x.copy()
        Y[lastidx] = y
        D[lastidx] = D[idx,:]
        Adj[closest, lastidx] = Adj[lastidx][closest] = 1
        lastidx += 1
        return lastidx-1, lastidx
    return idx, lastidx

def maxFeasible( x, d ):
    return min(np.abs((x - f.lb[0]) / d).min(), np.abs((f.ub[0] - x)/d).min())
    
def optNode(X, Y, Adj, D, lastidx, nodes, Step, thrl):

    nevals = 0
    
    for i in np.arange(nodes.shape[0]):
        idx = nodes[i]
        x1 = np.copy(X[idx,:])
        y1 = np.copy(Y[idx])
        
        dim = x1.shape[0]
        d = orthogonal(D[idx,:]) #genDir(dim)
        #print ('updating: ', x1,y1,d)

        step_g = Step[idx]
        alpha_i = 0
        alpha_f = step_g
        x2 = x1 + alpha_f*d
        
        #print(alpha_i, alpha_f)
        
        while not feaseable( x2 ) and alpha_i > 1e-9 and alpha_f > 1e-9:
            alpha_i /= 2.
            alpha_f /= 2.
            x2 = x1 + alpha_f*d
        
        while feaseable( x2 ) and np.abs(alpha_i-alpha_f)>1e-9:
            
            y2 = f(x2)
            alpha = np.random.uniform( alpha_i, alpha_f )
            xm = x1 + alpha*d
            ym = f(xm)
            dist = LineSimple( x1, y1, x2, y2, xm, ym )
            
            #print (x1,xm,x2,y1,ym,y2,alpha, d, dist)
            
            nevals += 2
            
            if dist > thrl: # it is a local optima
                if ym > y1 and ym > y2:  # maximum
                    idx = replaceNode(X, Y, xm, ym, idx, lastidx, thrl)
                    nevals += 1
                    alpha_i = 0
                    alpha_f = np.sqrt(np.square(xm-x2).sum())/4.
                    #print('max:',alpha_i, alpha_f)
                elif y1 > ym > y2:
                    alpha_i = alpha_i/2.
                    alpha_f = np.sqrt( np.square(xm-x1).sum() )
                    #print('max is around:',alpha_i, alpha_f)
                elif ym < y2:  # minimum
                    idx, lastidx = replaceOrInsert(X, Y, Adj, D, x2, y2, idx, lastidx, thrl)
                    nevals += 2
                    alpha_i = 0
                    alpha_f = 0
                    #print('min:',alpha_i, alpha_f)

            else: # increasing or decreasing direction
                if y2 > ym and ym >= y1: # increasing direction
                    idx = replaceNode(X, Y, x2, y2, idx, lastidx, thrl)                
                    nevals += 1
                    alpha_i = 0
                    alpha_f = min(2.*alpha, maxFeasible( x2, d )) # must ensure feasible
                    #print('inc:',alpha_i, alpha_f)
                elif y1 > y2 and y1 > ym: # decreasing direction
                    idx = replaceNode(X, Y, xm, ym, idx, lastidx, thrl)               
                    nevals += 1
                    alpha_i = alpha_f
                    alpha_f = min(2.*alpha_f, maxFeasible( xm, d ))
                    #print('dec:',alpha_i, alpha_f)
                    
                
            #step_g *= 2
            #alpha_i = alpha_f
            #alpha_f = alpha_i + step_g
            x1 = np.copy(X[idx,:])
            y1 = np.copy(Y[idx])
            x2 = x1 + alpha_f*d
        #print ('fini',x1,x2,y1,alpha_i, alpha_f)

    return lastidx, nevals 
   
#@autojit
def candidateNodes(X, Y, Adj, lastidx, npop):
    if lastidx < npop:
        return np.arange(lastidx)

    degree = lastidx - Adj[:lastidx,:lastidx].sum(axis=0) + 1.0
    degree = degree[:lastidx]
    candnodes = np.random.choice(np.arange(lastidx), npop,
            p=degree/degree.sum())
    return candnodes

#@autojit
def LinkedLineOpt(maxit, dim, npop, step, thrL,mute):

    maxpop = 1000
    X = np.zeros((maxpop,dim))
    D = np.zeros((maxpop,dim))
    Step = 0.01*np.ones(maxpop)
    Y = np.zeros(maxpop)
    Adj = np.zeros((maxpop,maxpop))
    lastidx = 1

    X[0,:] = f.lb[:dim] + (f.ub[:dim]-f.lb[:dim]) * 0.5 * np.ones(dim)
    D[0,:] = genDir(dim)
    Y[0] = f(X[0])
    nevals = 0
    
    for it in range(maxit):
        nodes = candidateNodes(X, Y, Adj, lastidx, npop)
        lastidx, nevals2 = optNode(X, Y, Adj, D, lastidx, nodes, Step, thrL)
        nevals += nevals2
           
        if it % 5 == 0 and not mute:
            cg = cgopt1(X[:lastidx])
            if not mute:
	            #print('size: {}, cg: {}, nvs: {}'.format(lastidx, cg, nevals))
	            pass
    if not mute:
        print("end")
    	print('size: {}, cg: {}, nvs: {}'.format(lastidx, cgopt1(X[:lastidx]), nevals))
    return X[:lastidx,:], Y[:lastidx], Adj[:lastidx,:lastidx], nevals
