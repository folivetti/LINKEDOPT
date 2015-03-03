# coding: utf-8
# numerical
import numpy as np
from scipy.optimize import minimize_scalar, minimize
from scipy.spatial.distance import euclidean
import networkx as nx
from numpy import array
from operator import itemgetter

# niching benchmarking
from lib.niching_func import niching_func
from lib.count_goptima import count_goptima

from numba import autojit

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

def AddEdge(G, x, y, thr):
    nn, dist = sorted( ((node, euclidean(x, G.node[node]['x'])) for j, node in
                  enumerate(G.nodes())), key=itemgetter(1) )[0]
    if dist <= thr:
        if y > G.node[nn]['y']:
            G.node[nn]['x'] = x
            G.node[nn]['y'] = y
    else:
        nidx = G.number_of_nodes()
        G.add_node(nidx, x=x, y=y)
        G.add_edge(nn,nidx)

def optNode(G, nodes, step, thr):

    nevals = 0
    for node in nodes:
        x = G.node[node]['x']
        dim = x.shape[0]
        y = G.node[node]['y']
        d = genDir(x.shape[0])
        maxf = lambda alpha: -LineSimple(x,y, x+alpha*d, f(x+alpha*d))
        rl = (f.lb[:dim] - x)/d
        ru = (f.ub[:dim] - x)/d
        r = np.append(rl[rl>=0], ru[ru>=0]).min()
        n = int(1 + (np.log(r) - np.log(step)) / np.log(2))
        rs = [step*(2**i) for i in range(n-1,-1,-1)]

        for ri in rs:
            ri2 = ri*2.0
            ri2 = ri2 if ri2 < r else r
            alpha = minimize_scalar(maxf, bounds=(ri,ri2), method='bounded')
            nevals += alpha.nfev
            alpha = alpha.x
            applyRule(G, node, x, y, alpha, d, thr)
    return nevals 

def applyRule(G, node, x, y, alpha, d, thr):
        x2 = x + alpha*d
        xm = (x+x2)/2.0
        y2, ym = f(x2), f(xm)
        if ym > y and ym > y2: #localMax
            G.node[node]['x'] = xm
            G.node[node]['y'] = ym
        elif ym < y2 and ym < y: #localMin
            AddEdge(G, x2,y2, thr)
        elif y2 > ym and y2 > y:
            G.node[node]['x'] = x2
            G.node[node]['y'] = y2

def candidateNodes(G, npop):
    nnodes = G.number_of_nodes()
    if nnodes < npop:
        return G.nodes()

    degree = np.fromiter( (nnodes - n[1] + 1 for n in G.degree_iter()), float )
    candnodes = np.random.choice(np.arange(degree.shape[0]), npop,
            p=degree/degree.sum())
    '''

    nodes, degree = zip(*[ (n[0], nnodes - n[1] + 1) for n in G.degree_iter() ])
    degree = np.array(degree, dtype=float)

    candnodes = np.random.choice(nodes, npop, p=degree/degree.sum())
    '''
    return candnodes

def createLinks(G, step, thr, npop):
   
    nodes = candidateNodes(G, npop)
    nevals = optNode(G,nodes,step, thr)
    
    return nevals

def Supress(G, thr, thrL):
    tabu = []
    Gn = nx.Graph()
    idx = 0

    for ni in G.nodes():
        x, y = G.node[ni]['x'], G.node[ni]['y']
        dist = ( (no, euclidean(x, G.node[no]['x'])) for no in G.nodes() )
        LS = lambda x2,y2: LineSimple(x,y,x2,y2)
        cluster, Y = zip(*((no, G.node[no]['y']) for no,d in dist if dist <= thr or LS(G.node[no]['x'],
            G.node[no]['y']) <= thrL))
        maxN = np.argmax(Y)
        node = cluster[maxN]
        if node not in tabu:
            Gn.add_node(idx,x=G.node[node]['x'], y=G.node[node]['y'])
            idx += 1
            tabu += cluster

    for n in Gn.nodes():
        dist = sorted( ((no, euclidean(x,Gn.node[no]['x'])) for no in Gn.nodes() if
                no!=n), key=itemgetter(1))
        Gn.add_edge(n, dist[0][0])

    return Gn

def LinkedLineOpt(maxit, dim, npop, step, thr, thrL):

    x = f.lb[:dim] + (f.ub[:dim]-f.lb[:dim]) * 0.5 * np.ones(dim)
    G = nx.Graph()
    G.add_node( 0, x=x, y=f(x) )
    nevals = 0
    
    for it in range(maxit):
        nodes = candidateNodes(G, npop)
        nevals += optNode(G,nodes,step, thr)
        #nevals += createLinks(G, step, thr, npop)
        if it%5 == 0 and G.number_of_nodes() > 50:
            G = Supress(G, thr, thrL)
           
        if it % 5 == 0:
            X = np.array([G.node[n]['x'] for n in G.nodes()])
            print len(X), cgopt1(X), nevals
    G = Supress(G, thr, thrL)
    print "end"
    return G, nevals
