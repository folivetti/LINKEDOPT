"""
Version: 1.0
Last modified on: 17 November, 2014
Developers: Eduardo Nobre Luis, Fabricio Olivetti de Franca.
     email: eduardo_(DOT)_luis_(AT)_aluno_(DOT)_ufabc_(DOT)_edu_(DOT)_br
          : folivetti_(AT)_ufabc_(DOT)_edu_(DOT)_br
Based on source-code by Michael G. Epitropakis and Xiaodong Li
available at http://goanna.cs.rmit.edu.au/~xiaodong/cec15-niching/competition/
"""

import numpy as np
from numba.decorators import jit, autojit

@autojit
def sphere_func(x):
    return (x**2).sum()

@autojit
def rastrigin_func(x):
    return np.sum( (x**2)-(10.*np.cos( 2.*np.pi*x ))+10)

@autojit
def grienwank_func(x):
    i = np.sqrt(np.arange(x.shape[0])+1.0)
    return np.sum(x**2)/4000.0 - np.prod(np.cos(x/i)) + 1.0

@autojit
def weierstrass_func(x):
    alpha = 0.5
    beta = 3.0
    kmax = 20
    D = x.shape[0]

    c1 = alpha**np.arange(kmax+1)
    c2 = 2.0*np.pi*beta**np.arange(kmax+1)
    c = -D*np.sum(c1*np.cos(c2*0.5))

    C1 = np.tile(c1, (D,1))
    C2 = np.tile(c2, (D,1))
    X = np.reshape(np.repeat(x,kmax+1),(D,kmax+1)) + 0.5
    return (C1*np.cos(C2*X)).sum() + c 
    
@autojit
def F8F2_func(x):
    f2 = 100.0 * ( (x[0]+1.0)**2 - x[1] - 1.0 )**2 + x[0]**2
    return 1.0 + (f2**2)/4000.0 - np.cos(f2)

@autojit
def FEF8F2_func(x):
    D = x.shape[0]
    fi = np.zeros(D)
    for i in np.arange(D-1):
        fi[i] = F8F2_func( (x[i],x[i+1]) )
    fi[D-1] = F8F2_func( (x[D-1],x[0]) )
    return fi.sum()
    '''
    return sum( F8F2_func(y) for y in zip( x[:D-1], x[1:D] ) ) + F8F2_func(
            (x[D-1],x[0]) )
    '''
@autojit
def RotMatrixCondition(D,c):
    A = np.random.randn( D, D )
    P,r = np.linalg.qr(A)
    A = np.random.randn( D, D )
    Q,r = np.linalg.qr(A)[0]
    u = np.random.rand(1,D)
    Dm = np.diag(c**( (u-np.min(u))(np.max(u)-np.min(u)) ))
    M = P.dot(Dm).dot(Q)
    return M
   
@autojit
def hybrid_comp( x, fn, f, o, sigma, lamb, bias, M ):

    D = x.shape[0]

    #criando w
    dx = x - o[:fn]
    w = np.exp(-(dx**2).sum(axis=1)/(2.0*D*sigma**2))
    '''
    w = np.zeros(fn)
    for i in range(fn):
        w[i] = np.exp(-np.sum( (x - o[i,:])**2 )/(2.0*D*sigma[i]**2))
    #w = np.array([ np.exp(-np.sum( (x - o[i,:])**2 )/(2.0*D*sigma[i]**2)) for i in range(fn) ])
    '''
    # cuidado aqui, como vamos alterar w, temos que armazenar o valor maximo, pois poderemos altera-lo
    # veja tambem que nao precisa do else!
    maxw = w.max()
    maxw10 = maxw**10 # para nao ter que calcular isso n vezes
    w[w != maxw] = w[w != maxw]*(1.0 - maxw10)
    w = w/w.sum()

    f_hat = np.zeros( fn )
    xx = np.ones(D)*5.0/lamb
    dx = dx/lamb

    for i in range(fn):
        f_hat[i] = f[i](dx[i,:].dot(M[i])) / f[i](xx[i,:].dot(M[i]))
    #f_hat = 2000.0*np.array([f[i](((x - o[i,:])/lamb[i,:]).dot(M[i])) / f[i]((xx/lamb[i,:]).dot(M[i])) for i in range(fn)])
    return -(2000*w*f_hat).sum()  # note que embora na documentacao ele retorna o valor positivo, no codigo exemplo eles retornam negativo
