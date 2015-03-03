"""
Version: 1.0
Last modified on: 17 November, 2014
Developers: Eduardo Nobre Luis, Fabricio Olivetti de Franca.
     email: eduardo_(DOT)_luis_(AT)_aluno_(DOT)_ufabc_(DOT)_edu_(DOT)_br
          : folivetti_(AT)_ufabc_(DOT)_edu_(DOT)_br
Based on source-code by Michael G. Epitropakis and Xiaodong Li
available at http://goanna.cs.rmit.edu.au/~xiaodong/cec15-niching/competition/
"""

# Composition_function_3

from lib.hybrid import *
import numpy as np

# aqui fiz uma forma de facilitar a criacao das funcoes composicao
# veja o arquivo hybrid.py para verificar como faco a funcao hybrid_comp
def f11(x = None):

    if x is None:
        f11.lb = -5*np.ones(100)
        f11.ub = 5*np.ones(100)
        f11.nopt = 6
        f11.fgoptima = 0
        f11.first_time = 0
        return None
    
    if f11.first_time == 0:
        f11.first_time=1
        D = x.shape[0]  # .shape[0] vai dar o len do vetor x
        f11.fn = 6 # numero de funcoes composicao
        lb, ub = -5, 5 # dominio da funcao
        o = np.loadtxt('data/optima.dat') # carrega o vetor de otimos
        if o.shape[1] >= D:
            f11.o = o[:,:D] # deixa o com tamanho D
        else:
            f11.o = lb + (ub - lb) * np.random.rand( (n,D) ) # senao cria matriz aleatoria (precisa deixar estatico posteriormente)

        # aqui eu criei um dicionario de funcoes, assim podemos acessar a funcao
        # usando f[0](x) por exemplo
        f11.f = {0:FEF8F2_func, 1:FEF8F2_func,
             2:weierstrass_func, 3:weierstrass_func,
             4:grienwank_func, 5:grienwank_func}

        f11.bias = np.zeros(f11.fn)  # embora o bias seja sempre zero, eh bom deixar aqui para futuros usos
        f11.sigma = np.array( [1,1,2,2,2,2] ) # vetor n-dimensional
        f11.lamb = np.array([1.0/4.0, 1.0/10.0, 2.0, 1.0, 2.0, 5.0]) # array lambda como um numpy array
        f11.lamb = np.tile(f11.lamb, (D,1)).T # lambda se torna uma matriz fn x D para facilitar certas operacoes futuras

        f11.M = []
        if D == 2:
            Md = np.loadtxt('data/CF3_M_D2.dat')
            for i in range(f11.fn):
                f11.M.append( Md[ i*D:i*D+D ,:] )
        elif D==3:
            Md = np.loadtxt('data/CF3_M_D3.dat')
            for i in range(f11.fn):
                f11.M.append( Md[ i*D:i*D+D ,:] )       
        elif D==5:
            Md = np.loadtxt('data/CF3_M_D5.dat')
            for i in range(f11.fn):
                f11.M.append( Md[ i*D:i*D+D ,:] )
        elif D==10:
            Md = np.loadtxt('data/CF3_M_D10.dat')
            for i in range(f11.fn):
                f11.M.append( Md[ i*D:i*D+D ,:] )
        elif D==20:
            Md = np.loadtxt('data/CF3_M_D20.dat')
            for i in range(f11.fn):
                f11.M.append( Md[ i*D:i*D+D ,:] )
        else:
            for i in range(f11.fn):
                f11.M.append( RotMatrixCondition(D,1) )

    return hybrid_comp( x, f11.fn, f11.f, f11.o, f11.sigma, f11.lamb, f11.bias, f11.M )
    # agora para os outros Fs basta setar esse parametro e usar hybrid_comp
