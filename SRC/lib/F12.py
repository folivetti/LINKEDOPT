"""
Version: 1.0
Last modified on: 17 November, 2014
Developers: Eduardo Nobre Luis, Fabricio Olivetti de Franca.
     email: eduardo_(DOT)_luis_(AT)_aluno_(DOT)_ufabc_(DOT)_edu_(DOT)_br
          : folivetti_(AT)_ufabc_(DOT)_edu_(DOT)_br
Based on source-code by Michael G. Epitropakis and Xiaodong Li
available at http://goanna.cs.rmit.edu.au/~xiaodong/cec15-niching/competition/
"""

# Composition_function_4
from lib.hybrid import *
import numpy as np

# aqui fiz uma forma de facilitar a criacao das funcoes composicao
# veja o arquivo hybrid.py para verificar como faco a funcao hybrid_comp
def f12(x = None):

    if x is None:
        f12.lb = -5*np.ones(100)
        f12.ub = 5*np.ones(100)
        f12.nopt = 8
        f12.fgoptima = 0
        f12.first_time = 0
        return None
    
    if f12.first_time == 0:
        f12.first_time = 1
        D = x.shape[0]  # .shape[0] vai dar o len do vetor x
        f12.fn = 8 # numero de funcoes composicao
        lb, ub = -5, 5 # dominio da funcao
        
        o = np.loadtxt('data/optima.dat') # carrega o vetor de otimos
        if o.shape[1] >= D:
            f12.o = o[:,:D] # deixa o com tamanho D
        else:
            f12.o = lb + (ub - lb) * np.random.rand( (n,D) ) # senao cria matriz aleatoria (precisa deixar estatico posteriormente)

        # aqui eu criei um dicionario de funcoes, assim podemos acessar a funcao
        # usando f[0](x) por exemplo
        f12.f = {0:rastrigin_func, 1:rastrigin_func,
             2:FEF8F2_func, 3:FEF8F2_func,
             4:weierstrass_func, 5:weierstrass_func,
             6:grienwank_func, 7:grienwank_func}

        f12.bias = np.zeros(f12.fn)  # embora o bias seja sempre zero, eh bom deixar aqui para futuros usos
        f12.sigma = np.array( [1,1,1,1,1,2,2,2] ) # vetor n-dimensional
        f12.lamb = np.array([4.0,1.0,4.0,1.0,1/10.0,1/5.0,1/10.0,1/40.0]) # array lambda como um numpy array
        f12.lamb = np.tile(f12.lamb, (D,1)).T # lambda se torna uma matriz fn x D para facilitar certas operacoes futuras

        f12.M = []
        if D == 2:
            Md = np.loadtxt('data/CF4_M_D2.dat')
            for i in range(f12.fn):
                f12.M.append( Md[ i*D:i*D+D ,:] )
        elif D==3:
            Md = np.loadtxt('data/CF4_M_D3.dat')
            for i in range(f12.fn):
                f12.M.append( Md[ i*D:i*D+D ,:] )       
        elif D==5:
            Md = np.loadtxt('data/CF4_M_D5.dat')
            for i in range(f12.fn):
                f12.M.append( Md[ i*D:i*D+D ,:] )
        elif D==10:
            Md = np.loadtxt('data/CF4_M_D10.dat')
            for i in range(f12.fn):
                f12.M.append( Md[ i*D:i*D+D ,:] )
        elif D==20:
            Md = np.loadtxt('data/CF4_M_D20.dat')
            for i in range(f12.fn):
                f12.M.append( Md[ i*D:i*D+D ,:] )
        else:
            for i in range(f12.fn):
                f12.M.append( RotMatrixCondition(D,1) )

    return hybrid_comp( x, f12.fn, f12.f, f12.o, f12.sigma, f12.lamb, f12.bias,
            f12.M ) # agora para os outros Fs basta setar esse parametro e usar hybrid_comp

