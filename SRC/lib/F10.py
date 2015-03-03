"""
Version: 1.0
Last modified on: 17 November, 2014
Developers: Eduardo Nobre Luis, Fabricio Olivetti de Franca.
     email: eduardo_(DOT)_luis_(AT)_aluno_(DOT)_ufabc_(DOT)_edu_(DOT)_br
          : folivetti_(AT)_ufabc_(DOT)_edu_(DOT)_br
Based on source-code by Michael G. Epitropakis and Xiaodong Li
available at http://goanna.cs.rmit.edu.au/~xiaodong/cec15-niching/competition/
"""

# Composition Function 2
from lib.hybrid import *
import numpy as np

# aqui fiz uma forma de facilitar a criacao das funcoes composicao
# veja o arquivo hybrid.py para verificar como faco a funcao hybrid_comp
def f10(x = None):

    if x is None:
        f10.lb = -5*np.ones(100)
        f10.ub = 5*np.ones(100)
        f10.nopt = 8
        f10.fgoptima = 0
        f10.first_time = 0
        return None
    
    if f10.first_time==0:
        f10.first_time = 1
        D = x.shape[0]  # .shape[0] vai dar o len do vetor x
        f10.fn = 8 # numero de funcoes composicao
        lb, ub = -5, 5 # dominio da funcao
        o = np.loadtxt('data/optima.dat') # carrega o vetor de otimos
        if o.shape[1] >= D:
            f10.o = o[:,:D] # deixa o com tamanho D
        else:
            f10.o = lb + (ub - lb) * np.random.rand( (n,D) ) # senao cria matriz aleatoria (precisa deixar estatico posteriormente)

        # aqui eu criei um dicionario de funcoes, assim podemos acessar a funcao
        # usando f[0](x) por exemplo
        f10.f = {0:rastrigin_func, 1:rastrigin_func,
             2:weierstrass_func, 3:weierstrass_func,
             4:grienwank_func, 5:grienwank_func,
             6:sphere_func, 7:sphere_func}

        f10.bias = np.zeros(f10.fn)  # embora o bias seja sempre zero, eh bom deixar aqui para futuros usos
        f10.sigma = np.ones(f10.fn) # vetor n-dimensional
        f10.lamb = np.array([1.0, 1.0, 10.0, 10.0, 1/10.0, 1/10.0, 1/7.0, 1/7.0]) # array lambda como um numpy array
        f10.lamb = np.tile(f10.lamb, (D,1)).T # lambda se torna uma matriz fn x D para facilitar certas operacoes futuras
        f10.M = [ np.eye(D) ]*8 # isso gasta bastante memoria, mas fica mais facil generalizar as outras funcoes

    return hybrid_comp( x, f10.fn, f10.f, f10.o, f10.sigma, f10.lamb, f10.bias, f10.M ) # agora para os outros Fs basta setar esse parametro e usar hybrid_comp
