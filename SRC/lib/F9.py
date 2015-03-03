"""
Version: 1.0
Last modified on: 17 November, 2014
Developers: Eduardo Nobre Luis, Fabricio Olivetti de Franca.
     email: eduardo_(DOT)_luis_(AT)_aluno_(DOT)_ufabc_(DOT)_edu_(DOT)_br
          : folivetti_(AT)_ufabc_(DOT)_edu_(DOT)_br
Based on source-code by Michael G. Epitropakis and Xiaodong Li
available at http://goanna.cs.rmit.edu.au/~xiaodong/cec15-niching/competition/
"""

# Composition_function_1
from lib.hybrid import *
import numpy as np

# aqui fiz uma forma de facilitar a criacao das funcoes composicao
# veja o arquivo hybrid.py para verificar como faco a funcao hybrid_comp
def f9(x = None):
    
    if x is None:
        f9.lb = -5*np.ones(100)
        f9.ub = 5*np.ones(100)
        f9.nopt = 6
        f9.fgoptima = 0
        f9.first_time = 0
        return None

    D = x.shape[0]  # .shape[0] vai dar o len do vetor x
    fn = 6 # numero de funcoes composicao
    lb, ub = -5, 5 # dominio da funcao

    if f9.first_time==0:
        f9.first_time = 1
        f9.o = np.loadtxt('data/optima.dat') # carrega o vetor de otimos
        if f9.o.shape[1] >= D:
            f9.o = f9.o[:,:D] # deixa o com tamanho D
        else:
            f9.o = lb + (ub - lb) * np.random.rand( (n,D) ) # senao cria matriz aleatoria (precisa deixar estatico posteriormente)


        # aqui eu criei um dicionario de funcoes, assim podemos acessar a funcao
        # usando f[0](x) por exemplo
        f9.f = {0:grienwank_func, 1:grienwank_func,
             2:weierstrass_func, 3:weierstrass_func,
             4:sphere_func, 5:sphere_func}

        f9.bias = np.zeros(fn)  # embora o bias seja sempre zero, eh bom deixar aqui para futuros usos
        f9.sigma = np.ones(fn) # vetor n-dimensional
        f9.lamb = np.array([1.0, 1.0, 8.0, 8.0, 1/5.0, 1/5.0]) # array lambda como um numpy array
        f9.lamb = np.tile(f9.lamb, (D,1)).T # lambda se torna uma matriz fn x D para facilitar certas operacoes futuras
        f9.M = [ np.eye(D) ]*6 # isso gasta bastante memoria, mas fica mais facil generalizar as outras funcoes
        

    return hybrid_comp( x, fn, f9.f, f9.o, f9.sigma, f9.lamb, f9.bias, f9.M ) # agora para os outros Fs basta setar esse parametro e usar hybrid_comp


