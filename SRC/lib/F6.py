"""
Version: 1.0
Last modified on: 17 November, 2014
Developers: Eduardo Nobre Luis, Fabricio Olivetti de Franca.
     email: eduardo_(DOT)_luis_(AT)_aluno_(DOT)_ufabc_(DOT)_edu_(DOT)_br
          : folivetti_(AT)_ufabc_(DOT)_edu_(DOT)_br
Based on source-code by Michael G. Epitropakis and Xiaodong Li
available at http://goanna.cs.rmit.edu.au/~xiaodong/cec15-niching/competition/
"""

# shubert
#Variable ranges: x_i in  [10, 10]^n, i=1,2,...,n
#No. of global peaks: n*3^n
#No. of local peaks: many

import numpy as np

def f6(x = None):

    if x is None:
        f6.lb = -10*np.ones(100)
        f6.ub = 10*np.ones(100)
        f6.nopt = 81
        f6.fgoptima = 186.731
        return None    
    
    D = x.shape[0]
    js = np.tile(np.arange(1,6),(D,1))
    mx = np.reshape(np.repeat(x, 5), (D,5))
    return -np.prod((js*np.cos(mx*(js+1.0) + js)).sum(axis=1))
