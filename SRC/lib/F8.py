"""
Version: 1.0
Last modified on: 17 November, 2014
Developers: Eduardo Nobre Luis, Fabricio Olivetti de Franca.
     email: eduardo_(DOT)_luis_(AT)_aluno_(DOT)_ufabc_(DOT)_edu_(DOT)_br
          : folivetti_(AT)_ufabc_(DOT)_edu_(DOT)_br
Based on source-code by Michael G. Epitropakis and Xiaodong Li
available at http://goanna.cs.rmit.edu.au/~xiaodong/cec15-niching/competition/
"""

# modified_rastrigin_all
#Variable ranges: x_i in [0, 1]^n, i=1,2,...,n
#No. of global peaks: \prod_{i=1}^n k_i
#No. of local peaks: 0

import numpy as np
def f8(x = None):

    if x is None:
        f8.lb = 0*np.ones(100)
        f8.ub = 1*np.ones(100)
        f8.nopt = 12
        f8.fgoptima = -2.0
        f8.k2 = np.array([3, 4])
        f8.k8 = np.array([1, 2, 1, 2, 1, 3, 1, 4])
        f8.k16 = np.array([1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 3, 1, 1, 1, 4])
        return None
    
    D = x.shape[0]
    if D==2:
        k = f8.k2 
    elif D==8:
        k = f8.k8
    elif D==16:
        k = f8.k16

    return -(10 + 9.0*np.cos(2.0*np.pi*k*x)).sum()
