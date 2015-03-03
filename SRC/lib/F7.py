"""
Version: 1.0
Last modified on: 17 November, 2014
Developers: Eduardo Nobre Luis, Fabricio Olivetti de Franca.
     email: eduardo_(DOT)_luis_(AT)_aluno_(DOT)_ufabc_(DOT)_edu_(DOT)_br
          : folivetti_(AT)_ufabc_(DOT)_edu_(DOT)_br
Based on source-code by Michael G. Epitropakis and Xiaodong Li
available at http://goanna.cs.rmit.edu.au/~xiaodong/cec15-niching/competition/
"""

# Vincent
#Variable range: x_i in [0.25, 10]^n, i=1,2,...,n
#No. of global optima: 6^n
#No. of local optima: 0

import numpy as np 
def f7(x = None):

    if x is None:
        f7.lb = 0.25*np.ones(100)
        f7.ub = 10*np.ones(100)
        f7.nopt = 216
        f7.fgoptima = 1.0
        return None
    
    D = x.shape[0]
    return (np.sin(10.0*np.log10(x))).sum()/float(D)
