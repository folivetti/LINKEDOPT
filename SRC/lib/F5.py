"""
Version: 1.0
Last modified on: 17 November, 2014
Developers: Eduardo Nobre Luis, Fabricio Olivetti de Franca.
     email: eduardo_(DOT)_luis_(AT)_aluno_(DOT)_ufabc_(DOT)_edu_(DOT)_br
          : folivetti_(AT)_ufabc_(DOT)_edu_(DOT)_br
Based on source-code by Michael G. Epitropakis and Xiaodong Li
available at http://goanna.cs.rmit.edu.au/~xiaodong/cec15-niching/competition/
"""

# six_hump_camel_back
#Variable ranges: x in [-1.9, 1.9]; y in [-1.1, 1.1]
#No. of global peaks: 2
#No. of local peaks: 2
import numpy as np

def f5(x = None):

    if x is None:
        f5.lb = np.array([-1.9,-1.1])
        f5.ub = np.array([1.9,1.1])
        f5.nopt = 2
        f5.fgoptima = 1.03163
        return None
    
    y2 = x[1]**2
    x2 = x[0]**2
    return -(4.0 - 2.1*x2 + x[0]**4/3.0)*x2 -x[0]*x[1] - 4.0*y2*(y2 - 1.0)
