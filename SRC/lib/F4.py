"""
Version: 1.0
Last modified on: 17 November, 2014
Developers: Eduardo Nobre Luis, Fabricio Olivetti de Franca.
     email: eduardo_(DOT)_luis_(AT)_aluno_(DOT)_ufabc_(DOT)_edu_(DOT)_br
          : folivetti_(AT)_ufabc_(DOT)_edu_(DOT)_br
Based on source-code by Michael G. Epitropakis and Xiaodong Li
available at http://goanna.cs.rmit.edu.au/~xiaodong/cec15-niching/competition/
"""

# himmelblau
#Variable ranges: x, y in [6, 6]
#No. of global peaks: 4
#No. of local peaks: 0
import numpy as np
def f4(x = None):
    
    if x is None:
        f4.lb = np.array([-6,-6])
        f4.ub = np.array([6,6])
        f4.nopt = 4
        f4.fgoptima = 200.0
        return None
    
    return 200 - (x[0]**2 + x[1] - 11)**2 - (x[0] + x[1]**2 - 7)**2
