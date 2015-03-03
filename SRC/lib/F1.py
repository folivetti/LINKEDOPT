"""
Version: 1.0
Last modified on: 17 November, 2014
Developers: Eduardo Nobre Luis, Fabricio Olivetti de Franca.
     email: eduardo_(DOT)_luis_(AT)_aluno_(DOT)_ufabc_(DOT)_edu_(DOT)_br
          : folivetti_(AT)_ufabc_(DOT)_edu_(DOT)_br
Based on source-code by Michael G. Epitropakis and Xiaodong Li
available at http://goanna.cs.rmit.edu.au/~xiaodong/cec15-niching/competition/
"""

# Five uneven Peak Trap
#Variable ranges: x in [0, 30]
#No. of global peaks: 2
#No. of local peaks: 3

import numpy as np

def f1(x=None):
    
    f1.lb = np.array([0])
    f1.ub = np.array([30])
    f1.nopt = 2
    f1.fgoptima = 200

    if x is None:
        return None

    if isinstance(x,np.ndarray):
        x=x[0]
    
    if (x>=0 and x<2.50):
        return 80*(2.5-x)
    elif (x>=2.5 and x<5):
        return 64*(x-2.5)
    elif (x >= 5.0 and x < 7.5):
        return 64*(7.5-x)
    elif (x >= 7.5 and x < 12.5):
        return 28*(x-7.5)
    elif (x >= 12.5 and x < 17.5):
        return 28*(17.5-x)
    elif (x >= 17.5 and x < 22.5):
        return 32*(x-17.5)
    elif (x >= 22.5 and x < 27.5):
        return 32*(27.5-x)
    elif (x >= 27.5 and x <= 30):
        return 80*(x-27.5)
    return -1e10
