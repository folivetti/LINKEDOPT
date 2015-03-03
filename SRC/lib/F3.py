"""
Version: 1.0
Last modified on: 17 November, 2014
Developers: Eduardo Nobre Luis, Fabricio Olivetti de Franca.
     email: eduardo_(DOT)_luis_(AT)_aluno_(DOT)_ufabc_(DOT)_edu_(DOT)_br
          : folivetti_(AT)_ufabc_(DOT)_edu_(DOT)_br
Based on source-code by Michael G. Epitropakis and Xiaodong Li
available at http://goanna.cs.rmit.edu.au/~xiaodong/cec15-niching/competition/
"""

# uneven decreasing maxima
#Variable ranges: x in [0, 1]
#No. of global peaks: 1
#No. of local peaks: 4
import numpy as np

def f3(x=None):
    
    f3.lb = np.array([0]) 
    f3.ub = np.array([1])
    f3.nopt = 1
    f3.fgoptima = 1

    if x is None:
      return None
    
    return (np.exp(-2.0*np.log(2)*((x-0.08)/0.854)**2)*(np.sin(5*np.pi*(x**0.75-0.05)))**6).sum()
