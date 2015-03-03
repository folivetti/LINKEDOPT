"""
Version: 1.0
Last modified on: 17 November, 2014
Developers: Eduardo Nobre Luis, Fabricio Olivetti de Franca.
     email: eduardo_(DOT)_luis_(AT)_aluno_(DOT)_ufabc_(DOT)_edu_(DOT)_br
          : folivetti_(AT)_ufabc_(DOT)_edu_(DOT)_br
Based on source-code by Michael G. Epitropakis and Xiaodong Li
available at http://goanna.cs.rmit.edu.au/~xiaodong/cec15-niching/competition/
"""

#Equal maxima
#Variable ranges: x in [0, 1]
#No. of global peaks: 5
#No. of local peaks: 0

import numpy as np

def f2(x = None):

  f2.lb = np.array([0])
  f2.ub = np.array([1])
  f2.nopt = 5
  f2.fgoptima = 1.0

  if x is None:
    return None

  return (np.sin(5*np.pi*x)**6).sum()
