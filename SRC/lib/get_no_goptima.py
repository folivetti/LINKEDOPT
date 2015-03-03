"""
Version: 1.0
Last modified on: 17 November, 2014
Developers: Eduardo Nobre Luis, Fabricio Olivetti de Franca.
     email: eduardo_(DOT)_luis_(AT)_aluno_(DOT)_ufabc_(DOT)_edu_(DOT)_br
          : folivetti_(AT)_ufabc_(DOT)_edu_(DOT)_br
Based on source-code by Michael G. Epitropakis and Xiaodong Li
available at http://goanna.cs.rmit.edu.au/~xiaodong/cec15-niching/competition/
"""

import numpy as np
def get_no_goptima(nfunc):
    nopt = [2, 5, 1, 4, 2, 18, 36, 81, 216, 12, 6, 8, 6, 6, 8, 6, 8, 6, 8, 8]
    return nopt[nfunc]
