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
def get_fgoptima(nfunc):
    fgoptima = [200.0, 1.0, 1.0, 200.0, 1.03163, 186.731, 1.0,
                2709.0935, 1.0, -2.0, np.zeros([1,10])]
    return fgoptima[nfunc]
