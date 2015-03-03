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
def get_ub(fno):
    if (fno == 1):
        ub = 30
    elif(fno == 2 or fno == 3):
        ub = 1
    elif(fno == 4):
        ub = 6*np.ones([1,2])
    elif(fno == 5):
        ub = np.array( [1.9, 1.1] )
    elif(fno == 6 or fno == 8):
        ub = 10*np.ones([1,2])
    elif(fno == 7 or fno == 9):
        ub = 10*np.ones([1,2])
    elif(fno == 10):
        ub = np.ones([1,2]) 
    elif(fno == 11 or fno == 12 or fno == 13):
        dim = 2
        ub = 5*np.ones([1,dim])
    elif(fno == 14 or fno == 15):
        dim = 3
        ub = 5*np.ones([1,dim])
    elif(fno == 16 or fno == 17):
        dim = 5
        ub = 5*np.ones([1,dim])
    elif(fno == 18 or fno == 19):
        dim = 10
        ub = 5*np.ones([1,dim])
    elif(fno == 20):
        dim = 20
        ub = 5*np.ones([1,dim])
    else:
        ub = []
    return ub
