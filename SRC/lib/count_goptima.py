"""
Version: 1.0
Last modified on: 17 November, 2014
Developers: Eduardo Nobre Luis, Fabricio Olivetti de Franca.
     email: eduardo_(DOT)_luis_(AT)_aluno_(DOT)_ufabc_(DOT)_edu_(DOT)_br
          : folivetti_(AT)_ufabc_(DOT)_edu_(DOT)_br
Based on source-code by Michael G. Epitropakis and Xiaodong Li
available at http://goanna.cs.rmit.edu.au/~xiaodong/cec15-niching/competition/
"""

from lib.niching_func import niching_func
import numpy as np
from scipy.spatial.distance import pdist, squareform

def count_goptima(pop, nfunc, accuracy):
    # pop: NP, D
    if len(pop.shape)==1:
      NP, D = pop.shape[0], 1
    else:
      NP,D = pop.shape[0], pop.shape[1]

    # Parameters for the competition
    nfuncs = [1,2,3,4,5,6,7,6,7,8,9,10,11,11,12,11,12,11,12,12]
    rho = [0.001, 0.001, 0.001, 0.001, 0.5, 0.5, 0.2, 0.5, 0.2] + [0.01]*11
    fgoptima = [200.0, 1.0, 1.0, 200.0, 1.03163, 186.731, 1.0,
                2709.0935, 1.0, -2.0] + [0]*10
    nopt = [2, 5, 1, 4, 2, 18, 36, 81, 216, 12, 6, 8, 6, 6, 8, 6, 8, 6, 6, 8]

    nf = nfuncs[nfunc]
    f = niching_func[nf]
    f()

    # Evaluate pop
    fpop = np.array([f(pop[i]) for i in range(NP)])
    # Descent sorting
    order = np.argsort(fpop)[::-1]

    # Sort population based on its fitness values
    cpop = pop[order]
    cpopfits = fpop[order]

    #Get seeds
    distance = squareform(pdist(cpop, 'sqeuclidean'))
    seeds = [ (cpop[i], cpopfits[i]) for i in range(NP) if all(np.where( distance[i] <=
        rho[nfunc] )[0] >= i) ]
    count = 0
    seeds, seedsfit = zip(*seeds)
    seedsfit = np.array(seedsfit)

    # Based on the accuracy: check wich seeds are global optimizers
    idx = np.where(np.abs(seedsfit - fgoptima[nfunc])<=accuracy)[0]
    count = min(len(idx), nopt[nfunc])
    finalseeds = np.array(seeds)[idx]

    return count, finalseeds
