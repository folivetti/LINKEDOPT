import numpy as np
import Startup as St
import LinkedLineNumpy2 as LLO

# thrL = (ub - lb)*sqrt(d)*0.1
def testeLeo(testnum, rep):
    mute = False if rep==0 else True
    cgopt, params, fevals = St.Start(testnum, mute)
    max_it, npop, step, thr, thrL = params
    f, dim, nopt, max_fes, nea2 = fevals
    LLO.f = f
    St.f = f
    cgopt1, cgopt2, cgopt3, cgopt4, cgopt5, optWhat = cgopt
    LLO.cgopt1 = cgopt1
    LLO.nopt = nopt
    
    thrL = (f.ub[0] - f.lb[0])*np.sqrt(dim)*0.1
    print(thrL)
    X, Y, Adj, nevals = LLO.LinkedLineOpt(max_it,dim, npop, 0.1, thrL, False)
    PR = np.array([cgopt1(X), cgopt2(X), cgopt3(X), cgopt4(X), cgopt5(X)])
    SR = PR==nopt

    Xopt, Yopt, nevals2 = X,Y,nevals #St.SciOpt(X,Y)
    PRopt = np.array([cgopt1(Xopt), cgopt2(Xopt), cgopt3(Xopt), cgopt4(Xopt), cgopt5(Xopt)])
    SRopt = PRopt==nopt
    nopt = float(nopt)
    #print 'Opt: ', optWhat(X)
    return PR/nopt, SR, nevals, PRopt/nopt, SRopt, nevals+nevals2


fPR = open('results2/LLopt_PR.dat', 'a')
fSR = open('results2/LLopt_SR.dat', 'a')
fNEV = open('results2/LLopt_NEV.dat', 'a')
fPRL = open('results2/LLoptL_PR.dat', 'a')
fSRL = open('results2/LLoptL_SR.dat', 'a')
fNEVL = open('results2/LLoptL_NEV.dat', 'a')
repetitions = 1
for testnum in range(10,12):
    PR = np.zeros( (repetitions,5) )
    SR = np.zeros( (repetitions,5) )
    NEV = np.zeros(repetitions)
    PRL = np.zeros( (repetitions,5) )
    SRL = np.zeros( (repetitions,5) )
    NEVL = np.zeros(repetitions)
    for i in range(repetitions):
        P, S, nevals, Popt, Sopt, nevals2 = testeLeo(testnum, i)
        PR[i,:] = P
        SR[i,:] = S
        NEV[i] = nevals
        PRL[i,:] = Popt
        SRL[i,:] = Sopt
        NEVL[i] = nevals2
    print 'PR: ', PR.mean(axis=0)
    print 'SR: ', SR.mean(axis=0)
    print 'NEV: ', NEV.mean(), NEV.std()
    np.savetxt(fPR,PR.mean(axis=0),newline='\t')
    np.savetxt(fSR,SR.mean(axis=0),newline='\t')
    np.savetxt(fNEV,np.array([NEV.mean(), NEV.std()]),newline='\t')
    np.savetxt(fPRL,PRL.mean(axis=0),newline='\t')
    np.savetxt(fSRL,SRL.mean(axis=0),newline='\t')
    np.savetxt(fNEVL,np.array([NEVL.mean(), NEVL.std()]), newline='\t')

    fPR.write('\n')
    fSR.write('\n')
    fNEV.write('\n')
    fPRL.write('\n')
    fSRL.write('\n')
    fNEVL.write('\n')

fPR.close()
fSR.close()
fNEV.close()
fPRL.close()
fSRL.close()
fNEVL.close()
