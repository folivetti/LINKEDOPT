import numpy as np
import Startup as St
import LinkedLineNumpy as LLO

def testeLeo(testnum):
    cgopt, params, fevals = St.Start(testnum)
    max_it, npop, step, thr, thrL = params
    f, dim, nopt, max_fes, nea2 = fevals
    LLO.f = f
    St.f = f
    cgopt1, cgopt2, cgopt3, cgopt4, cgopt5, optWhat = cgopt
    LLO.cgopt1 = cgopt1
    LLO.nopt = nopt
    X, Y, nevals = LLO.LinkedLineOpt(max_it,dim, npop, step, thr, thrL,True)
    PR = np.array([cgopt1(X), cgopt2(X), cgopt3(X), cgopt4(X), cgopt5(X)])
    SR = PR==nopt

    Xopt, Yopt, nevals2 = X,Y,nevals #St.SciOpt(X,Y)
    PRopt = np.array([cgopt1(Xopt), cgopt2(Xopt), cgopt3(Xopt), cgopt4(Xopt), cgopt5(Xopt)])
    SRopt = PRopt==nopt
    nopt = float(nopt)
    print optWhat(X)
    return PR/nopt, SR, nevals, PRopt/nopt, SRopt, nevals+nevals2


fPR = open('results/LLopt_PR.dat', 'a')
fSR = open('results/LLopt_SR.dat', 'a')
fNEV = open('results/LLopt_NEV.dat', 'a')
fPRL = open('results/LLoptL_PR.dat', 'a')
fSRL = open('results/LLoptL_SR.dat', 'a')
fNEVL = open('results/LLoptL_NEV.dat', 'a')
repetitions = 20
for testnum in range(0,20):
    PR = np.zeros( (repetitions,5) )
    SR = np.zeros( (repetitions,5) )
    NEV = np.zeros(repetitions)
    PRL = np.zeros( (repetitions,5) )
    SRL = np.zeros( (repetitions,5) )
    NEVL = np.zeros(repetitions)
    for i in range(repetitions):
        P, S, nevals, Popt, Sopt, nevals2 = testeLeo(testnum)
        PR[i,:] = P
        SR[i,:] = S
        NEV[i] = nevals
        PRL[i,:] = Popt
        SRL[i,:] = Sopt
        NEVL[i] = nevals2
    print PR.mean(axis=0)
    print SR.mean(axis=0)
    print NEV.mean(), NEV.std()
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
