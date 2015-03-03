import Startup as St

def testeLeo():
    import LinkedLineNumpy as LLO
    cgopt, params, fevals = St.Start(6)
    max_it, npop, step, thr, thrL = params
    f, dim, nopt, max_fes, nea2 = fevals
    LLO.f = f
    St.f = f
    cgopt1, cgopt2, cgopt3, cgopt4, cgopt5 = cgopt
    LLO.cgopt1 = cgopt1
    X, Y, nevals = LLO.LinkedLineOpt(max_it,dim, npop, step, thr, thrL)

testeLeo()
