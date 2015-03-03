""" 11 150 10 0.01 0.01 0.01
 12 50 30 0.1 0.05 0.5
 13 100 20 0.01 0.05 0.05
 14 400 10 0.005 0.05 0.05
 15 100 20 0.01 0.05 0.05
 16 100 20 0.01 0.05 0.05
 17 100 20 0.01 0.05 0.05
 18 100 20 0.01 0.05 0.05
 19 100 20 0.01 0.05 0.05
 20 100 20 0.01 0.05 0.05

"""
from lib.niching_func import niching_func
from lib.count_goptima import count_goptima

from scipy.optimize import minimize
from bokeh.plotting import *
from scipy.spatial.distance import euclidean, pdist, cdist, squareform
import numpy as np

from deap import base
from deap import creator
from deap import tools
from deap import cma, algorithms

def plot_graph1D(xs,ys):
    # plot them
    x = np.arange(f.lb,f.ub,0.01)
    y = [f(xi) for xi in x]
    p = figure(plot_width=1024, plot_height=1024)
    hold(True)

    line(x,y)

    circle(xs[:,0],ys, size=np.ones(ys.shape)*15, color="green")

    for xi, yi in zip(xs[:,0],ys):
        idx = np.argsort( (xi-xs[:,0])**2 )[1]
        print xi, xs[idx,0]
        line([xi,xs[idx,0]],[yi,ys[idx]], color="green", line_width=3)
    
    p.ygrid.grid_line_color = "white"
    p.ygrid.grid_line_width = 2
    p.xgrid.grid_line_color = "white"
    p.xgrid.grid_line_width = 2
    p.axis.major_label_text_font_size = "18pt"
    p.axis.major_label_text_font_style = "bold"
    show()
    
def plot_graph2D(Xs,Ys):
    # plot them
    p = figure(x_range=[f.lb[0],f.ub[0]],y_range=[f.lb[1],f.ub[1]],plot_width=1024, plot_height=1024)
    hold(True)
    step = 200
    x = np.linspace(f.lb[0],f.ub[0],step)
    y = np.linspace(f.lb[1],f.ub[1],step)
    X, Y = np.meshgrid(x,y)
    
    Z = [[f(np.array([x,y])) for x,y in zip(xl,yl)] for xl,yl in zip(X,Y)]    
    image(
        image=[Z], x=[f.lb[0]], y=[f.lb[1]], dw=[f.ub[0]-f.lb[0]],
        dh=[f.ub[1]-f.lb[1]], palette="Spectral11")

    X1 = Xs[:,0]
    X2 = Xs[:,1]

    #Y = [y-Ys.min()+3 for y in Ys]
    Y = [15 for y in Ys]
    
    circle(X1,X2, size=Y, color="green")

    tabu = []
    for i,xi in enumerate(Xs):
        idx = np.argsort( ((xi-Xs)**2).sum(axis=1) )[1]
        tabu.append(idx)
        line([xi[0],Xs[idx,0]],[xi[1],Xs[idx,1]], color="green", line_width=3)

    p.ygrid.grid_line_color = "white"
    p.ygrid.grid_line_width = 2
    p.xgrid.grid_line_color = "white"
    p.xgrid.grid_line_width = 2
    p.axis.major_label_text_font_size = "18pt"
    p.axis.major_label_text_font_style = "bold"

    show()

def tupleize(func):
    """A decorator that tuple-ize the result of a function. This is useful
    when the evaluation function returns a single value.
    """
    def wrapper(*args, **kargs):
        return func(*args, **kargs),
    return wrapper

def checkBounds(lb, ub):
    def decorator(func):
        def wrapper(*args, **kargs):
            offspring = func(*args, **kargs)
            for child in offspring:
                for i in xrange(len(child)):
                    if child[i] > ub[i]:
                        child[i] = ub[i]
                    elif child[i] < lb[i]:
                        child[i] = lb[i]
            return offspring
        return wrapper
    return decorator

def CMAOpt(X,Y, Adj):

    Xopt = np.zeros(X.shape)
    Yopt = np.zeros(Y.shape)
    fopt = lambda x: -f(x)
    nevals = X.shape[0]*50*50 #10*X.shape[1]
    for i in range(X.shape[0]):

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)
        toolbox = base.Toolbox()
        toolbox.register("evaluate", f)
        toolbox.decorate("evaluate", tupleize)        
        
        neigh = np.where(Adj[i,:])[0]
        if neigh.shape[0] > 2:
            sigma = 2.0*((X[i]-X[neigh])**2).max()
        else:
            sigma = 0.2
        strategy = cma.Strategy(centroid=X[i], sigma=sigma, lambda_=50)#10*X.shape[1])
        toolbox.register("generate", strategy.generate, creator.Individual)
        toolbox.register("update", strategy.update)
        toolbox.decorate("generate", checkBounds(f.lb, f.ub))
        
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("max", np.max)
        hof = tools.HallOfFame(1, similar=np.array_equal)
        
        try:
	    algorithms.eaGenerateUpdate(toolbox, ngen=100, stats=stats, halloffame=hof, verbose=False)
	    #algorithms.eaGenerateUpdate(toolbox, ngen=50, stats=stats, halloffame=hof, verbose=False)
	    #algorithms.eaGenerateUpdate(toolbox, ngen=50, stats=stats, halloffame=hof, verbose=False)
            Xopt[i,:] = hof[0]
            Yopt[i] = f(hof[0])
        except:
            Xopt[i,:] = X[i,:]
            Yopt[i] = Y[i]
    return Xopt,Yopt, nevals

def SciOpt(X,Y):

    Xopt = np.zeros(X.shape)
    Yopt = np.zeros(Y.shape)
    fopt = lambda x: -f(x)
    nevals = 0
    for i in range(X.shape[0]):
        x = np.copy(X[i,:])
        y = np.copy(Y[i])
        dim = x.shape[0]
        xstar = minimize(fopt,x,bounds=zip(f.lb[:dim],f.ub[:dim]))
        Xopt[i,:] = xstar.x
        Yopt[i] = f(xstar.x)
        nevals += xstar.nfev
    return Xopt,Yopt, nevals

def Start( idx ):
    
    params = {}
    with open("parameters") as f:
        for i, line in enumerate(f):
            max_it, npop, step, thr, thrL = map(float,line.rstrip().split())
            params[i] = (int(max_it), int(npop), step, thr, thrL)
    f.closed

    nfuncs = [1,2,3,4,5,6,7,6,7,8,9,10,11,11,12,11,12,11,12,12]
    dims = [1, 1, 1, 2, 2, 2, 2, 3, 3, 2, 2, 2, 2, 3, 3, 5, 5, 10, 10, 20]
    nopts = [2, 5, 1, 4, 2, 18, 36, 81, 216, 12, 6, 8, 6, 6, 8, 6, 8, 6, 8, 8]
    max_fes = [5e4]*5 + [2e5]*2 + [4e5]*2 + [2e5]*4 + [4e5]*7
    nea2 = [1.0, 1.0, 1.0, 1.0, 1.0, 0.963, 0.945, 0.241, 0.621, 1.0, 0.98,
            0.852, 0.977, 0.83, 0.743, 0.673, 0.695, 0.667, 0.667, 0.362]
    
    nf, dim, nopt = nfuncs[idx], dims[idx], nopts[idx]
    f = niching_func[nf]
    f()

    print f.lb[0], f.ub[0], nopt, max_fes[idx], dim

    # results
    cgopt1 = lambda x: count_goptima(x,idx,1e-1)[0]
    cgopt2 = lambda x: count_goptima(x,idx,1e-2)[0]
    cgopt3 = lambda x: count_goptima(x,idx,1e-3)[0]
    cgopt4 = lambda x: count_goptima(x,idx,1e-4)[0]
    cgopt5 = lambda x: count_goptima(x,idx,1e-5)[0]
    optWhat = lambda x: count_goptima(x,idx,1e-1)[1]

    cgopt = (cgopt1, cgopt2, cgopt3, cgopt4, cgopt5, optWhat)
    return cgopt, params[idx], (f, dim, nopt, max_fes[idx], nea2[idx])
