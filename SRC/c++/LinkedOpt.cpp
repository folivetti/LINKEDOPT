/******************************************************************************
 * Version: 1.0
 * Last modified on: 21 January, 2013 
 * Developers: Michael G. Epitropakis, Xiaodong Li.
 *      email: mge_(AT)_cs_(DOT)_stir_(DOT)_ac_(DOT)_uk 
 *           : xiaodong_(DOT)_li_(AT)_rmit_(DOT)_edu_(DOT)_au 
 * ***************************************************************************/
#include <iostream>
#include <cstdlib>
#include <valarray>

#include "cec2013.h"

struct Point{
    valarray< double > x;
    valarray< double > d;
    double y;
};

typedef struct Point Point;

#define MAXPOP 1000

using namespace std;

double LineSimple( Point *p1, Point *p2, Point *pm ){
    double eps = 1e-20;
    double sumx = 0, sumy = 0, sumdy = 0;
    sumx = ((p2->x - p1->x)*(p2->x - p1->x)).sum();
    sumy = (p2->y - p1->y)*(p2->y - p1->y);
    sumdy = 0.5*(y1+y2) - ym;
    sumdy *= sumdy;

    return sqrt(sumdy*sumx/(sumy+eps));
}

int feasable(vector< double > x){
    for(int i=0; i<x.size(); i++){
        if( x[i] < pF->lbound(i) || x[i] > pF->ubound(i) )
            return 0;
    }
    return 1;
}

vector< double > genDir(dim){

}

vector< int > candidateNodes( vector< double > Y, char [][] Adj, npop ){
    vector< int > candidates;
    if( Y.size() < npop ){
        for(int i=0;i<Y.size();i++){
            candidates.push_back(i);
        }
        return candidates;
    }

    vector< int > degree(Y.size());
    for(int i=0;i<Y.size();i++){
        degree[i] = Y.size() - sumVec(Adj[i]) + 1;
    }
    std::discrete_distribution<> dist(degree);
    std::mt19937 eng(std::time(0));
    vector< int > mask(Y.size());
    while( candidates.size() < npop ){
        int choice = dist(eng);
        if( !mask[choice] ){
            candidates.push_back(choice);
            mask[choice] = 1;
        }
    }

    return candidates;
}

void LinkedLine(maxit, dim, npop, thrL, pF){

    vector< vector<double> > X(MAXPOP), D(MAXPOP);
    vector< double > Y(MAXPOP);
    char Adj[MAXPOP][MAXPOP];
    
    double step[MAXPOP];
    for(int i=0;i<MAXPOP;i++) step=0.01;

    for(int i=0; i<dim; i++){
        X[0].push_back( pF->get_lbound(i) + (pF->get_ubound(i) - pF->get_lbound(i))*0.5);
    }
    D[0] = genDir(dim);
    Y[0] = pF->evaluate(X[0]);
    for( int it=0; it<maxit; it++ ){
        nodes = candidateNodes(X, Y, Adj, lastidx, npop);
        lastidx = optNode(X, Y, Adj, D, lastidx, nodes, step, thrL);
    }

}

int main()
{
    /* Demonstration of all functions */
    CEC2013 f1(1), f2(2), f3(3), f4(4), f5(5), f6(6), f7(7), f8(8), 
            f9(9), f10(10), f11(11), f12(12), f13(13), f14(14), 
            f15(15), f16(16), f17(17), f18(18), f19(19), f20(20);

    /* Iterate through functions */
    CEC2013 *pFunc;
    for (int index=1; index<=20; ++index) {
        /* Create one */
        pFunc = new CEC2013(index);

        int dim = pFunc->get_dimension();
        LinkedLine(maxit, dim, npop, thrL, pFunc);
        delete pFunc;
    }

    /**********************************************************************
     *  Demonstration of using how_many_goptima function 
     *********************************************************************/
    pFunc = new CEC2013(14);
    /* Create a population: std::vector< std::vector<double> > */
    std::vector< std::vector<double> > pop;
    const int dim(pFunc->get_dimension());
    for (int item=0; item<10; ++item) {
        vector<double> x(dim);
        for (int i=0;i<dim; ++i) {
            x[i] = rand_uniform();
        }
        pop.push_back(x);
    }
    /* Print population */
    cout << "-------------------------------------------------------" <<endl;
    for (std::vector< std::vector<double> >::iterator it = pop.begin(); 
            it != pop.end(); ++it) {
        cout << "Fitness: " << pFunc->evaluate(*it) << "\tGene:\t";
        for (std::vector<double>::iterator jt = it->begin();
                jt != it->end(); ++jt) {
            cout << *jt << "\t";
        }
        cout << endl;
    }
    cout << "-------------------------------------------------------" <<endl;

    /* Calculate how many global optima are in the population */
    double accuracy=0.001;
    std::vector< std::vector<double> > seeds; 
    cout << "In the current population there exist " 
        << how_many_goptima(pop, seeds, pFunc, accuracy, pFunc->get_rho()) 
        << " global optimizers." << endl;

    /* Clean up */
    delete pFunc;

    return EXIT_SUCCESS;
}
