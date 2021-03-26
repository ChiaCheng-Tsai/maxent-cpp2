#include <iostream>
#include <vector>
#include <Eigen/Dense>

using namespace std;
using Eigen::MatrixXd;
using namespace Eigen;

#include "Maxent.hpp"

int main(int argc, const char * argv[])
{
    /*
     a 1D cloud of 3 nodes
    */

    int dim=1; //spatial dimension

    int NNodesC = 3; //Number Nodes contribute

    VectorXd point(dim); //coordinates of the sample points p(x,y,z)
    point(0)=0.3;

    MatrixXd NodesContribute(NNodesC,dim); //Vector class Node contribute
    NodesContribute(0,0)=0.;
    NodesContribute(1,0)=0.5;
    NodesContribute(2,0)=1.;

    int MaxIter=100; //maximum number of iteration

    double ctol= 1E-14; //convergence tolerance

    double gamma=1.0; // parameter that controls the support size of the basic function

    double hnode = 1; //characteristic nodal spacing

    string prior = "uniform"; // prior in the weight function either "uniform", "cubic", "quartic", "gaussian"

    bool maxentprint = true; // print information

    Maxent maxent(dim, prior, MaxIter, ctol, maxentprint, gamma, hnode);
    maxent.BasicFunc(NodesContribute,point);
    maxent.CheckConsistency();

    return 0;
}
