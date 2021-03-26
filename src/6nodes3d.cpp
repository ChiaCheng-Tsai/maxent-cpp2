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
     a 3D cloud of 6 nodes
    */

    int dim=3; //spatial dimension

    int NNodesC = 6; //Number Nodes contribute

    VectorXd point(dim); //coordinates of the sample points p(x,y,z)
    point(0)=0.1;
    point(1)=0.1;
    point(2)=0.1;

    MatrixXd NodesContribute(NNodesC,dim); //Vector class Node contribute
    NodesContribute(0,0) =  0; NodesContribute(0,1) =  0; NodesContribute(0,2) =  0;
    NodesContribute(1,0) =  1; NodesContribute(1,1) =  0; NodesContribute(1,2) =  0;
    NodesContribute(2,0) =  0; NodesContribute(2,1) =  1; NodesContribute(2,2) =  0;
    NodesContribute(3,0) =  0; NodesContribute(3,1) =  0; NodesContribute(3,2) =  1;
    NodesContribute(4,0) =  1; NodesContribute(4,1) =  1; NodesContribute(4,2) =  0;
    NodesContribute(5,0) =  1; NodesContribute(5,1) =  0; NodesContribute(5,2) =  1;

    int MaxIter=100; //maximum number of iteration

    double ctol= 1E-10; //convergence tolerance

    double gamma=3; // 3: for compare with fortran code //4.2426; // parameter that controls the support size of the basic function

    double hnode = 1; //characteristic nodal spacing

    string prior = "cubic"; // prior in the weight function either "uniform", "cubic", "quartic", "gaussian".

    bool maxentprint = true; // print total information

    Maxent maxent(dim, prior, MaxIter, ctol, maxentprint, gamma, hnode);
    maxent.BasicFunc(NodesContribute,point);
    maxent.CheckConsistency();

    return 0;
}
