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
     a 3D cloud of 8 nodes
    */

    int dim=3; //spatial dimension

    int NNodesC = 8; //Number Nodes contribute

    VectorXd point(dim); //coordinates of the sample points p(x,y,z)
    point(0)=0.25;
    point(1)=0.15;
    point(2)=0.05;

    MatrixXd NodesContribute(NNodesC,dim); //Vector class Node contribute
    NodesContribute(0,0) =  0; NodesContribute(0,1) =  0; NodesContribute(0,2) =  0;
    NodesContribute(1,0) =  1; NodesContribute(1,1) =  0; NodesContribute(1,2) =  0;
    NodesContribute(2,0) =  0; NodesContribute(2,1) =  1; NodesContribute(2,2) =  0;
    NodesContribute(3,0) =  1; NodesContribute(3,1) =  1; NodesContribute(3,2) =  0;
    NodesContribute(4,0) =  0; NodesContribute(4,1) =  0; NodesContribute(4,2) =  1;
    NodesContribute(5,0) =  1; NodesContribute(5,1) =  0; NodesContribute(5,2) =  1;
    NodesContribute(6,0) =  0; NodesContribute(6,1) =  1; NodesContribute(6,2) =  1;
    NodesContribute(7,0) =  1; NodesContribute(7,1) =  1; NodesContribute(7,2) =  1;

    int MaxIter=100; //maximum number of iteration

    double ctol= 1E-10; //convergence tolerance

    double gamma = 1; // parameter that controls the support size of the basic function

    double hnode = 1; //characteristic nodal spacing

    string prior = "uniform"; // prior in the weight function either "uniform", "cubic", "quartic", "gaussian".

    bool maxentprint = true; // print total information

    Maxent example(dim, prior, MaxIter, ctol, maxentprint, gamma, hnode);
    example.BasicFunc(NodesContribute,point);
    example.CheckConsistency();

    system("pause");
    return 0;
}
