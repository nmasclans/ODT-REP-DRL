#include "dv_reynolds_stress.h"
#include <cstdlib>
#include <iostream>

using namespace std;

void testCartesianToBarycentricConversion(double &xmap1, double &xmap2) {
    
    dv_reynolds_stress testObj;
    testObj = dv_reynolds_stress();

    // Perform the conversion
    vector<double> lambda(3, 0.0);
    testObj.getBarycentricCoordFromCartesianCoord(xmap1, xmap2, lambda);
    // Perform the inverse conversion
    double xmap1R, xmap2R;
    testObj.getCartesianCoordFromBarycentricCoord(lambda, xmap1R, xmap2R);
    
    // Print the results
    cout << "\nOriginal Cartesian Coordinates: (" << xmap1 << ", " << xmap2 << ")" << endl;
    cout << "Barycentric Coordinates: (" << lambda[0] << ", " << lambda[1] << ", " << lambda[2] << ")" << endl;
    cout << "Recovered Cartesian Coordinates: (" << xmap1R << ", " << xmap2R << ")" << endl;
    cout << "CHECK TO-DO: if 'original' and 'recovered' cartesian coordinates are the same, then the conversion is correct!" << endl;
    
}

void testEnsureRealizabilityEigenvalues(double &xmap1, double &xmap2) {
    
    dv_reynolds_stress testObj;
    testObj = dv_reynolds_stress();
    testObj.enforceRealizabilityXmap(xmap1, xmap2);
    cout << "Realizable Cartesian Coordinates: (" << xmap1 << ", " << xmap2 << ")" << endl;

}

int main() {
    
    // Run the tests
    double xmap1, xmap2;
    cout << endl << endl << "Point A" << endl;
    xmap1 = -0.4; 
    xmap2 = 0.3;
    testCartesianToBarycentricConversion(xmap1, xmap2);
    testEnsureRealizabilityEigenvalues(xmap1, xmap2);

    cout << endl << endl << "Point B" << endl;
    xmap1 = 1.1; 
    xmap2 = 0.5;
    testCartesianToBarycentricConversion(xmap1, xmap2);
    testEnsureRealizabilityEigenvalues(xmap1, xmap2);

    
    cout << endl << endl << "Point C" << endl;
    xmap1 = -0.1; 
    xmap2 = 0.75;
    testCartesianToBarycentricConversion(xmap1, xmap2);
    testEnsureRealizabilityEigenvalues(xmap1, xmap2);


    cout << endl << endl << "Point D" << endl;
    xmap1 = 1.1; 
    xmap2 = 1.2;
    testCartesianToBarycentricConversion(xmap1, xmap2);
    testEnsureRealizabilityEigenvalues(xmap1, xmap2);


    cout << endl << endl << "Point E" << endl;
    xmap1 = 0.2; 
    xmap2 = -0.3;
    testCartesianToBarycentricConversion(xmap1, xmap2);
    testEnsureRealizabilityEigenvalues(xmap1, xmap2);


    cout << endl << endl << "Point F" << endl;
    xmap1 = 0.2; 
    xmap2 = -0.1;
    testCartesianToBarycentricConversion(xmap1, xmap2);
    testEnsureRealizabilityEigenvalues(xmap1, xmap2);


    cout << endl << endl << "Point G" << endl;
    xmap1 = 0.6; 
    xmap2 = 0.2;
    testCartesianToBarycentricConversion(xmap1, xmap2);
    testEnsureRealizabilityEigenvalues(xmap1, xmap2);

    return 0;
}