// compile and run as:
// $ g++ eigenDecomposition_test.cc eigenDecomposition_testclass.cc -o eigenDecomposition_test

#include "eigenDecomposition_testclass.h"
#include <iostream>
#include <cmath>

using namespace std;

void printMatrix(const double matrix[3][3]) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            cout << matrix[i][j] << " ";
        }
        cout << "\n";
    }
    cout << "\n";
}

int main() {
    eigenDecomposition_testclass eigenDecomp;

    double A[3][3] = {{1.0, 2.0, 3.0}, {2.0, 4.0, 5.0}, {3.0, 5.0, 6.0}};
    double Q[3][3];
    double D[3][3];

    eigenDecomp.sym_diagonalize(A, Q, D);

    cout << "Original Matrix A:\n";
    printMatrix(A);
    cout << "Matrix Q:\n";
    printMatrix(Q);
    cout << "Matrix D:\n";
    printMatrix(D);

    // Reconstruct matrix A from Q and D
    double reconstructedA[3][3];
    eigenDecomp.reconstruct_matrix_from_decomposition(D, Q, reconstructedA);
    cout << "Reconstructed Matrix A:\n";
    printMatrix(reconstructedA);

    // sort eigen values in descending order
    vector<double> eigenValues = {D[2][2], D[1][1], D[0][0]};
    double avgEigenValues      = (eigenValues[0] + eigenValues[1] + eigenValues[2]) / 3.0;

    // Adjust the eigenvalues to ensure their sum is 0
    for (double &eigval : eigenValues){
        eigval -= avgEigenValues;
    }

    cout << "Eigen values: " << eigenValues[0] << ", " << eigenValues[1] << ", " << eigenValues[2] << endl;

    return 0;
}