#include "eigenDecomposition.h"
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
    eigenDecomposition eigenDecomp;

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

    return 0;
}