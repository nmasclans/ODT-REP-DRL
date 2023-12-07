// compile and run as:
// $  g++ eigenDecomposition_.cc eigenDecomposition_test.cc -o eigenDecomposition_test.x; ./eigenDecomposition_test.x

#include "eigenDecomposition_.h"
#include <iostream>
#include <cmath>

using namespace std;

void printMatrix(const vector<vector<double>> matrix) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            cout << matrix[i][j] << " ";
        }
        cout << "\n";
    }
    cout << "\n";
}

int main() {

    eigenDecomposition_ eigenDecomp;

    vector<vector<double>> A(3, vector<double>(3, 0.0));
    vector<vector<double>> Q(3, vector<double>(3, 0.0));
    vector<vector<double>> D(3, vector<double>(3, 0.0));

    // A symmetric matrix for testing
    A[0][0] = 1.0;  A[0][1] = 2.0;  A[0][2] = 3.0;
    A[1][0] = 2.0;  A[1][1] = 4.0;  A[1][2] = 5.0;
    A[2][0] = 3.0;  A[2][1] = 5.0;  A[2][2] = 6.0;

    eigenDecomp.sym_diagonalize(A, Q, D);

    cout << "Original Matrix A:\n";
    printMatrix(A);
    cout << "Matrix Q:\n";
    printMatrix(Q);
    cout << "Matrix D:\n";
    printMatrix(D);

    // Reconstruct matrix A from Q and D
    vector<vector<double>> reconstructedA(3, vector<double>(3, 0.0));;
    eigenDecomp.reconstruct_matrix_from_decomposition(D, Q, reconstructedA);
    cout << "Reconstructed Matrix A:\n";
    printMatrix(reconstructedA);
    cout << "-> Note matrix A and reconstructed-A are the same!" << endl;
    cout << "---> This proves methods sym_diagonalize (eigen-decomposition) and reconstruct_matrix_from_decomposition are correct!" << endl << endl;

    // print eigenvalues
    cout << "--- Non-sorted eigenvalues ---" << endl;
    cout << "Non-sorted eigenvalues: "; 
    for (int i=0; i<3; i++){
        cout << D[i][i] << ", ";
    }
    cout << endl;
    cout << "Non-sorted eigenvectors:" << endl;
    printMatrix(Q);
    
    // print sorted eigenvalues in descending order
    eigenDecomp.sortEigenValuesAndEigenVectors(Q, D);
    cout << "--- Sorted eigenvalues (descending order) ---" << endl;
    cout << "Sorted eigenvalues: ";
    for (int i=0; i<3; i++){
        cout << D[i][i] << ", ";
    }
    cout << endl;
    cout << "Sorted eigenvectors:" << endl;
    printMatrix(Q);

    return 0;
}