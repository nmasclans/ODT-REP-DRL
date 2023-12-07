/**
 * @file eigenDecomposition.cc
 * @brief Header file for class \ref eigenDecomposition
 */

#include "eigenDecomposition_.h"
//#include "domain.h"   // commented for testing compilation
#include <cmath>
#include <algorithm>

using namespace std;

// todo: include these methods to dv:dv_stress_tensor class

///////////////////////////////////////////////////////////////////////////////
/** eigenDecomposition initialization function
 *
 * @param p_domn  \input set domain pointer with.
 */

// void eigenDecomposition_::init(domain *p_domn) {    // commented for testing compilation
void eigenDecomposition_::init() {

//    domn    = p_domn; // commented for testing compilation

}

///////////////////////////////////////////////////////////////////////////////
/** Symmetric diagonalization of a 3D matrix
 * 
 * The diagonal matrix D = QT * A * Q;  and  A = Q*D*QT
 * obtained from: 
 *       http://stackoverflow.com/questions/4372224/
 *       fast-method-for-computing-3x3-symmetric-matrix-spectral-decomposition
 * 
 * @param A     \input matrix to diagonalize, must be a symmetric matrix
 * @param Q     \output matrix of eigenvectors
 * @param D     \output diagonal matrix of eigenvalues
 */

void eigenDecomposition_::sym_diagonalize(
    const vector<vector<double>> &A, vector<vector<double>> &Q, vector<vector<double>> &D) {

    const int maxsteps=24;
    int k0, k1, k2;
    double o[3], m[3];
    double q [4] = {0.0,0.0,0.0,1.0};
    double jr[4];
    double sqw, sqx, sqy, sqz;
    double tmp1, tmp2, mq;
    double AQ[3][3];
    double thet, sgn, t, c;
    
    for(int i=0;i < maxsteps;++i) {
        // quat to matrix
        sqx      = q[0]*q[0];
        sqy      = q[1]*q[1];
        sqz      = q[2]*q[2];
        sqw      = q[3]*q[3];
        Q[0][0]  = ( sqx - sqy - sqz + sqw);
        Q[1][1]  = (-sqx + sqy - sqz + sqw);
        Q[2][2]  = (-sqx - sqy + sqz + sqw);
        tmp1     = q[0]*q[1];
        tmp2     = q[2]*q[3];
        Q[1][0]  = 2.0 * (tmp1 + tmp2);
        Q[0][1]  = 2.0 * (tmp1 - tmp2);
        tmp1     = q[0]*q[2];
        tmp2     = q[1]*q[3];
        Q[2][0]  = 2.0 * (tmp1 - tmp2);
        Q[0][2]  = 2.0 * (tmp1 + tmp2);
        tmp1     = q[1]*q[2];
        tmp2     = q[0]*q[3];
        Q[2][1]  = 2.0 * (tmp1 + tmp2);
        Q[1][2]  = 2.0 * (tmp1 - tmp2);

        // AQ = A * Q
        AQ[0][0] = Q[0][0]*A[0][0]+Q[1][0]*A[0][1]+Q[2][0]*A[0][2];
        AQ[0][1] = Q[0][1]*A[0][0]+Q[1][1]*A[0][1]+Q[2][1]*A[0][2];
        AQ[0][2] = Q[0][2]*A[0][0]+Q[1][2]*A[0][1]+Q[2][2]*A[0][2];
        AQ[1][0] = Q[0][0]*A[0][1]+Q[1][0]*A[1][1]+Q[2][0]*A[1][2];
        AQ[1][1] = Q[0][1]*A[0][1]+Q[1][1]*A[1][1]+Q[2][1]*A[1][2];
        AQ[1][2] = Q[0][2]*A[0][1]+Q[1][2]*A[1][1]+Q[2][2]*A[1][2];
        AQ[2][0] = Q[0][0]*A[0][2]+Q[1][0]*A[1][2]+Q[2][0]*A[2][2];
        AQ[2][1] = Q[0][1]*A[0][2]+Q[1][1]*A[1][2]+Q[2][1]*A[2][2];
        AQ[2][2] = Q[0][2]*A[0][2]+Q[1][2]*A[1][2]+Q[2][2]*A[2][2];

        // D = Qt * AQ
        D[0][0] = AQ[0][0]*Q[0][0]+AQ[1][0]*Q[1][0]+AQ[2][0]*Q[2][0];
        D[0][1] = AQ[0][0]*Q[0][1]+AQ[1][0]*Q[1][1]+AQ[2][0]*Q[2][1];
        D[0][2] = AQ[0][0]*Q[0][2]+AQ[1][0]*Q[1][2]+AQ[2][0]*Q[2][2];
        D[1][0] = AQ[0][1]*Q[0][0]+AQ[1][1]*Q[1][0]+AQ[2][1]*Q[2][0];
        D[1][1] = AQ[0][1]*Q[0][1]+AQ[1][1]*Q[1][1]+AQ[2][1]*Q[2][1];
        D[1][2] = AQ[0][1]*Q[0][2]+AQ[1][1]*Q[1][2]+AQ[2][1]*Q[2][2];
        D[2][0] = AQ[0][2]*Q[0][0]+AQ[1][2]*Q[1][0]+AQ[2][2]*Q[2][0];
        D[2][1] = AQ[0][2]*Q[0][1]+AQ[1][2]*Q[1][1]+AQ[2][2]*Q[2][1];
        D[2][2] = AQ[0][2]*Q[0][2]+AQ[1][2]*Q[1][2]+AQ[2][2]*Q[2][2];
        o[0]    = D[1][2];
        o[1]    = D[0][2];
        o[2]    = D[0][1];
        m[0]    = std::abs(o[0]);
        m[1]    = std::abs(o[1]);
        m[2]    = std::abs(o[2]);

        k0      = (m[0] > m[1] && m[0] > m[2])?0: (m[1] > m[2])? 1 : 2; // index of largest element of offdiag
        k1      = (k0+1)%3;
        k2      = (k0+2)%3;
        if (o[k0]==0.0) {
            break;  // diagonal already
        }

        thet    = (D[k2][k2]-D[k1][k1])/(2.0*o[k0]);
        sgn     = (thet > 0.0)?1.0:-1.0;
        thet   *= sgn; // make it positive
        t       = sgn /(thet +((thet < 1.E6)? std::sqrt(thet*thet+1.0):thet)) ; // sign(T)/(|T|+sqrt(T^2+1))
        c       = 1.0/std::sqrt(t*t+1.0); //  c= 1/(t^2+1) , t=s/c 
        if(c==1.0) {
            break;  // no room for improvement - reached machine precision.
        }

        jr[0 ]  = jr[1] = jr[2] = jr[3] = 0.0;
        jr[k0]  = sgn*std::sqrt((1.0-c)/2.0);  // using 1/2 angle identity sin(a/2) = std::sqrt((1-cos(a))/2)  
        jr[k0] *= -1.0; // since our quat-to-matrix convention was for v*M instead of M*v
        jr[3 ]  = std::sqrt(1.0f - jr[k0] * jr[k0]);
        if(jr[3]==1.0) {
            break; // reached limits of floating point precision
        }

        q[0]    = (q[3]*jr[0] + q[0]*jr[3] + q[1]*jr[2] - q[2]*jr[1]);
        q[1]    = (q[3]*jr[1] - q[0]*jr[2] + q[1]*jr[3] + q[2]*jr[0]);
        q[2]    = (q[3]*jr[2] + q[0]*jr[1] - q[1]*jr[0] + q[2]*jr[3]);
        q[3]    = (q[3]*jr[3] - q[0]*jr[0] - q[1]*jr[1] - q[2]*jr[2]);
        mq      = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
        q[0]   /= mq;
        q[1]   /= mq;
        q[2]   /= mq;
        q[3]   /= mq;
    }

}

///////////////////////////////////////////////////////////////////////////////
/** Reconstruct matrix from eigen decomposition of a 3D matrix
 * 
 * The diagonal matrix D = QT * A * Q;  and  A = Q*D*QT
 * 
 * @param D     \input diagonal matrix of eigenvalues
 * @param Q     \input matrix of eigenvectors
 * @param A     \output reconstructed symmetric matrix
 */

void eigenDecomposition_::reconstruct_matrix_from_decomposition(
  const vector<vector<double>> &D, const vector<vector<double>> &Q, vector<vector<double>> &A)
{

    // A = Q*D*QT
    vector<vector<double>> QT(3, vector<double>(3, 0.0));
    vector<vector<double>> B(3, vector<double>(3, 0.0));

    // compute QT
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            QT[j][i] = Q[i][j];
        }
    }

    //mat-vec, B = Q*D
    matrix_matrix_multiply(Q,D,B);

    // mat-vec, A = (Q*D)*QT = B*QT
    matrix_matrix_multiply(B,QT,A);

}

///////////////////////////////////////////////////////////////////////////////
/**
 * Sort eigenvalues and eigenvectors 
 * 
 * Sort eigenvalues in decreasing order (and eigenvectors equivalently), with
 * Qij[0][0] >= Qij[1][1] >= Qij[2][2]
 * 
 * @param Qij   \inputoutput matrix of eigenvectors, with Qij[0][:] the first eigenvector
 * @param Dij   \inputoutput diagonal matrix of eigenvalues
 * 
 */

void eigenDecomposition_::sortEigenValuesAndEigenVectors(vector<vector<double>> &Qij, vector<vector<double>> &Dij){

    // create pairs of indices and eigenvalues for sorting
    vector<pair<double,size_t>> sortedEigVal(3);
    for (size_t i = 0; i < 3; ++i) {
        sortedEigVal[i] = {Dij[i][i], i}; // {value, idx} pair
    }

    // sort eigenvalues  in descending order
    std::sort(sortedEigVal.begin(), sortedEigVal.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;
    });

    // Rearrange eigenvalues and eigenvectors based on the sorted indices
    vector<vector<double>> tempQij(3, vector<double>(3, 0.0));
    vector<vector<double>> tempDij(3, vector<double>(3, 0.0));
    for (size_t i = 0; i < 3; ++i) {
        tempDij[i][i] = Dij[sortedEigVal[i].second][sortedEigVal[i].second];
        for (size_t j = 0; j < 3; j++){
            tempQij[j][i] = Qij[j][sortedEigVal[i].second];
        }
    }

    // update matrices
    Qij = tempQij;
    Dij = tempDij;

}


///////////////////////////////////////////////////////////////////////////////
/** Matrix-Matrix Multiplication 3D
 * 
 * Matrix multiplication C = A*B
 * 
 * @param A     \input matrix
 * @param B     \input matrix
 * @param C     \output matrix result of matrix-matrix multiplication 
 */
//--------------------------------------------------------------------------
//-------- matrix_matrix_multiply 3D ---------------------------------------
//--------------------------------------------------------------------------
void eigenDecomposition_::matrix_matrix_multiply(
  const vector<vector<double>> &A, const vector<vector<double>> &B, vector<vector<double>> &C)
{

    // Perform multiplication C = A*B
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double sum = 0;
            for (int k = 0; k < 3; ++k) {
                sum = sum + A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }

}