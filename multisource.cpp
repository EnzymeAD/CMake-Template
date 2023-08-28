#include "myblas.h"
#include <assert.h>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// #define EIGEN_USE_BLAS

#include<Eigen/Core>

double dotabs(struct complex* alpha, struct complex* beta, int n) {
  struct complex prod = myblas_cdot(alpha, beta, n);
  return myblas_cabs(prod);
}

void __enzyme_autodiff(void*, ...);
int enzyme_const, enzyme_dup, enzyme_out;

using Eigen::MatrixXd;
using Eigen::VectorXd;

void foo(MatrixXd *m, VectorXd *v) { *v = *m * *v; }

int main(int argc, char *argv[]) {
  // int size = 50;
  //// VectorXf is a vector of floats, with dynamic size.
  // Eigen::VectorXf u(size), v(size), w(size);
  // u = v + w;

  MatrixXd m = MatrixXd::Random(30, 30);
  MatrixXd dm = MatrixXd::Random(30, 30);
  m = (m + MatrixXd::Constant(30, 30, 1.2)) * 50;
  std::cout << "m =" << std::endl << m << std::endl;
  VectorXd v = VectorXd::Random(30);
  VectorXd dv = VectorXd::Random(30);
  // v << 1, 2, 3;
  // std::cout << "m * v =" << std::endl << m * v << std::endl;

  __enzyme_autodiff((void *)foo, &m, &dm, &v, &dv);
  std::cout << "dm, dv: =" << std::endl << dm << std::endl << dv << std::endl;

  // int n = 3;
  // if (argc > 1) {
  //   n = atoi(argv[1]);
  // }

  // struct complex *A = (struct complex*)malloc(sizeof(struct complex) * n);
  // assert(A != 0);
  // for(int i=0; i<n; i++)
  //  A[i] = (struct complex){(i+1), (i+2)};

  // struct complex *grad_A = (struct complex*)malloc(sizeof(struct complex) *
  // n); assert(grad_A != 0); for(int i=0; i<n; i++)
  //  grad_A[i] = (struct complex){0,0};

  // struct complex *B = (struct complex*)malloc(sizeof(struct complex) * n);
  // assert(B != 0);
  // for(int i=0; i<n; i++)
  //  B[i] = (struct complex){-3-i, 2*i};

  // struct complex *grad_B = (struct complex*)malloc(sizeof(struct complex) *
  // n); assert(grad_B != 0); for(int i=0; i<n; i++)
  //  grad_B[i] = (struct complex){0,0};

  //__enzyme_autodiff((void*)dotabs, A, grad_A, B, grad_B, n);
  // printf("Gradient dotabs(A)[0] = %f\n", grad_A[0].r);

  return 0;
}
