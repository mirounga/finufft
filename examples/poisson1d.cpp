#include <finufft.h>

#include <vector>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex>
#include <functional>

using namespace std;

double xcoordinate(double);
double xcoordinate_derivative(double);
double right_side_function(double);
double exact_solution(double);
int poisson1d(int, double, function<double (double)> , function<double (double)> , function<double (double)>, function<double (double)>);

int main(int argc, char *argv[])
{
    int i, er;
    int start = 20, end = 120, step = 10;
    double nufft_acc = 1e-12;

    if (argc < 4) 
       cout  << "Running with default option: start = " << start << ", end = " << end << ", step = " << step << std::endl;

    if (argc == 4)
    {
        start = atoi(argv[1]);
        end = atoi(argv[2]);
        step = atoi(argv[3]);
        cout  << "Running with start = " << start << ", end = " << end << ", step = " << step << std::endl;

    }


    for(i = start; i <=end; i += step)
    {
        er = poisson1d(i, nufft_acc, xcoordinate, xcoordinate_derivative, right_side_function, exact_solution);
        if (er > 1)
        {
            cout << "Type-1 NUFFT failed - " << er;
        }
        if (er < -1)
        {
            cout << "Type-2 NUFFT failed - " << -er;
         }
    }
    return 0;
}



int poisson1d(int N, double acc, function<double (double)> xcoordinate, function<double (double)> xcoordinate_derivative, 
            function<double (double)> right_side_function, function<double (double)> exact_solution)
{
    
    int Nk = 0.5 * N;
    Nk = 2 * ceil(Nk / 2.); //  modes to trust due to quadr err - from 2d example, is it valid for 1d?

    int i;

    nufft_opts* opts = new nufft_opts;     // opts is pointer to struct
    finufft_default_opts(opts);

    complex<double> R = complex<double>(1.0,0.0);  // the real unit

    vector<double> t(N); // for uniform grid
    vector<double> x(N); // for nonuniform grid
    vector<double> dx(N); // for nonuniform grid derivative
    vector<complex<double> > c(N); // for strengths
    vector<complex<double> > result(N); 


    vector<complex<double> > fhat(Nk); //for Fourier modes
    vector<int> k(Nk);
    vector<double> kinv(Nk);
    vector<complex<double> > fhatweighted(Nk); 


 
    for(i = 0; i < N; i++)
    {
    // generate uniform grid
        t[i] = 2 * M_PI * i / N;
    // generate non-uniform grid
        x[i] = xcoordinate(t[i]);
    // derivative for nonuniform grid, Jacobian in general case
        dx[i] = xcoordinate_derivative(t[i]);
    // weights (aka strengths) defined by quadrature rule with equally spaced points 2*pi/n
    // 2*pi is missing since it cancels normalization coeff 1/(2*pi)
        c[i] = (right_side_function(x[i]) * dx[i] / N) * R ;
    }

    int ier1 = finufft1d1(N, &x[0], &c[0], -1, acc, Nk, &fhat[0], opts); // type-1 NUFFT
    if (ier1 > 1) return ier1;

    for(i = 0; i < Nk; i++)
    {
        k[i] = -Nk / 2 + i;
        if(k[i] == 0) 
            kinv[i] = 0;
        else 
            kinv[i] = 1. / (k[i] * k[i]);
        fhatweighted[i] = fhat[i] * kinv[i];
    }

    int ier2 = finufft1d2(N, &x[0], &result[0], +1, acc, Nk, &fhatweighted[0], opts); // type-2 NUFFT
    if (ier2 > 1) return -ier2;

    // checking the error, l_inf norm - do we want other one?
    double error = 0.;
    for (i = 0; i < N; i++) 
    {
        error = max(fabs(exact_solution(x[i]) -  result[i].real()), error); 
    }
    cout << "N = " << N << ", type-1 NUFFT - " << ier1 <<", type-2 NUFFT - " << ier2 << ", error - "<< error << endl ;  


 /*   for(i = 0; i < N; i++)
    {
        std::cout<< x[i] << ": " << t[i] << ": " << exact_solution(x[i]) << " " << result[i].real() << std::endl ;  
    }
*/
   return 0;
}



double xcoordinate(double t)
{
    return (t + 0.5 * sin(t));
}

double xcoordinate_derivative(double t)
{
    return (1 + 0.5 * cos(t));
}

double right_side_function(double x)
{
    return sin(x);
}

double exact_solution(double x)
{
    return sin(x);
}