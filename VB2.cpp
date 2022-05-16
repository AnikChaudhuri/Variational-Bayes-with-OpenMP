#include <iostream>
#include <omp.h>
#include <random>


std::default_random_engine generator;



void calc1(double* mean, int dim, int gene, double lambdan, double an, double bn, double mu_n, double lambda0, int i, double* B, double* L){
    
    lambdan = (lambda0+(dim*gene))*(an/bn);
    L[0] = lambdan;
    std::normal_distribution<double> dist(mu_n, 1/lambdan);
    mean[i] = dist(generator);
    
}

void calc2(double* var, double b, double mu0, double xb, int dim, double x_sq, int gene, double lambdan, double an, double bn, 
            double mu_n, double lambda0, int i, double* B, double* L){
    
    bn = b +0.5*((lambda0+(dim*gene))*(1./lambdan + (mu_n*mu_n))-2.0*(lambda0*mu0 + xb)*mu_n+(x_sq)+(lambda0*mu0*mu0));
    B[0] = bn;
    std::gamma_distribution<double> dist1(an, 1/bn);
    var[i] = sqrt(1/dist1(generator));

    
}

int main(){
    int gene = 30;
    int dim = 2;
    double* ex_a = new double[dim*gene]; double* mean = new double[1000]; double* var = new double[1000];
    double* B = new double[1]; double* L = new double[1];
    
    std::normal_distribution<double> distribution(15.0,2.0);
   
    
    #pragma omp parallel
    {
        #pragma omp for
        for(int i = 0; i < dim*gene; i++){ex_a[i] = distribution(generator);} 

    }
    
    //for(int i = 0; i < dim*gene; i++) std::cout << ex_a[i] <<std::endl;
    double xb = 0; double x_sq = 0; double a = 0.3; double lambda0 = 1; double mu0 = 2; double b = 2; double lambdan = 3;

    double an = a + ((dim*gene)+1)/2.0; 
    double bn=2;
    double Mean, Var;

    #pragma omp parallel
    {
        #pragma omp for reduction(+:xb, x_sq)
    
        for(int i = 0; i< dim*gene; i++){
        xb = xb + ex_a[i];
        x_sq = x_sq + (ex_a[i]*ex_a[i]);
        }
    
    }
    double xbar = xb/(dim*gene);
    double mu_n = (lambda0*mu0 + dim*gene*xbar)/(lambda0 + (dim*gene));

    for(int i =0; i < 1000; i++){
            #pragma omp parallel sections
            {
            #pragma omp section
        {
            calc1( mean, dim, gene, lambdan, an, bn, mu_n, lambda0, i, B, L);
        }

        #pragma omp section
        {
            calc2(var, b, mu0, xb, dim, x_sq, gene, lambdan, an, bn, mu_n, lambda0,i, B, L);
        }
        
            }
            lambdan = L[0];
            bn = B[0];
    }
    
    
    #pragma omp parallel
    {
        #pragma omp single
        std::cout<<"num of threads "<< omp_get_num_threads()<< std::endl;

        #pragma omp for reduction(+:Mean, Var)
        for(int k = 0; k < 1000; k++){
            Mean = (Mean + mean[k]);
            Var = Var + var[k];
        }

        #pragma omp single
        std::cout<<"mean: "<< Mean/1000 <<" and sd: " << Var/1000 << std::endl;

    }
    
    //for(int i =0; i < 1000; i++)std::cout<<"mean is "<< mean[i] <<" and variance is " << var[i] << std::endl;
    

}