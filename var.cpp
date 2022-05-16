// This code implements the example shown in the Variational Bayes page on Wikipedia. Link: https://en.wikipedia.org/wiki/Variational_Bayesian_methods
// Normal samples were drawn from a normal distribution and stored in vector ex_a.
//These samples were used to estimate the mean and standard deviation by using Variational Bayes algorithm.
// The output is the mean and standard deviation calculated in each iteration.

#include <iostream>
#include <omp.h>
#include <random>



int main(){
    int gene = 30;
    int dim = 2;
    double* ex_a = new double[dim*gene]; double* mean = new double[1000]; double* var = new double[1000];
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(10.0,2.0);
    
    #pragma omp parallel
    {
        #pragma omp for
        for(int i = 0; i < dim*gene; i++){ex_a[i] = distribution(generator);} //generating samples

    }
    
    //for(int i = 0; i < dim*gene; i++) std::cout << ex_a[i] <<std::endl;
    double xb = 0; double x_sq = 0; double a = 0.3; double lambda0 = 1; double mu0 = 2; double b = 2; double lambdan = 3;

    double an = a + ((dim*gene)+1)/2.0; 
    double bn;
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


    #pragma omp parallel 
    {
        #pragma omp for
        for(int j = 0; j<1000; j++){
        bn = b +0.5*((lambda0+(dim*gene))*(1./lambdan + (mu_n*mu_n))-2.0*(lambda0*mu0 + xb)*mu_n+(x_sq)+(lambda0*mu0*mu0));
        lambdan = (lambda0+(dim*gene))*(an/bn);
        std::normal_distribution<double> dist(mu_n, 1/lambdan);
        std::gamma_distribution<double> dist1(an, 1/bn);

        mean[j] = dist(generator);
        var[j] = sqrt(1/dist1(generator));

        
        
        }
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
