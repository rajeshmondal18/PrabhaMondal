#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<fftw3.h>
#include<omp.h>

#include "nbody.h"

float bin_size=0.2;  /*   in powers of 10   */

//*******************************************************************************
//                    global  variables from Nbody_comp 
//*******************************************************************************

float  vhh, // Hubble parameter in units of 100 km/s/Mpc
  vomegam, // Omega_matter; total matter density (baryons+CDM) parameter
  vomegalam, // Cosmological Constant 
  vomegab, //Omega_baryon
  sigma_8_present ,//  Last updated value of sigma_8 (Presently WMAP)
  vnn; // Spectral index of primordial Power spectrum

long N1,N2,N3;// box dimension (grid) 
int NF, // Fill every NF grid point 
  Nbin; // Number of bins to calculate final P(k) (output)

float   LL; // grid spacing in Mpc

long    MM; // Number of particles
// global variables  (calculated )

int zel_flag=1, // memory allocation for zel is 3 times that for nbody
  fourier_flag;//for fourier transfrom
float  DM_m, // Darm matter mass of simulation particle in 10^10 M_sun h^-1 unit 
  norm, // normalize Pk
  pi=M_PI;

io_header    header1;

// arrays for storing data
float ***ro; // for density/potential
fftwf_plan p_ro; // for FFT
fftwf_plan q_ro; // for FFT

//*******************************************************************************
//                    done global variables from Nbody_comp 
//*******************************************************************************

//*******************************************************************************
//                    For fitting function
//*******************************************************************************

float tcmb=2.728; // CMBR temperature 
float R,rho_c,rho,rho_z,kcsqinv;

//*******************************************************************************
//    this sigma_func() is different from the sigma_func() in the powerspec.c
//*******************************************************************************

float sigma_func(float x)
{
  float y;
  
  y=3.*(sin(R*x)-(R*x)*cos(R*x))*pow(R*x,-3);
  y=(x*x*Pk(x)*y*y);
  return(y);
}

//*******************************************************************************
//*******************************************************************************

float sig_dsigdr_func(float x)
{
  float w,w_prime,l,y;
  
  l=x*R;
  w=3.*(sin(l)-(l)*cos(l))*pow(l,-3);
  w_prime=(-9./pow(l,4.))*(sin(l)-l*cos(l)) +  3.*sin(l)/pow(l,2.)  ;
  y=(x*x*x*Pk(x)*w*w_prime);
  return(y);
}

//*******************************************************************************
//*******************************************************************************


void mass(float vaa)
{
  float DD,M,sigma_sq,ln_sig_inv,mass_func,dn_dlnsigin,sig_dsigdr,dlogMdr,dlnsiginv_dlogM,dn_dlogM;
  
  float A_st=0.322,a_st=0.707,p_st=0.3;
  
  //float A_st=0.353,a_st=0.73,p_st=0.175; //another parameter set for Jenkins mass function
  
  float delta_c=1.68647, p_nu,nu;
  
  int N,flag,i;
  int charac;
  FILE *out;
  
  //*******************************************************************************
  
  rho_c=2.7755*vhh*vhh*(1.e11); //rho_c in units of M_sun/Mpc^3
  rho=rho_c*vomegam; //comoving density in units of M_sun/Mpc^3
  rho_z=pow(vaa,-3.)*rho; //rho(z) in units of M_sun/Mpc^3
  DD=Df(vaa); // growth factor of density perturbation at this z
  
  printf("vaa=%e omega_m=%e omega_lam=%e rho_c=%e rho=%e rho_z=%e\n",vaa,vomegam,vomegalam,rho_c,rho,rho_z);
  
  //*******************************************************************************
  
  out=fopen("mass_func_fit","w");
  for(i=0;i<20;i++)
    {
      M=pow(10,((i*0.5)+7.)); // mass in units of M_sun
      R=pow((3.*M/(4.*pi*rho)),(1.0/3.0)); // comoving radius corresponding to mass M in Mpc 
      
      sigma_sq=simp(sigma_func,0.00001,20./R,100000)*DD*DD; // value of sigma(R)^2
      
      sig_dsigdr=simp(sig_dsigdr_func,0.00001,20./R,100000)*DD*DD; // value of sigma*d(sigma)/dR
      
      dlnsiginv_dlogM=-sig_dsigdr*(R/3.)/sigma_sq;// d(ln sigma^-1)/d(log M)
      
      nu=delta_c*delta_c*a_st/sigma_sq;
      
      mass_func=A_st*sqrt(2*a_st/pi)*(1.+pow(nu,-1.*p_st))*(delta_c/sqrt(sigma_sq))*exp(-1.*nu/2.);// Sheth and Tormen Mass Function as eq(7) of Jenkins et.al. 2001 MNRAS
      
      //mass_func=0.315*exp(-pow(fabs(-0.5*log(sigma_sq)+0.61),3.8)); // fitting formula for mass function as eq(9) of Jenkins et.al. 2001 MNRAS
      
      dn_dlogM=mass_func*(rho/M)*dlnsiginv_dlogM; // dn/d(log M)
      
      fprintf(out,"%e\t%e\t%e\t%e\t%e\t%e\n",R,sigma_sq,sig_dsigdr,sqrt(nu),M*vhh,dn_dlogM*pow(vhh,-3.));
    }
  fclose(out);
}

//*******************************************************************************
//                       done fitting function
//*******************************************************************************


//*******************************************************************************
//                       main function for binning
//*******************************************************************************

void main()
{
  float vaa;
  int i, n_bin= 0;
  long totcluster, ii, tmp;
  double x, lowest, highest, box;
  int output_flag;
  float **halo;
  
  FILE *write1, *plot1;
  
  //*******************************************************************************
  
  //reading the halo catalogue
  read_fof("../halo_catalogue_9.000",1,&output_flag,&totcluster,halo,&vaa);
  halo= allocate_float_2d(totcluster,7);
  read_fof("../halo_catalogue_9.000",2,&output_flag,&totcluster,halo,&vaa);
  
  /*-------------------------------------------------------------------------------*/
  
  printf(" totcluster=%ld\n N1=%ld\n LL=%e\n DM_m=%e\n vaa=%e\n vhh=%e\n vomegam=%e\n vomegalam=%e\n vomegab=%e\n sigma_8_present=%e\n ", totcluster, N1, LL, DM_m,  vaa, vhh, vomegam, vomegalam, vomegab, sigma_8_present);
  
  /*-------------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------*/
  /*                           initialize power spectrum                 */
  /*---------------------------------------------------------------------*/
  
  
  TFset_parameters(vomegam*vhh*vhh,vomegab/vomegam,tcmb);
  
  /*---------------------------------------------------------------------*/
  /*                           done intitializing power spectrum         */
  /*---------------------------------------------------------------------*/
  /*              normalizing  the power spectrum  using sigma_8         */
  /*---------------------------------------------------------------------*/
  norm=1.;
  R=8./vhh; // this need because, we have used different sigma_func() than in the powerspec.c
  norm=simp(sigma_func,0.00001,3.5,100000);
  norm=pow(sigma_8_present,2.)/norm;       //normalization factor for Pk(k)
  /*---------------------------------------------------------------------*/
  /*              Normalization of powerspectrum done                    */
  /*---------------------------------------------------------------------*/
  
  
  mass(vaa); //for fitting mass function //very inportant
  
  /*-------------------------------------------------------------------------------*/
  
  
  box=N1*N2*N3*pow((vhh*LL),3.0);   /*   box size (provided in Mpc*h^-1)^3  */
  
  /*-------------------------------------------------------------------------------*/
  
  for(ii=0; ii<totcluster; ii++)
    halo[ii][0]= log10(halo[ii][0]*DM_m)+ 10.0; /* masses are converted from 10^10*M_sun*h^-1 unit to M_sun*h^-1 unit*/
  
  
  highest= halo[0][0]; /* masses are converted from 10^10*M_sun*h^-1 unit to M_sun*h^-1 unit (in log scale)*/
  lowest= halo[(totcluster-1)][0]; /* masses are converted from 10^10*M_sun*h^-1 unit to M_sun*h^-1 unit (in log scale)*/
  printf("log(lowest)= %e,\tlog(highest)= %e\n", lowest, highest);
  
  /*-------------------------------------------------------------------------------*/
  
  printf("Bin size= %.2f\n", bin_size);
  /* x= lowest+ bin_size/2.0; */
  /* while(x<= highest) */
  /*   { */
  /*     n_bin++; */
  /*     x+= bin_size; */
  /*   } */
  n_bin = (highest - lowest + 0.02)/bin_size;
  printf("Number of bins= %d\n", n_bin);
  /*-------------------------------------------------------------------------------*/
  
  long *bin;
  double *mass;
  
  bin=calloc((size_t)n_bin,sizeof(long));
  mass=calloc((size_t)n_bin,sizeof(double));

  for(i=0;i<n_bin+1;i++)
    {
      bin[i]=0;
      mass[i]=0.0;
    }
  
  for(ii=0; ii<totcluster; ii++)
    {
      i=(long)floor((halo[ii][0]-lowest)/bin_size);
      bin[i]++;
      mass[i]=mass[i]+pow(10,(halo[ii][0]))+10.0;
    }

  /*-------------------------------------------------------------------------------*/
  tmp = 0;
  write1 = fopen("mass_func_out", "w");
  
  for(i=0;i<n_bin;i++)
    {
      tmp=tmp+bin[i];
      
      //fprintf(write1, "%ld\n", bin[i]);
      fprintf(write1, "%e\t %e\t %e\t %e\t %e\t %ld\n", mass[i]/(1.0*bin[i]), bin[i]*1.0/(box*bin_size*log(10)), pow(10, (lowest + i*bin_size)),pow(10, (lowest + (i+1)*bin_size)), sqrtf(bin[i]*1.0)/(box*bin_size*log(10)), bin[i]);
    }
  fclose(write1);
  
  /*first columb is the (M in h^-1 M_sun) bin position (in x-axis),
    second columb is the no. of halo per unit vol per unit mass range dn/d(lnM)(in y-axis)
    [we devide by log(10)=ln(10) to convert dn/d(logM) to dn/d(lnM)]*/
  /*(3rd-4th) is the error in x and 5th is error in y*/

  /*-------------------------------------------------------------------------------*/

  printf("%ld\t%ld\n",totcluster,tmp);
  
  
  //for gnuplot (next two lines only for plot, you can skip)
  plot1= popen("gnuplot", "w");	/*file pointer for gnuplot*/
  fprintf(plot1, "set term postscript eps enhanced \n set output \"massfunc25.0.eps\" \n set logscale\n set grid\n set xlabel \"M (in M_{{/=12 O}&{/*-.766 O}{/=12 \267}} h^{-1})\"\n set ylabel \"dn/d(lnM) (in h^{3}Mpc^{-3})\" \n set xrange [(7.1e+07):(3.1e+11)]\n p \"mass_func_fit\" u 5:6 ti \"Theoretical mass function\" w l, \"mass_func_out\" u 1:2:($3):($4):($2-$5):($2+$5) ti \"Results from simulation \" pt 9 ps 1.1 w xyerrorbars\n");

}
