/* in allotarrays.c */
float **allocate_float_2d(long N1,int N2);
float  ***allocate_fftwf_3d(long N1,long N2,long N3);
float Hf(float aa),Df(float aa),ff(float aa);
void Setting_Up_Memory_For_Ro(float av);
void cic(float** rra);
void Get_Phi(int i);
void Update_v(float aa,float delta_aa,float **rra,float **vva);
void Update_x(float aa,float delta_aa,float **rra,float **vva);
void calpow(int f_flag,int Nbin,double* power, double* powerk, double* kmode,long *no);
void Zel_move_gradphi(float av,float **rra,float **vva);     
void grad_phi(int ix);
int write_output(char *fname,long int seed,int output_flag,float **rra,float **vva,float vaa);
int read_output(char *fname, int read_flag,long int *seed,int *output_flag,int *in_flag,float **rra,float **vva,float *aa);
void read_fof(char *fname, int read_flag,int *output_flag, long *totcluster, float **halo, float *aa);


//Functions required to generate the power spectrum by Hu and Eisentine
void TFset_parameters(float omega0hh, float f_baryon, float Tcmb); 
float TFfit_onek(float k, float *tf_baryon, float *tf_cdm);

float Pk(float kk); // power spectrum 
float sigma_func(float kk); // for calculating sigma 8
float simp(float (*func)(float),float a,float b,int N); // for integration

void delta_fill(long*); // fills value of phases for Fourier modes


void mass(float vaa);
float sig_dsigdr_func(float x);


typedef struct 
  {
    long      npart[6];
    double   mass[6];
    double   time;
    double   redshift;
    int     flag_sfr;
    int     flag_feedback;
    long    npartTotal[6];
    int     flag_cooling;
    int     num_files;
    double   BoxSize;
    double   Omega0;
    double   OmegaLambda;
    double   HubbleParam; 
    double   Omegab; 
    double   sigma_8_present;
    long  Nx;
    long  Ny;
    long  Nz;
    float LL;
    int output_flag;
    int in_flag;
    long int seed;
    char  fill[256- 6*4- 6*8- 2*8- 2*4- 6*4- 2*4 - 6*8 -6*4 -8];  /* fills to 256 Bytes */
  }  io_header;
