// static int klo = -1 ;
// static int khi = -1 ;

int spline( double *x   , double *y   , int  n   , 
                    double  yp1 , double  ypn , double *y2 );

int splint( double *xa , double *ya , double *y2a , 
                    int n    , double x   , double *y   );

int spline1_c( double *xi , double *yi , int *nin  ,
                double *xo , double *yo , int *nout );