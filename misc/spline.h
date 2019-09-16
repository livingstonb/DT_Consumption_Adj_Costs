static int klo = -1 ;
static int khi = -1 ;

static int spline( float *x   , float *y   , int  n   , 
                    float  yp1 , float  ypn , float *y2 );

static int splint( float *xa , float *ya , float *y2a , 
                    int n    , float x   , float *y   );

int spline1_c( float *xi , float *yi , int *nin  ,
                float *xo , float *yo , int *nout );