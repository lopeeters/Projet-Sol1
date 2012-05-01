#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <GL/glfw.h>

void tsunamiCompute(double dt, int nmax, int sub, int order)
{;}

void tsunamiAnimate()
{;}

double tsunamiInitialConditionOkada(double x, double y)
{
  	double R = 6371220;
  	double x3d = 4*R*R*x / (4*R*R + x*x + y*y);
  	double y3d = 4*R*R*y / (4*R*R + x*x + y*y);
  	double z3d = R*(4*R*R - x*x - y*y) / (4*R*R + x*x + y*y);
  	double lat = asin(z3d/R)*180/M_PI;
  	double lon = atan2(y3d,x3d)*180/M_PI;
  	double lonMin = 142;
    double lonMax = 143.75;
    double latMin = 35.9;
    double latMax = 39.5;
  	double olon = (lonMin+lonMax)/2;
  	double olat = (latMin+latMax)/2;
  	double angle = -12.95*M_PI/180; 
  	double lon2 = olon + (lon-olon)*cos(angle) + (lat-olat)*sin(angle);
  	double lat2 = olat - (lon-olon)*sin(angle) + (lat-olat)*cos(angle);
  	if ( lon2 <= lonMax && lon2 >= lonMin && 
         lat2 >= latMin && lat2 <= latMax )	
    		return 1.0;
  	else	return 0.0; 
}

void tsunamiWriteFile(const char *filename, double *X, int size)
{
    int i;
    FILE* file = fopen(filename,"w");
    fprintf(file, "Number of values %d \n", size);
    for (i = 0; i < size; ++i) {
        fprintf(file,"%le;",X[i]); 
        if (i%5 == 4) fprintf(file,"\n"); }    
    fclose(file);
}