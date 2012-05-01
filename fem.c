/*
 *  fem.c
 *  Library for MECA1120 : Finite Elements for dummies
 *
 *  Copyright (C) 2012 UCL-IMMC : Vincent Legat
 *  All rights reserved.
 *
 */

#include "fem.h"


static const int    _gaussTriangleNumber    = 3;
static const double _gaussTriangleXsi[3]    = {0.166666666666667,0.666666666666667,0.166666666666667};
static const double _gaussTriangleEta[3]    = {0.166666666666667,0.166666666666667,0.666666666666667};
static const double _gaussTriangleWeight[3] = {0.166666666666667,0.166666666666667,0.166666666666667};

static const int    _gaussEdgeNumber     	= 2;
static const double _gaussEdgeXsi[2]     	= {-5.773502691896257e-01,5.773502691896257e-01};
static const double _gaussEdgeWeight[2]  	= {1.0,1.0};


int    femGaussTriangleNumber(void)     	{ return _gaussTriangleNumber; }
double femGaussTriangleXsi(int i)       	{ return _gaussTriangleXsi[i]; }
double femGaussTriangleEta(int i)       	{ return _gaussTriangleEta[i]; }
double femGaussTriangleWeight(int i)    	{ return _gaussTriangleWeight[i]; }

int    femGaussEdgeNumber(void)     		{ return _gaussEdgeNumber; }
double femGaussEdgeXsi(int i)       		{ return _gaussEdgeXsi[i]; }
double femGaussEdgeWeight(int i)    		{ return _gaussEdgeWeight[i]; }

int    femShapesTriangleNumber(void)        { return 3; }
int    femShapesEdgeNumber(void)         	{ return 2; }

femMesh *femMeshNew(void)
{
	femMesh *myMesh = malloc(sizeof(femMesh));
	myMesh->elem = NULL; 
	myMesh->nElem = 0;
	myMesh->nNode = 0;
    return myMesh;
}

void femMeshFree(femMesh *myMesh)  
{ 
    if (myMesh->elem != NULL) free(myMesh->elem);
    free(myMesh); 
}

femEdges *femEdgesNew(femMesh *myMesh)
{
	femEdges *myEdges = malloc(sizeof(femEdges));
	femMeshEdges(myMesh,myEdges);
    return myEdges;
}

void femEdgesFree(femEdges *myEdges)  
{ 
    if (myEdges->edges != NULL) free(myEdges->edges);
    free(myEdges);
}


void femMeshGmshRead(femMesh *theMesh, const char *filename)
{
    int i,trash,nNode,nElem,*elem;
    char buffer[256];
    
    FILE* file = fopen(filename,"r");
    if (file == NULL) Error("No mesh file !");
    
    // Reading nodes
    
    while(!feof(file)) {    fscanf(file,"%256s",buffer);
        if(strncmp(buffer,"$Nodes",6) == 0) break;   }
    fscanf(file, "%d", &nNode);
    theMesh->X = malloc(sizeof(double)*nNode);
    theMesh->Y = malloc(sizeof(double)*nNode);

    
    // Reading elements
    
    while(!feof(file)) {    fscanf(file,"%256s",buffer);
                            if(strncmp(buffer, "$Elements", 9) == 0) break;   }
    fscanf(file, "%d", &nElem);
    theMesh->elem = malloc(sizeof(int)*3*nElem);
    theMesh->nElem = nElem;
    for (i = 0; i < nElem; ++i) {
        elem = &(theMesh->elem[i*3]);
        fscanf(file,"%d %d %d %d %d %d %d %d", &trash,&trash,&trash,&trash,&trash,&elem[0],&elem[1],&elem[2]);   }

    fclose(file);
}

void femMeshRead(femMesh *theMesh, const char *filename)
{
    int i,trash,*elem;
    
    FILE* file = fopen(filename,"r");
    if (file == NULL) Error("No mesh file !");

    fscanf(file, "Number of nodes %d \n", &theMesh->nNode);
    theMesh->X = malloc(sizeof(double)*theMesh->nNode);
    theMesh->Y = malloc(sizeof(double)*theMesh->nNode);
    for (i = 0; i < theMesh->nNode; ++i) {
        fscanf(file,"%d : %le %le \n",&trash,&theMesh->X[i],&theMesh->Y[i]); }
    
    fscanf(file, "Number of elements %d \n", &theMesh->nElem); 
    theMesh->elem = malloc(sizeof(int)*3*theMesh->nElem);
    for (i = 0; i < theMesh->nElem; ++i) {
        elem = &(theMesh->elem[i*3]);
        fscanf(file,"%d : %d %d %d \n", &trash,&elem[0],&elem[1],&elem[2]);  }
    
    fclose(file);
}

void femMeshWrite(const femMesh *theMesh, const char *filename)
{
    int i,*elem;
    
    FILE* file = fopen(filename,"w");
    
    fprintf(file, "Number of nodes %d \n", theMesh->nNode);
    for (i = 0; i < theMesh->nNode; ++i) {
        fprintf(file,"%6d : %14.7e %14.7e \n",i+1,theMesh->X[i],theMesh->Y[i]); }
    
    fprintf(file, "Number of elements %d \n", theMesh->nElem);  
    for (i = 0; i < theMesh->nElem; ++i) {
        elem = &(theMesh->elem[i*3]);
        fprintf(file,"%6d : %6d %6d %6d \n", i,elem[0],elem[1],elem[2]);   }
    
    fclose(file);
}

int *femMeshElem(const femMesh *theMesh, const int i)
{
    return &(theMesh->elem[i*3]);
}

int femEdgeCompare(const void *edgeOne, const void *edgeTwo)
{
    int *nodeOne = ((femEdge*) edgeOne)->node;
    int *nodeTwo = ((femEdge*) edgeTwo)->node;  
    int  diffMin = Min(nodeOne[0],nodeOne[1]) - Min(nodeTwo[0],nodeTwo[1]);
    int  diffMax = Max(nodeOne[0],nodeOne[1]) - Max(nodeTwo[0],nodeTwo[1]);
    
    if (diffMin < 0)    return  1;
    if (diffMin > 0)    return -1;
    if (diffMax < 0)    return  1;
    if (diffMax > 0)    return -1; 
                        return  0;
}


void femEdgePrint(femEdges *theEdges)
{
    int i;    
    for (i = 0; i < theEdges->nEdge; ++i) {
        printf("%6d : %4d %4d : %4d %4d \n",i,
                theEdges->edges[i].node[0],theEdges->edges[i].node[1],
                theEdges->edges[i].elem[0],theEdges->edges[i].elem[1]); }
}
               
               

void femMeshEdges(femMesh *theMesh, femEdges *theEdges)
{
	int i,j,*elem;  
    int n = theMesh->nElem * 3;
    femEdge* edges = malloc(n * sizeof(femEdge));
	
    theEdges->mesh  = theMesh;
    theEdges->edges = edges;
    theEdges->nEdge = n;
    theEdges->nBoundary = n;
		
    for (i = 0; i < theMesh->nElem; i++) {
        elem = &(theMesh->elem[i*3]);
        for (j = 0; j < 3; j++) {
            int id = i * 3 + j;
            edges[id].elem[0] = i;
            edges[id].elem[1] = -1;
            edges[id].node[0] = elem[j];
            edges[id].node[1] = elem[(j + 1) % 3]; }}
	
    qsort(theEdges->edges, theEdges->nEdge, sizeof(femEdge), femEdgeCompare);

    int index = 0;          
    int nBoundary = 0;
    
    for (i=0; i < theEdges->nEdge; i++) {
      if (i == theEdges->nEdge - 1 || femEdgeCompare(&edges[i],&edges[i+1]) != 0) {
              edges[index] = edges[i];
              nBoundary++; }
      else {  edges[index] = edges[i];
              edges[index].elem[1] = edges[i+1].elem[0];
              i = i+1;}
      index++; }
      
    theEdges->edges = realloc(edges, index * sizeof(femEdge));
    theEdges->nEdge = index;
    theEdges->nBoundary = nBoundary;
	
}


void femGridRead(femGrid *theGrid,const char *filename)
{
    int i,j,nx,ny;
    float missingData;
    
    FILE* file = fopen(filename,"r");
    if (file == NULL) Error("No grid file !");
    
    fscanf(file, "NCOLS        %d \n",&theGrid->nx); nx = theGrid->nx;
    fscanf(file, "NROWS        %d \n",&theGrid->ny); ny = theGrid->ny;
    fscanf(file, "XLLCENTER    %f \n",&theGrid->Ox);
    fscanf(file, "YLLCENTER    %f \n",&theGrid->Oy);
    fscanf(file, "CELLSIZE     %f \n",&theGrid->dx);
    fscanf(file, "NODATA_VALUE %f \n",&missingData);
    
    theGrid->elem = malloc(sizeof(int*) * nx);
    theGrid->elem[0] = malloc(sizeof(int) * nx * ny);
    for (i=1 ; i < nx ; i++) 
        theGrid->elem[i] = theGrid->elem[i-1] + ny;
    
    for (j = 0; j < ny; j ++)
        for (i = 0; i < nx; i++)
            fscanf(file,"%f",&theGrid->elem[i][ny-j-1]);
    
    //
    // -!- La première valeur dans les données correspond au max lat et min lon.
    //      
        
    fclose(file);
}

femMatrix *femMatrixNew(int size)
{
	femMatrix *myMatrix = malloc(sizeof(femMatrix));
	myMatrix->A = malloc(sizeof(double*) * size); 
	myMatrix->A[0] = malloc(sizeof(double) * size * size);   
	myMatrix->size = size;
	
	int i;
	for (i=1 ; i < size ; i++) {
		myMatrix->A[i] = myMatrix->A[i-1] + size; }
	for (i=0 ; i < size*size; i++) 
		myMatrix->A[0][i] = 0.0;
    return myMatrix;
}

void femMatrixPrint(femMatrix *myMatrix)
{
	double **A  = myMatrix->A;
	int    size = myMatrix->size, i,j;
	for (i=0; i < size; i++) {
		for (j=0; j < size; j++)
			if (A[i][j] == 0)  printf("         ");   
			else               printf(" %+.10e",A[i][j]);
		printf("\n"); }
}
					  
void femMatrixFree(femMatrix *myMatrix)
{
	free(myMatrix->A[0]);
	free(myMatrix->A);
	free(myMatrix);
}

femVector *femVectorNew(int size)
{
	femVector *myVector = malloc(sizeof(femVector));
	myVector->X = malloc(sizeof(double) * size); 
	myVector->size = size;
	femVectorInit(myVector);
    return myVector;
}

void femVectorInit(femVector *myVector)
{
   int    size = myVector->size, i;
   for (i=0; i < size; i++)
		myVector->X[i] = 0.0;
}

void femVectorPrint(femVector *myVector)
{
	double *X  = myVector->X;
	int    size = myVector->size, i;
	for (i=0; i < size; i++) 
        printf(" %+.1e\n",X[i]); 
}

double femVectorNorm(const femVector *myVector)
{
	double *X  = myVector->X;
	int    size = myVector->size, i;
    double norm = 0.0;
	for (i=0; i < size; i++) 
        norm += X[i]*X[i]; 
    return sqrt(norm);
}

void femVectorFree(femVector *myVector)
{
	free(myVector->X);
	free(myVector);
}

void femVectorLinspace(femVector* myVector, double start, double end)
{
	double *X  = myVector->X;
	int    size = myVector->size, i;
    double h = (end-start)/(size-1);
	for (i=0; i < size; i++) 
        X[i] = start + i*h; 
}

void femVectorRead(femVector *theVector, const char *filename)
{
    int i;
    
    FILE* file = fopen(filename,"r");
    if (file == NULL) Error("No vector file !");

    fscanf(file, "Number of values %d \n", &theVector->size);
    for (i = 0; i < theVector->size; ++i) {
        fscanf(file,"%le;",&theVector->X[i]); 
        if (i%5 == 4) fscanf(file,"\n"); }
    fclose(file);
}

void femVectorWrite(const femVector *theVector, const char *filename)
{
    int i;
    
    FILE* file = fopen(filename,"w");
    if (file == NULL) Error("Cannot create the vector file !");

    fprintf(file, "Number of values %d \n", theVector->size);
    for (i = 0; i < theVector->size; ++i) {
        fprintf(file,"%le;",theVector->X[i]); 
        if (i%5 == 4) fprintf(file,"\n"); }    
    fclose(file);
}


void femFullSystemAlloc(femFullSystem *mySystem, int size)
{
	int i;	
	double *elem = malloc(sizeof(double) * size * (size+1)); 
	mySystem->A = malloc(sizeof(double*) * size); 
	mySystem->B = elem;
	mySystem->A[0] = elem + size;  
	mySystem->size = size;
	for (i=1 ; i < size ; i++) 
		mySystem->A[i] = mySystem->A[i-1] + size;
}

void femFullSystemInit(femFullSystem *mySystem)
{
	int i,size = mySystem->size;
	for (i=0 ; i < size*(size+1) ; i++) 
		mySystem->B[i] = 0;}


void femFullSystemPrint(femFullSystem *mySystem)
{
	double  **A, *B;
	int     i, j, size;
	
	A    = mySystem->A;
	B    = mySystem->B;
	size = mySystem->size;
	
	for (i=0; i < size; i++) {
		for (j=0; j < size; j++)
			if (A[i][j] == 0)  printf("         ");   
			else               printf(" %+.1e",A[i][j]);
		printf(" :  %+.1e \n",B[i]); }
}


double* femFullSystemEliminate(femFullSystem *mySystem)
{
	double  **A, *B, factor;
	int     i, j, k, size;
	
	A    = mySystem->A;
	B    = mySystem->B;
	size = mySystem->size;
	
	/* Gauss elimination */
	
	for (k=0; k < size; k++) {
		if ( A[k][k] <= 1e-8 ) {
			printf("Pivot index %d  ",k);
			printf("Pivot value %e  ",A[k][k]);
			Error("Cannot eliminate with such a pivot"); }
		for (i = k+1 ; i <  size; i++) {
			factor = A[i][k] / A[k][k];
			for (j = k+1 ; j < size; j++) 
				A[i][j] = A[i][j] - A[k][j] * factor;
			B[i] = B[i] - B[k] * factor; }}
	
	/* Back-substitution */
	
	for (i = size-1; i >= 0 ; i--) {
		factor = 0;
		for (j = i+1 ; j < size; j++)
			factor += A[i][j] * B[j];
		B[i] = ( B[i] - factor)/A[i][i]; }
	
	return(mySystem->B);	
}

void  femFullSystemConstrain(femFullSystem *mySystem, 
							 int myNode, double myValue) 
{
	double  **A, *B;
	int     i, size;
	
	A    = mySystem->A;
	B    = mySystem->B;
	size = mySystem->size;
	
	for (i=0; i < size; i++) {
		B[i] -= myValue * A[i][myNode];
		A[i][myNode] = 0; }
	
	for (i=0; i < size; i++) 
		A[myNode][i] = 0; 
	
	A[myNode][myNode] = 1;
	B[myNode] = myValue;
}

void femInverseDefPosMatrix(femMatrix *theMatrix, femMatrix *theInverse)
{
    double  **A, **B, factor;
    int     i, j, k, size;
    
    A    = theMatrix->A;
    B    = theInverse->A;
    size = theMatrix->size;
    if (theMatrix->size != theInverse->size)  Error("Incompatible matrices");
    for (k=0; k < size; k++)
        B[k][k] = 1.0;
    
    /* Gauss elimination */
    
    for (k=0; k < size; k++) {
        if ( A[k][k] <= 1e-8 ) {
            printf("Pivot index %d  ",k);
            printf("Pivot value %e  ",A[k][k]);
            Error("Cannot eliminate with such a pivot"); }
        for (i = k+1 ; i <  size; i++) {
            factor = A[i][k] / A[k][k];
            for (j = k+1 ; j < size; j++) 
                A[i][j] = A[i][j] - A[k][j] * factor;
            for (j = 0 ; j < size; j++)
                B[i][j] = B[i][j] - B[k][j] * factor; }}
    
    /* Back-substitution */
    
    for (i = size-1; i >= 0 ; i--) {
        for (k=0; k < size; k++) {
            factor = 0;
            for (j = i+1 ; j < size; j++)
                factor += A[i][j] * B[j][k];
            B[i][k] = ( B[i][k] - factor)/A[i][i]; }}
    
}

# ifndef NOINTEGRALSTRIANGLES

void femStommelAddIntegralsTriangles(femProblem *myProblem,femVector *vectorB)
{
     
    int 	sizeLoc = 3;
    int     sizeGlo = myProblem->mesh->nElem * sizeLoc + 1; 
    
    double *B = vectorB->X;
    double *E = &(myProblem->field->X[0]);
    double *U = &(myProblem->field->X[sizeGlo]);
    double *V = &(myProblem->field->X[sizeGlo*2]);
    
    double  h     = myProblem->height;
    double  g     = myProblem->gravity;
    double  L     = myProblem->L;
    double  gamma = myProblem->gamma;
    double  tau0  = myProblem->tau0;
    double  rho   = myProblem->rho;
    double  f0    = myProblem->f0;
    double  beta  = myProblem->beta;

    double  xLoc[3],yLoc[3],dphidx[3],dphidy[3], phi[3];
    double  xsi,eta,weight,jac;
    double  y,u,v,e;
    double  f, tau;
    int     i,k,elem,mapElem[3];
    int mapBidon[3] = {0,1,2};
    
    for (elem=0; elem < myProblem->mesh->nElem; elem++) {
        femTriangleMap(myProblem,elem,mapElem);
        femTriangleCoordinates(myProblem,elem,xLoc,yLoc);
        jac = (xLoc[1] - xLoc[0]) * (yLoc[2] - yLoc[0]) - (yLoc[1] - yLoc[0]) * (xLoc[2] - xLoc[0]);
        dphidx[0] = (yLoc[1] - yLoc[2])/jac;
        dphidx[1] = (yLoc[2] - yLoc[0])/jac;
        dphidx[2] = (yLoc[0] - yLoc[1])/jac;
        dphidy[0] = (xLoc[2] - xLoc[1])/jac;
        dphidy[1] = (xLoc[0] - xLoc[2])/jac;
        dphidy[2] = (xLoc[1] - xLoc[0])/jac;        
        for (k=0; k < femGaussTriangleNumber(); k++) {
            xsi = femGaussTriangleXsi(k);
            eta = femGaussTriangleEta(k);
            weight = femGaussTriangleWeight(k);     
            femShapesTrianglePhi(phi,xsi,eta);
            u = femInterpolate(phi,U,mapElem,3);
            v = femInterpolate(phi,V,mapElem,3);
            e = femInterpolate(phi,E,mapElem,3);
            y = femInterpolate(phi,yLoc,mapBidon,3);
            
            f = f0 + beta*(y - L/2.0); 
            tau = tau0*sin(M_PI*(y-(L/2.0))/L);
            
            for (i=0; i < 3; i++) {
                /* ajouter 1a */
                B[mapElem[i]] += (u*h*dphidx[i] + v*h*dphidy[i])*jac*weight;
                /* ajouter 1b */
                B[mapElem[i]+sizeGlo] += (phi[i]*(f*v+tau/(rho*h) - gamma*u) + dphidx[i]*g*e)*jac*weight;
                /* ajouter 1c */
                B[mapElem[i]+2*sizeGlo] += (phi[i]*(-f*u-gamma*v) + dphidy[i]*g*e)*jac*weight;
            }
        }
    }
}

# endif
# ifndef NOINTEGRALSEDGES

void femStommelAddIntegralsEdges(femProblem *myProblem,femVector *vectorB)
{
    
    int 	sizeLoc = 3;
    int     sizeGlo = myProblem->mesh->nElem * sizeLoc + 1; 
    
    double *B = vectorB->X;
    double *E = &(myProblem->field->X[0]);
    double *U = &(myProblem->field->X[sizeGlo]);
    double *V = &(myProblem->field->X[sizeGlo*2]);
 
    double  h = myProblem->height;
    double  g = myProblem->gravity;

    
    double  xEdge[2],yEdge[2],phiEdge[2];
    double  xsi,weight,jac;
    double  eetoile,unetoile;
    double  uRight, uLeft, vRight, vLeft, eRight, eLeft;
    int     i,k,edge,mapEdge[2][2];
    

    for (edge=0; edge < myProblem->edges->nEdge; edge++) {
        femEdgeMap(myProblem,edge,mapEdge);
        femEdgeCoordinates(myProblem,edge,xEdge,yEdge);
        double dxdxsi = xEdge[1] - xEdge[0];
        double dydxsi = yEdge[1] - yEdge[0];
        double norm = sqrt(dxdxsi*dxdxsi + dydxsi*dydxsi);
        double normal[2] = {dydxsi/norm, -dxdxsi/norm};
        jac = norm / 2.0;
        for (k=0; k < femGaussEdgeNumber(); k++) {
            xsi = femGaussEdgeXsi(k);
            weight = femGaussEdgeWeight(k);
            femShapesEdgePhi(phiEdge,xsi);
            uLeft = femInterpolate(phiEdge,U,mapEdge[0],2);
            uRight = femInterpolate(phiEdge,U,mapEdge[1],2);
            vLeft = femInterpolate(phiEdge,V,mapEdge[0],2);
            vRight = femInterpolate(phiEdge,V,mapEdge[1],2);
            eLeft = femInterpolate(phiEdge,E,mapEdge[0],2);
               
            double betaLeft = uLeft*normal[0]+ vLeft*normal[1];
            double betaRight;
            if (myProblem->edges->edges[edge].elem[1] == -1) {
                betaRight = -betaLeft;
                eRight = eLeft;
            } else {
                betaRight = uRight*normal[0]+ vRight*normal[1];
                eRight = femInterpolate(phiEdge,E,mapEdge[1],2);
            }
            
            eetoile = (eLeft+eRight)/2.0 + sqrt(h/g)*(betaLeft-betaRight)/2.0;
            unetoile = (betaLeft+betaRight)/2.0 + sqrt(h/g)*(eLeft-eRight)/2.0;
            for (i=0; i < 2; i++) {
                /* ajouter 2a */
                B[mapEdge[0][i]] -= h*unetoile*phiEdge[i] *jac*weight; 
                B[mapEdge[1][i]] += h*unetoile*phiEdge[i] *jac*weight;
                /* ajouter 2b */
                B[mapEdge[0][i]+sizeGlo] -= normal[0]*g*eetoile*phiEdge[i] *jac*weight; 
                B[mapEdge[1][i]+sizeGlo] += normal[0]*g*eetoile*phiEdge[i] *jac*weight;
                /* ajouter 2c */
                B[mapEdge[0][i]+2*sizeGlo] -= normal[1]*g*eetoile*phiEdge[i] *jac*weight; 
                B[mapEdge[1][i]+2*sizeGlo] += normal[1]*g*eetoile*phiEdge[i] *jac*weight;
            }
        }
    }  
}

# endif

void femShapesTriangleNodes(double *xsi, double *eta) 
{
    xsi[0] = 0.0;     eta[0] = 0.0;
    xsi[1] = 1.0;     eta[1] = 0.0;
    xsi[2] = 0.0;     eta[2] = 1.0;}

void femShapesTrianglePrint(void)
{
    int i,j;
    int n = femShapesTriangleNumber();
    double xsi[n],eta[n],phi[n];
    
    femShapesTriangleNodes(xsi,eta);
    for (i=0; i< n; i++) {
        femShapesTrianglePhi(phi, xsi[i],eta[i]);
        printf(" - Node %d === %+2.1f %+2.1f ==== ",i,xsi[i],eta[i]);
        for (j=0; j< n; j++)
            printf(" %+2.1f ",phi[j]);
        printf("\n");  }
}

void femShapesTrianglePhi(double *phi, double xsi, double eta) 
{   
    phi[0] = (1.0 - xsi - eta);
    phi[1] =               xsi;  
    phi[2] =               eta;  
    
}

void femShapesTriangleDphi(double *dphidxsi, double *dphideta, double xsi, double eta) 
{   
    dphidxsi[0] = -1.0;
    dphidxsi[1] =  1.0;  
    dphidxsi[2] =  0.0;  
    dphideta[0] = -1.0;
    dphideta[1] =  0.0;  
    dphideta[2] =  1.0;     
}

void femShapesEdgeNodes(double *xsi) 
{
    xsi[0] = -1.0; 
    xsi[1] =  1.0; 
}

void femShapesEdgePrint(void)
{
    int i,j;
    int n = femShapesEdgeNumber();
    double xsi[n],phi[n];
    
    femShapesEdgeNodes(xsi);
    for (i=0; i< n; i++) {
        femShapesEdgePhi(phi, xsi[i]);
        printf(" - Node %d === %+2.1f  ==== ",i,xsi[i]);
        for (j=0; j< n; j++)
            printf(" %+2.1f ",phi[j]);
        printf("\n");  }
}

void femShapesEdgePhi(double *phi, double xsi) 
{   
    phi[0] = (1.0 - xsi)/2.0;
    phi[1] = (1.0 + xsi)/2.0;   
    
}

void femShapesEdgeDphi(double *dphidxsi, double xsi) 
{   
    dphidxsi[0] =  -1.0/2.0;
    dphidxsi[1] =   1.0/2.0;  
}

void femMassMatrixTriangle(femMatrix *theMatrix)
{
    int     i,j,k,n = femShapesTriangleNumber();
    double  phi[n],xsi,eta,weight;
    double  **A = theMatrix->A; 
    
    for (k=0; k < femGaussTriangleNumber(); k++) {
        xsi = femGaussTriangleXsi(k);
        eta = femGaussTriangleEta(k);
        weight = femGaussTriangleWeight(k);     
        femShapesTrianglePhi(phi,xsi,eta);
        for (i=0; i < n; i++) {
            for (j=0; j < n; j++) {
                A[i][j] += phi[i]*phi[j]*weight; }}}    
}

void femUpdateEuler(femProblem *myProblem, double dt)
{
    int  size = myProblem->field->size, i;
    double* U = myProblem->field->X;
    
 	femVector *f = femVectorNew(size); double* F = f->X;
    
    femComputeRightHandSide(myProblem,f);
    for (i=0; i < size; i++) 
        U[i] += dt * F[i];
    
    femVectorFree(f);
}

void femUpdateRungeKutta(femProblem *myProblem, double dt)
{
    int  size = myProblem->field->size, i,j;
    double* U = myProblem->field->X;
    
 	femVector *uold  = femVectorNew(size); double* Uold = uold->X;
    femVector *unew  = femVectorNew(size); double* Unew = unew->X;
    femVector *k     = femVectorNew(size); double* K = k->X;
    for (i=0; i < size; i++) {
        Uold[i] = U[i];
        Unew[i] = U[i]; }
    
    const int    nStage   = 4;
    const double beta[4]  = {0.0,     0.5,   0.5, 1.0  }; 
    const double gamma[4] = {1.0/6, 2.0/6, 2.0/6, 1.0/6};
    
    for(j = 0; j < nStage; j++) {
        for (i=0; i < size; i++) 
        	U[i] = Uold[i] + dt * beta[j] * K[i];
    	femComputeRightHandSide(myProblem,k);
    	for (i=0; i < size; i++) 
        	Unew[i] += dt * gamma[j] * K[i]; }
   
    for (i=0; i < size; i++) {
        U[i] = Unew[i]; }
    
    femVectorFree(uold);
    femVectorFree(unew);
    femVectorFree(k);
}

double femInterpolate(double *phi, double *U, int *map, int n)
{
    double u = 0.0; int i;
    for (i=0; i <n; i++)
        u += phi[i]*U[map[i]];
    return u;
}

void femEdgeMap(femProblem *myProblem, int index, int map[2][2])
{
    int i,j,k;
    
    for (j=0; j < 2; ++j) {
        int node = myProblem->edges->edges[index].node[j];
        for (k=0; k < 2; k++) {
            int elem = myProblem->edges->edges[index].elem[k];
            map[k][j] = (myProblem->mesh->nElem)*3;
            if (elem >= 0) {
                for (i=0; i < 3; i++) {
                    if (myProblem->mesh->elem[elem*3 + i] == node) {
                        map[k][j] = elem*3 + i;  }}}}}
}

void femEdgeCoordinates(femProblem *myProblem, int index, double x[2], double y[2])
{
    int j;
    
    for (j=0; j < 2; ++j) {
        int node = myProblem->edges->edges[index].node[j] - 1;
        x[j] = myProblem->mesh->X[node];
        y[j] = myProblem->mesh->Y[node]; }
}

void femTriangleMap(femProblem *myProblem, int index, int map[3])
{
    int j;
    for (j=0; j < 3; ++j) 
        map[j] = index*3 + j; 
}

void femTriangleCoordinates(femProblem *myProblem, int index, double x[3], double y[3])
{
    int j;
    
    for (j=0; j < 3; ++j) {
        int node = myProblem->mesh->elem[index*3 + j] - 1;
        x[j] = myProblem->mesh->X[node];
        y[j] = myProblem->mesh->Y[node]; }
}

void femTriangleMapCoordinates(femProblem *myProblem, int index, int map[3])
{
    int j;
    for (j=0; j < 3; ++j) 
        map[j] = myProblem->mesh->elem[index*3 + j] - 1; 
}

femProblem *femStommelNew(const char *meshFileName)
{
    femProblem *myProblem = malloc(sizeof(femProblem));
       
//
// More realistic parameters, if you want to try
//   
//   myProblem->f0      = 1e-4;
//   myProblem->gravity = 9.81;
//   myProblem->beta    = 2e-11;
//   myProblem->height  = 1000.0;
//   myProblem->rho     = 1000.0;
//   myProblem->gamma   = 1e-6;
//   myProblem->L       = 1e6;
//   myProblem->tau0    = 0.1;
//
 
    myProblem->f0       = 1.0;
    myProblem->gravity  = 1.0;
    myProblem->beta     = 5.0;
    myProblem->height   = 1.0;
    myProblem->rho      = 1.0;
    myProblem->gamma    = 1.0/3.0;
    myProblem->L        = 1.0;
    myProblem->tau0     = 1.0;
    myProblem->residual = 1.0;
              
    myProblem->mesh = femMeshNew();
    femMeshRead(myProblem->mesh,meshFileName);
    int i; for (i = 0; i < myProblem->mesh->nNode; ++i) {
        myProblem->mesh->X[i] = myProblem->mesh->X[i] * myProblem->L;
        myProblem->mesh->Y[i] = myProblem->mesh->Y[i] * myProblem->L;  }
    myProblem->edges = femEdgesNew(myProblem->mesh);
    
    int sizeLoc = femShapesTriangleNumber();
    femMatrix *A    = femMatrixNew(sizeLoc);
    myProblem->invA = femMatrixNew(sizeLoc);    
    femMassMatrixTriangle(A);
    femInverseDefPosMatrix(A,myProblem->invA);
    femMatrixFree(A);
    
    int sizeGlo = myProblem->mesh->nElem * sizeLoc + 1; 
    myProblem->field = femVectorNew(sizeGlo*3); 
    myProblem->dataFields = malloc(sizeof(femVector*) * 3);
    myProblem->dataFields[0] = femVectorNew(sizeGlo);
    myProblem->dataFields[1] = femVectorNew(sizeGlo);
    myProblem->dataFields[2] = femVectorNew(sizeGlo);
    
    double  xLoc[3],yLoc[3],u,v,e;
    int     elem,mapElem[3];

    for (elem=0; elem < myProblem->mesh->nElem; elem++)  {
        femTriangleMap(myProblem,elem,mapElem);
        femTriangleCoordinates(myProblem,elem,xLoc,yLoc);
        for (i=0; i < 3; i++) {
        	femStommel(xLoc[i],yLoc[i],myProblem,&u,&v,&e);
            myProblem->dataFields[0]->X[mapElem[i]] = e;
    		myProblem->dataFields[1]->X[mapElem[i]] = u;
            myProblem->dataFields[2]->X[mapElem[i]] = v;}}
    
    return myProblem;
}
    
void femStommelFree(femProblem *myProblem)
{
    femVectorFree(myProblem->field);
    femVectorFree(myProblem->dataFields[0]);
    femVectorFree(myProblem->dataFields[1]);
    femVectorFree(myProblem->dataFields[2]);
    free(myProblem->dataFields);
    femMatrixFree(myProblem->invA);
    femEdgesFree(myProblem->edges);
    femMeshFree(myProblem->mesh);
    free(myProblem);
}

void femComputeRightHandSide(femProblem *myProblem,femVector *F)
{
    femVector *B = femVectorNew((myProblem->field->size) + 1); 
    femStommelAddIntegralsTriangles(myProblem,B);
    femStommelAddIntegralsEdges(myProblem,B);
    femStommelMultiplyInverseMatrix(myProblem,B,F); 
    myProblem->residual = femVectorNorm(B);   
    femVectorFree(B);
}

void femStommelMultiplyInverseMatrix(femProblem *myProblem,femVector *vectorB,femVector *vectorF)
{
    double *B = vectorB->X;
    double *F = vectorF->X;
    double **invA = myProblem->invA->A;
    
    double  xLoc[3],yLoc[3],jac;
    int     i,j,elem,mapElem[3];
    int 	sizeLoc = 3;
    int     sizeGlo = myProblem->mesh->nElem * sizeLoc + 1; 
    
    for (elem=0; elem < myProblem->mesh->nElem; elem++) {
        femTriangleMap(myProblem,elem,mapElem);
        femTriangleCoordinates(myProblem,elem,xLoc,yLoc);
        jac = (xLoc[1] - xLoc[0]) * (yLoc[2] - yLoc[0]) - (yLoc[1] - yLoc[0]) * (xLoc[2] - xLoc[0]);
        for (i=0; i < 3; i++) { 
            for (j=0; j < 3; j++) {
                F[mapElem[i]            ] += invA[i][j] * B[mapElem[j]            ] / jac;
                F[mapElem[i] +   sizeGlo] += invA[i][j] * B[mapElem[j] +   sizeGlo] / jac;
                F[mapElem[i] + 2*sizeGlo] += invA[i][j] * B[mapElem[j] + 2*sizeGlo] / jac; }}}
 }

void femStommel(double x, double y, femProblem *myProblem, double *u, double *v, double *eta)
{
    //
    // Solution analytique de Stommel dans un carre [0,1]x[0,1]
    // Modelisation de l'elevation de l'ocean Atlantique dans un carre adimensionnel
    // Ce modele que l'on attribue generalement au grand oceanographe Henry M.
    // Stommel (1920-1992), est considere comme le premier modele qualitativement correct du Gulf Stream
    //
   
     
    const double delta = 1;
    double  tau0 = myProblem->tau0;
    double  L = myProblem->L;
    double  rho = myProblem->rho;
    double  f0 = myProblem->f0;
    double  beta = myProblem->beta;
    double  h = myProblem->height;
    double  g = myProblem->gravity;
    double  gamm = myProblem->gamma;
 
    
    double Y = (y/L - 0.5);
    double X = x/L;
    double epsilon = gamm / (L * beta);
    double Z1 = (-1 + sqrt(1 + (2 * M_PI * delta * epsilon) * (2 * M_PI * delta * epsilon))) / (2 * epsilon);
    double Z2 = (-1 - sqrt(1 + (2 * M_PI * delta * epsilon) * (2 * M_PI * delta * epsilon))) / (2 * epsilon);
    double D = ((exp(Z2) - 1) * Z1 + (1 - exp(Z1)) * Z2) / (exp(Z1) - exp(Z2));
    double f1 = M_PI / D * (1 + ((exp(Z2) - 1) * exp(X * Z1) + (1 - exp(Z1)) * exp(X * Z2)) / (exp(Z1) - exp(Z2)));
    double f2 = 1 / D* (((exp(Z2) - 1) * Z1 * exp(X * Z1) + (1 - exp(Z1)) * Z2 * exp(X * Z2)) / (exp(Z1) - exp(Z2)));
    
    eta[0] = D * tau0 * f0 * L / (M_PI * gamm * rho * delta * g * h) *
               ( - gamm / (f0 * delta * M_PI) * f2 * sin(M_PI * Y) 
                 + 1 / M_PI * f1 * (cos(M_PI * Y) * (1 + beta * Y) 
                 - beta / M_PI * sin(M_PI * Y) ) );
    u[0] = D * tau0 / (M_PI * gamm * rho * h) * f1 * sin(M_PI * Y);
    v[0] = D * tau0 / (M_PI * gamm * rho * delta * h) * f2 * cos(M_PI * Y);
}


double femMin(double *x, int n) 
{
    double myMin = x[0];
    int i;
    for (i=1 ;i < n; i++) 
        if (x[i] < myMin) myMin = x[i];
    return myMin;
}

double femMax(double *x, int n) 
{
    double myMax = x[0];
    int i;
    for (i=1 ;i < n; i++) 
        if (x[i] > myMax) myMax = x[i];
    return myMax;
}

void femError(char *text, int line, char *file)                                  
{ 
    printf("\n-------------------------------------------------------------------------------- ");
    printf("\n  Error in %s at line %d : \n  %s\n", file, line, text);
    printf("--------------------------------------------------------------------- Yek Yek !! \n\n");
    exit(69);                                                 
}

void femWarning(char *text, int line, char *file)                                  
{ 
    printf("\n-------------------------------------------------------------------------------- ");
    printf("\n  Warning in %s at line %d : \n  %s\n", file, line, text);
    printf("--------------------------------------------------------------------- Yek Yek !! \n\n");                                              
}
