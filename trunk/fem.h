/*
 *  fem.h
 *  Library for MECA1120 : Finite Elements for dummies
 *
 *  Copyright (C) 2012 UCL-IMMC : Vincent Legat
 *  All rights reserved.
 *
 */

# ifndef _FEM_H_
# define _FEM_H_
 
# include <stdio.h>
# include <stdlib.h>
# include <math.h>
# include <string.h>

# define Error(a)   femError(a,  __LINE__, __FILE__)
# define Warning(a) femWarning(a,  __LINE__, __FILE__)
# define Min(a,b)   ((a < b) ? a : b ) 
# define Max(a,b)   ((a > b) ? a : b )  
# define Abs(a)     ((a > 0) ? a : -(a))


typedef struct {
    double *B;
    double **A;
    int size;
} femFullSystem;

typedef struct {
    double **A;
    int size;
} femMatrix;

typedef struct {
    double *X;
    int size;
} femVector;

typedef struct {
    int *node;
    int size;
} femList;

typedef struct {
    int* elem;
    double *X;
    double *Y;
    int nElem;
    int nNode;
} femMesh;

typedef struct {
    float** elem;
    float Ox;
    float Oy;
    float dx;
    int nx;
    int ny;
} femGrid;

typedef struct {
    int elem[2];
    int node[2];
} femEdge;

typedef struct {
    femMesh *mesh;
    femEdge *edges;
    int nEdge;
    int nBoundary;
} femEdges;

typedef struct {
    double f0;
    double gravity;
    double beta;
    double height;
    double rho;
    double gamma;
    double L;
    double tau0;
    double residual;
    femMesh *mesh;
    femEdges *edges;
    femMatrix *invA;
    femVector *field;
    femVector **dataFields;
} femProblem;

int         femGaussTriangleNumber(void);
double      femGaussTriangleXsi(int i);
double      femGaussTriangleEta(int i);
double      femGaussTriangleWeight(int i);

int         femGaussEdgeNumber(void);
double      femGaussEdgeXsi(int i);
double      femGaussEdgeWeight(int i);

femMesh*    femMeshNew(void);
void        femMeshFree(femMesh *theMesh);
void        femMeshGmshRead(femMesh* myMesh, const char *filename);
void        femMeshRead(femMesh* myMesh, const char *filename);
void        femMeshWrite(const femMesh* myMesh, const char *filename);
int*        femMeshElem(const femMesh *theMesh, const int i);
void        femMeshEdges(femMesh *theMesh, femEdges *theEdges);

femEdges*   femEdgesNew(femMesh *theMesh);
void        femEdgesFree(femEdges *myEdges);
int         femEdgeCompare(const void* e0, const void *e1);
void        femEdgePrint(femEdges *theEdges);
void        femEdgeMap(femProblem *myProblem, int i, int map[2][2]);
void        femEdgeCoordinates(femProblem *myProblem, int index, double x[2], double y[2]);

void        femTriangleMap(femProblem *myProblem, int i, int map[3]);
void        femTriangleMapCoordinates(femProblem *myProblem, int index, int map[3]);
void        femTriangleCoordinates(femProblem *myProblem, int index, double x[3], double y[3]);
    
void        femGridRead(femGrid* myGrid, const char *filename);
double      femGridInterpolate(femGrid *theGrid, double x, double y);

void        femFullSystemAlloc(femFullSystem* mySystem,int size);
void        femFullSystemPrint(femFullSystem* mySystem);
void        femFullSystemInit(femFullSystem* mySystem);
double*     femFullSystemEliminate(femFullSystem* mySystem);
void        femFullSystemConstrain(femFullSystem* mySystem, int myNode, double value);

femMatrix*  femMatrixNew(int size);
void        femMatrixPrint(femMatrix* myMatrix);
void        femMatrixFree(femMatrix* myMatrix);
void        femInverseDefPosMatrix(femMatrix *A,femMatrix *invA);

femVector*  femVectorNew(int size);
void        femVectorPrint(femVector* myVector);
void        femVectorInit(femVector* myVector);
void        femVectorFree(femVector* myVector);
void        femVectorLinspace(femVector* myVector, double start, double endop);
void        femVectorWrite(const femVector* myVector, const char *filename);
void        femVectorRead(femVector* myVector, const char *filename);
double		femVectorNorm(const femVector* myVector);

void        femUpdateEuler(femProblem *myProblem, double dt);
void        femUpdateRungeKutta(femProblem *myProblem, double dt);
void        femComputeRightHandSide(femProblem *myProblem, femVector *F);

double      femInterpolate(double *phi, double *U, int *map, int n);

femProblem *femStommelNew(const char *meshFileName);
void        femStommelFree(femProblem *myProblem);
void        femStommelAddIntegralsTriangles(femProblem *myProblem, femVector *vectorB);
void        femStommelAddIntegralsEdges(femProblem *myProblem, femVector *vectorB);
void        femStommelMultiplyInverseMatrix(femProblem *myProblem, femVector *vectorB, femVector *f);
void        femStommel(double x, double y, femProblem *myProblem, double *u, double *v, double *eta);

int         femShapesTriangleNumber(void) ;
void        femShapesTriangleNodes(double *xsi, double *eta);
void        femShapesTrianglePhi(double *phi, double xsi, double eta);
void        femShapesTriangleDphi(double *dphdxsi, double *dphideta, double xsi, double eta);
void        femShapesTrianglePrint(void);
void        femMassMatrixTriangle(femMatrix *A);

int         femShapesEdgeNumber(void) ;
void        femShapesEdgeNodes(double *xsi);
void        femShapesEdgePhi(double *phi, double xsi);
void        femShapesEdgeDphi(double *dphdxsi, double xsi);
void        femShapesEdgePrint(void);

double      femMin(double *x, int n);
double      femMax(double *x, int n);
void        femError(char *text, int line, char *file);
void        femWarning(char *text, int line, char *file);

# endif