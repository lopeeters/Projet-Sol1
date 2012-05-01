/*
 *  glfem.h
 *  Library for MECA1120 : Finite Elements for dummies
 *
 *  Copyright (C) 2012 UCL-IMMC : Vincent Legat
 *  All rights reserved.
 *
 *  Pour GLFW (version utilisée 2.7.2)
 *  Pour l'installation de la librairie, voir http://www.glfw.org/
 *
 */

#ifndef _GLFEM_H_
#define _GLFEM_H_
#include <GL/glfw.h>
#include "fem.h"

typedef struct {
    femMesh* mesh;
    femEdges* edges;
    double Xmin, Xmax, Ymin, Ymax;
    double *field;
	void (*draw)(void); 	
	void (*animate)(void); 	
	void (*reset)(void); 	
	void (*key)(char); 	
    char *message;
} glfemPlot;

void glfemDrawColorTriangle(float* xy, double *values);
void glfemDrawTriangle(float* xy);
void glfemDrawColorRecursiveTriangle(double *x, double *y, 
									 double *xsi, double *eta, 
									 double (*f)(int,double,double,double,double),
									 double fMin, double fMax,
									 int iElem,
									 int level);
void glfemDrawNodes(double* x, double* y,int n);
void glfemDrawCurve(femVector *X, femVector *Y);
void glfemDrawCurveDiscrete(femVector *X, femVector *Y);

void glfemReshapeWindows(int width, int heigh);
void glfemPlotDiscField(femMesh *theMesh, double (*f)(int,double,double,double,double),double fMin, double fMax,int level);
void glfemPlotField(femMesh *theMesh, double *u);
void glfemPlotMesh(femMesh *theMesh);
void glfemMessage(char *message);

void glfemDrawMessage(int h, int v, char *message);
void glfemSetRasterSize(int width, int height);
void glfemInit(char *windowName, femMesh *theMesh, femEdges *theEdges, double *theField, char *message);
void glfemMainLoop(char *windowName);
void glfemSetDrawFunction(void (*draw)(void));
void glfemSetAnimateFunction(void (*animate)(void));
void glfemSetResetFunction(void (*reset)(void));
void glfemSetKeyFunction(void (*key)(char));
void glfemSetViewPort(double Xmin, double Xmax, double Ymin, double Ymax);
void glfemSetViewPortMesh(femMesh* theMesh);

#endif