#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 20:55:38 2021

@author: pengning
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import sksparse.cholmod as chol
from .spatialProjopt_vecs_Msparse import get_Tvec, get_ZTTcho_Tvec, get_ZTTcho_Tvec_gradTvec


###METHOD USING SPARSE CHOLESKY DECOMPOSITION FOR LINEAR SOLVE###
def get_spatialProj_dualgrad_fakeS_Msparse(Lags, grad, O_lin, O_quad, gradZTS_S, gradZTT, fSlist, dualconst=0.0, include=None, chofac=None, mineigtol=None):
    if include is None:
        include = [True]*len(Lags)
        
    ZTS_S = O_lin.copy()
    ZTT = O_quad.copy()
    for i in range(len(Lags)):
        ZTS_S += Lags[i] * gradZTS_S[i]
        ZTT += Lags[i] * gradZTT[i]

    # print('ZTS_S,', ZTS_S)
    ZTTcho, T = get_ZTTcho_Tvec(ZTT, ZTS_S, chofac=chofac)
    if mineigtol is None:
        ZTTfScho = ZTTcho
    else:
        if chofac is None:
            print('chofac is None')
            ZTTfScho = chol.cholesky(ZTT - mineigtol*sp.eye(ZTT.shape[0], format='csc'))
        else:
            ZTTfScho = chofac.cholesky(ZTT - mineigtol*sp.eye(ZTT.shape[0], format='csc'))
        
    dualval = dualconst
    dualval += np.real(np.vdot(T, ZTT @ T))
    
    # print(T)
    dualval_penalty = 0 
    dualval_penalty_grad = np.zeros(len(grad))
    if len(grad)>0:
        grad[:] = 0
        for i in range(len(Lags)):
            if include[i]:
                # print('quadratic grad:', -np.real(np.vdot(T, gradZTT[i].dot(T))))
                # print('linear grad:', 2*np.real(np.vdot(T, gradZTS_S[i])))
                grad[i] += -np.real(np.vdot(T, gradZTT[i].dot(T))) + 2*np.real(np.vdot(T, gradZTS_S[i]))

        for _, fS in enumerate(fSlist):
            ZTTfSinv_fS = ZTTfScho.solve_A(fS)
            # print(ZTTfSinv_fS)
            dualval += np.real(np.vdot(fS, ZTTfSinv_fS))
            dualval_penalty += np.real(np.vdot(fS, ZTTfSinv_fS))
            for i in range(len(Lags)):
                if include[i]:
                    grad[i] += -np.real(np.vdot(ZTTfSinv_fS, gradZTT[i].dot(ZTTfSinv_fS)))
                    dualval_penalty_grad[i] += -np.real(np.vdot(ZTTfSinv_fS, gradZTT[i].dot(ZTTfSinv_fS)))

    else:
        for _, fS in enumerate(fSlist):
            ZTTfSinv_fS = ZTTfScho.solve_A(fS)
            dualval += np.real(np.vdot(fS, ZTTfSinv_fS))
            dualval_penalty += np.real(np.vdot(fS, ZTTfSinv_fS))

    np.save('ldos_dualval_penalty.npy', dualval_penalty)
    np.save('ldos_grad_penalty.npy', dualval_penalty_grad)
    return dualval


def get_spatialProj_dualgradHess_fakeS_Msparse(Lags, grad, Hess, O_lin, O_quad, gradZTS_S, gradZTT, fSlist, dualconst=0.0, include=None, chofac=None, mineigtol=None):
    if include is None:
        include = [True]*len(Lags)
        
    ZTS_S = O_lin.copy()
    ZTT = O_quad.copy()
    for i in range(len(Lags)):
        ZTS_S += Lags[i] * gradZTS_S[i]
        ZTT += Lags[i] * gradZTT[i]

    ZTTcho, T, gradT = get_ZTTcho_Tvec_gradTvec(ZTT, gradZTT, ZTS_S, gradZTS_S, chofac=chofac)
    if mineigtol is None:
        ZTTfScho = ZTTcho
    else:
        if chofac is None:
            print('chofac is None')
            ZTTfScho = chol.cholesky(ZTT - mineigtol*sp.eye(ZTT.shape[0], format='csc'))
        else:
            ZTTfScho = chofac.cholesky(ZTT - mineigtol*sp.eye(ZTT.shape[0], format='csc'))

    dualval = dualconst + np.real(np.vdot(T, ZTT @ T))

    grad[:] = 0
    Hess[:,:] = 0
    
    for i in range(len(Lags)):
        if include[i]:
            grad[i] += -np.real(np.vdot(T, gradZTT[i] @ T)) + 2*np.real(np.vdot(T, gradZTS_S[i]))
            
    for i in range(len(Lags)):
        if not include[i]:
            continue
        for j in range(i,len(Lags)):
            if not include[j]:
                continue
            Hess[i,j] += 2*np.real(np.vdot(gradT[i],-gradZTT[j] @ T + gradZTS_S[j]))
            if i!=j:
                Hess[j,i] = Hess[i,j]

    for _, fS in enumerate(fSlist):
        ZTTfSinv_fS = ZTTfScho.solve_A(fS)
        dualval += np.real(np.vdot(fS, ZTTfSinv_fS))
        ZTTfSinv_gradZTT_ZTTfSinv_fS = []
        for i in range(len(Lags)):
            if include[i]:
                gradZTT_ZTTfSinv_fS = gradZTT[i] @ ZTTfSinv_fS
                grad[i] += -np.real(np.vdot(ZTTfSinv_fS, gradZTT_ZTTfSinv_fS))
                ZTTfSinv_gradZTT_ZTTfSinv_fS.append(ZTTfScho.solve_A(gradZTT_ZTTfSinv_fS))
            else:
                ZTTfSinv_gradZTT_ZTTfSinv_fS.append(None)

        for i in range(len(Lags)):
            if not include[i]:
                continue
            for j in range(i,len(Lags)):
                if not include[j]:
                    continue
                Hess[i,j] += 2*np.real(np.vdot(ZTTfSinv_fS, gradZTT[i] @ ZTTfSinv_gradZTT_ZTTfSinv_fS[j]))
                if i!=j:
                    Hess[j,i] = Hess[i,j]
                
    return dualval


def get_inc_spatialProj_dualgrad_fakeS_Msparse(incLags, incgrad, include, O_lin, O_quad, gradZTS_S, gradZTT, fSlist, dualconst=0.0, get_grad=True, chofac=None, mineigtol=None):

    Lagnum = len(include)
    Lags = np.zeros(Lagnum)
    Lags[include] = incLags[:]

    if get_grad:
        grad = np.zeros(Lagnum)
        dualval = get_spatialProj_dualgrad_fakeS_Msparse(Lags, grad, O_lin, O_quad, gradZTS_S, gradZTT, fSlist, dualconst=dualconst, include=include, chofac=chofac, mineigtol=mineigtol)
        incgrad[:] = grad[include]
    else:
        dualval = get_spatialProj_dualgrad_fakeS_Msparse(Lags, np.array([]), O_lin, O_quad, gradZTS_S, gradZTT, fSlist, dualconst=dualconst, include=include, chofac=chofac, mineigtol=mineigtol)

    return dualval


def get_inc_spatialProj_dualgradHess_fakeS_Msparse(incLags, incgrad, incHess, include, O_lin, O_quad, gradZTS_S, gradZTT, fSlist, dualconst=0.0, get_grad=True, get_Hess=True, chofac=None, mineigtol=None):
    Lagnum = len(include)
    Lags = np.zeros(Lagnum)
    Lags[include] = incLags[:]
    
    if get_Hess:
        grad = np.zeros(Lagnum)
        Hess = np.zeros((Lagnum,Lagnum))
        dualval = get_spatialProj_dualgradHess_fakeS_Msparse(Lags,grad,Hess, O_lin, O_quad, gradZTS_S, gradZTT, fSlist, dualconst=dualconst, include=include, mineigtol=mineigtol)
        incgrad[:] = grad[include] #[:] since we are modifying in place
        incHess[:,:] = Hess[np.ix_(include,include)]
    elif get_grad:
        grad = np.zeros(Lagnum)
        dualval = get_spatialProj_dualgrad_fakeS_Msparse(Lags,grad, O_lin, O_quad, gradZTS_S, gradZTT, fSlist, dualconst=dualconst, include=include, mineigtol=mineigtol)
        incgrad[:] = grad[include]
    else:
        dualval = get_spatialProj_dualgrad_fakeS_Msparse(Lags,np.array([]), O_lin, O_quad, gradZTS_S, gradZTT, fSlist, dualconst=dualconst, include=include, mineigtol=mineigtol)
        
    return dualval
