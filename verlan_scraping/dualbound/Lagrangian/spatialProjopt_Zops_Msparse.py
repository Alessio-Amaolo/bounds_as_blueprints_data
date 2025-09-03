#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 20:14:42 2021

This is part of the grad/Hess engine for spatial projection versions of the 
original global constraint <S|T>-<T|U|T>, formulated with sparse matrices based on the 
Maxwell operator. 

@author: pengning
"""

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import sksparse.cholmod as chol


def get_Msparse_gradZTT(UPlist):
    gradZ = []
    for i in range(len(UPlist)):
        SymUP = (UPlist[i]+UPlist[i].conj().T)/2
        AsymUP = (UPlist[i]-UPlist[i].conj().T)/(2j)
        gradZ.append(SymUP)
        gradZ.append(AsymUP)
    return gradZ


def get_multiSource_Msparse_gradZTT(n_S, UPlist):
    n_basis = UPlist[0].shape[0]
    n_pdof = n_S * n_basis #number of primal dofs, corresponding to size of ZTT / gradZTT matrices
    n_cplx_projLags = n_S**2

    Lagnum = len(UPlist) * 2 * n_cplx_projLags
    gradZ = [None] * Lagnum

    for l in range(len(UPlist)):
        UMP = UPlist[l]
        UMPH = UPlist[l].T.conj()
        
        SymUMP = (UMP+UMPH)/2
        AsymUMP = (UMP-UMPH)/2j
        ind_offset_Lags = l * 2 * n_cplx_projLags

        for i in range(n_S):
            for j in range(n_S): #go through all the cross-source constraints
                gradZ_re_blocks = [[sp.csc_matrix((n_basis,n_basis), dtype=complex) for p in range(n_S)] for q in range(n_S)] #black matrix representation of gradZ, to be used later with scipy.sparse.bmat
                gradZ_im_blocks = [[sp.csc_matrix((n_basis,n_basis), dtype=complex) for p in range(n_S)] for q in range(n_S)]
                if i==j:
                    gradZ_re_blocks[i][i] = SymUMP
                    gradZ_im_blocks[i][i] = AsymUMP
                else:
                    gradZ_re_blocks[i][j] = UMP/2
                    gradZ_re_blocks[j][i] = UMPH/2
                    gradZ_im_blocks[i][j] = -1j*UMP/2
                    gradZ_im_blocks[j][i] = 1j*UMPH/2

                #add to gradZ list
                ind_re = ind_offset_Lags + i*n_S + j
                ind_im = ind_offset_Lags + n_cplx_projLags + i*n_S + j
                gradZ[ind_re] = sp.bmat(gradZ_re_blocks, format='csc')
                gradZ[ind_im] = sp.bmat(gradZ_im_blocks, format='csc')

    return gradZ


def get_mSmF_Msparse_gradZTT(n_S, chilist, Ginvlist, Plist):
    """
    allows for different frequency cross source constraints
    chilist, Ginvlist are over the n_S different freqs
    Plist is over the projection constraints imposed
    """
    n_basis = Plist[0].shape[0]
    n_pdof = n_S * n_basis #number of primal dofs, corresponding to size of ZTT / gradZTT matrices
    n_proj = len(Plist) #number of projection operators
    n_cplx_projLags = n_S**2

    Lagnum = n_proj * 2 * n_cplx_projLags
    gradZ = [None] * Lagnum

    for l in range(n_proj):
        P_l = Plist[l]
        ind_offset_Lags = l * 2 * n_cplx_projLags

        for i in range(n_S):
            for j in range(n_S): #go through all the cross-source constraints
                
                #generate UMP
                Ginv_i = Ginvlist[i]; Ginv_j = Ginvlist[j]
                chi_i = chilist[i]
                UMP =  (Ginv_i.conj().T @ P_l @ Ginv_j)/np.conj(chi_i) - P_l @ Ginv_j
                #create the gradient matrices for these particular indices
                gradZ_re_blocks = [[sp.csc_matrix((n_basis,n_basis), dtype=complex) for p in range(n_S)] for q in range(n_S)] #black matrix representation of gradZ, to be used later with scipy.sparse.bmat
                gradZ_im_blocks = [[sp.csc_matrix((n_basis,n_basis), dtype=complex) for p in range(n_S)] for q in range(n_S)]
                
                if i==j:
                    SymUMP = (UMP + UMP.conj().T) / 2
                    AsymUMP = (UMP - UMP.conj().T) / 2j
                    gradZ_re_blocks[i][i] = SymUMP
                    gradZ_im_blocks[i][i] = AsymUMP
                else:
                    UMPH = UMP.conj().T
                    gradZ_re_blocks[i][j] = UMP/2
                    gradZ_re_blocks[j][i] = UMPH/2
                    gradZ_im_blocks[i][j] = -1j*UMP/2
                    gradZ_im_blocks[j][i] = 1j*UMPH/2

                ind_re = ind_offset_Lags + i*n_S + j
                ind_im = ind_offset_Lags + n_cplx_projLags + i*n_S + j

                gradZ[ind_re] = sp.bmat(gradZ_re_blocks, format='csc')
                gradZ[ind_im] = sp.bmat(gradZ_im_blocks, format='csc')

    return gradZ


def get_ZTT(Lags, O, gradZTT):
    ZTT = O.copy()
    for i in range(len(Lags)):
        ZTT += Lags[i] * gradZTT[i]
    return ZTT

def Cholesky_analyze_ZTT(O, gradZTT, verbose=True):
    Lags = np.random.rand(len(gradZTT))
    ZTT = get_ZTT(Lags, O, gradZTT)
    if verbose: print('analyzing ZTT of format and shape', ZTT.format, ZTT.shape, 'and # of nonzero elements', ZTT.count_nonzero())
    return chol.analyze(ZTT)


def get_Msparse_gradZTS_S(Si, GinvdagPdaglist):
    gradZTS_S = []

    for l in range(len(GinvdagPdaglist)):
        GinvdagPdag_S = GinvdagPdaglist[l] @ Si
        gradZTS_S.append(GinvdagPdag_S/2.0)
        gradZTS_S.append(1j*GinvdagPdag_S/2.0)
    return gradZTS_S


def get_multiSource_Msparse_gradZTS_S(n_S, Si_st, GinvdagPdaglist):
    n_basis = GinvdagPdaglist[0].shape[0]
    n_cplx_projLags = n_S**2
    gradZTS_S = [None] * (2*n_cplx_projLags*len(GinvdagPdaglist))

    for l in range(len(GinvdagPdaglist)):
        GinvdagPdag = GinvdagPdaglist[l]
        ind_offset_Lags = l * 2 * n_cplx_projLags
        for i in range(n_S):
            Si_i = Si_st[i*n_basis:(i+1)*n_basis]
            for j in range(n_S):
                ind_re = ind_offset_Lags + i*n_S + j
                ind_im = ind_offset_Lags + n_cplx_projLags + i*n_S + j
                gradZTS_S_re = np.zeros(len(Si_st), dtype=complex)
                gradZTS_S_re[j*n_basis:(j+1)*n_basis] = GinvdagPdag @ Si_i / 2
                gradZTS_S_im = 1j * gradZTS_S_re

                gradZTS_S[ind_re] = gradZTS_S_re
                gradZTS_S[ind_im] = gradZTS_S_im

    return gradZTS_S


def get_mSmF_Msparse_gradZTS_S(n_S, Si_st, Ginvlist, Plist):
    n_basis = Plist[0].shape[0]
    n_proj = len(Plist)
    n_cplx_projLags = n_S**2
    gradZTS_S = [None] * (2*n_cplx_projLags*n_proj)

    for l in range(n_proj):
        P_l = Plist[l]
        ind_offset_Lags = l * 2 * n_cplx_projLags
        for i in range(n_S):
            Si_i = Si_st[i*n_basis:(i+1)*n_basis]
            for j in range(n_S):
                ind_re = ind_offset_Lags + i*n_S + j
                ind_im = ind_offset_Lags + n_cplx_projLags + i*n_S + j
                gradZTS_S_re = np.zeros(len(Si_st), dtype=complex)
                gradZTS_S_re[j*n_basis:(j+1)*n_basis] = Ginvlist[j].conj().T @ (P_l.conj().T @ Si_i) / 2
                gradZTS_S_im = 1j * gradZTS_S_re

                gradZTS_S[ind_re] = gradZTS_S_re
                gradZTS_S[ind_im] = gradZTS_S_im

    return gradZTS_S


def check_Msparse_spatialProj_Lags_validity(Lags, O, gradZTT, chofac=None, mineigtol=None):
    ZTT = get_ZTT(Lags, O, gradZTT)
    if not (mineigtol is None):
        ZTT -= mineigtol * sp.eye(ZTT.shape[0], format='csc')
    try:
        if chofac is None:
            ZTTcho = chol.cholesky(ZTT)
            tmp = ZTTcho.L() # necessary to attempt to access raw factor for checking matrix definiteness
        else:
            ZTTcho = chofac.cholesky(ZTT)
            tmp = ZTTcho.L() # see above
    except chol.CholmodNotPositiveDefiniteError:
        return False
    return True

def check_Msparse_spatialProj_incLags_validity(incLags, include, O, gradZTT, chofac=None, mineigtol=None):
    Lags = np.zeros(len(include))
    Lags[include] = incLags[:]
    return check_Msparse_spatialProj_Lags_validity(Lags, O, gradZTT, chofac=chofac, mineigtol=mineigtol)


def get_Msparse_PD_ZTT_mineig(Lags, O, gradZTT, eigvals_only=False):
    """
    assuming ZTT is PD, find its minimum eigenvalue/vector using shift invert mode of spla.eigsh
    """
    ZTT = get_ZTT(Lags, O, gradZTT)
    if eigvals_only:
        try:
            eigw = spla.eigsh(ZTT, k=1, sigma=0.0, which='LM', return_eigenvectors=False)
        except BaseException as err:
            print('encountered error in sparse eigenvalue evaluation', err)
            eigw = la.eigvalsh(ZTT.todense())
        return eigw[0]
    else:
        try:
            eigw, eigv = spla.eigsh(ZTT, k=1, sigma=0.0, which='LM', return_eigenvectors=True)
        except BaseException as err:
            print('encountered error in sparse eigenvalue evaluation', err)
            eigw, eigv = la.eigh(ZTT.todense())
        return eigw[0], eigv[:,0]
    
    
def get_Msparse_ZTT_mineig(Lags, O, gradZTT, eigvals_only=False):
    ZTT = get_ZTT(Lags, O, gradZTT)
    if eigvals_only:
        eigw = spla.eigsh(ZTT, k=1, which='SA', return_eigenvectors=False)
        return eigw[0]
    else:
        eigw, eigv = spla.eigsh(ZTT, k=1, which='SA', return_eigenvectors=True)
        return eigw[0], eigv[:,0]


def get_Msparse_inc_ZTT_mineig(incLags, include, O, gradZTT, eigvals_only=False):
    Lags = np.zeros(len(include))
    Lags[include] = incLags[:]
    return get_Msparse_ZTT_mineig(Lags, O, gradZTT, eigvals_only=eigvals_only)


def get_Msparse_inc_PD_ZTT_mineig(incLags, include, O, gradZTT, eigvals_only=False):
    Lags = np.zeros(len(include))
    Lags[include] = incLags[:]
    return get_Msparse_PD_ZTT_mineig(Lags, O, gradZTT, eigvals_only=eigvals_only)
    

###method for finding derivatives of mineig of ZTT, to use for phase I (entering domain of duality) of optimization

def get_Msparse_ZTT_mineig_grad(ZTT, gradZTT):
    eigw, eigv = spla.eigsh(ZTT, k=1, which='SA', return_eigenvectors=True)
    eiggrad = np.zeros(len(gradZTT))
    
    for i in range(len(eiggrad)):
        eiggrad[i] = np.real(np.vdot(eigv[:,0], gradZTT[i].dot(eigv[:,0])))
    return eigw[0], eiggrad
