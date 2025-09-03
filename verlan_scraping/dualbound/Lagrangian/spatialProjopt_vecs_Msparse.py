#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 20:49:43 2021

@author: pengning
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import sksparse.cholmod as chol

def get_Tvec(ZTT, ZTS_S, solver=None):
    #the assumption is that ZTT is a sparse matrix
    if solver=='cg':
        Tvec = spla.cg(ZTT, ZTS_S)
    else:
        Tvec = spla.spsolve(ZTT, ZTS_S)
    return Tvec


def get_ZTTcho_Tvec(ZTT, ZTS_S, chofac=None):
    if chofac is None:
        ZTTcho = chol.cholesky(ZTT)
    else:
        ZTTcho = chofac.cholesky(ZTT)
    Tvec = ZTTcho.solve_A(ZTS_S)
    return ZTTcho, Tvec


def get_ZTTcho_Tvec_gradTvec(ZTT, gradZTT, ZTS_S, gradZTS_S, chofac=None):
    if chofac is None:
        ZTTcho = chol.cholesky(ZTT)
    else:
        ZTTcho = chofac.cholesky(ZTT)
    Tvec = ZTTcho.solve_A(ZTS_S)

    gradTvec = []
    for i in range(len(gradZTT)):
        gradTvec.append(ZTTcho.solve_A(-gradZTT[i] @ Tvec + gradZTS_S[i]))

    return ZTTcho, Tvec, gradTvec
