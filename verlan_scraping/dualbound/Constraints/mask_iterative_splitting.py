import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from .arb_domains import get_arb_proj_in_rect_region

def split_sparseP_by_violation(Lags, spPlist, violation):
    """
    splits local constraints into finer local constraints according
    to the spatial distribution of the constraint violation
    Re{<S|T>-<T|U|T>} as given by the 1D array violation
    """

    pos_violation = violation>0.0
    neg_violation = violation<=0.0
    new_spPlist = []
    new_Lags = []
    for i in range(len(spPlist)):
        spP = spPlist[i]
        Psupport = spP.diagonal()>0.5
        posP = np.logical_and(pos_violation,Psupport)
        if np.sum(posP)>0:
            posP = sp.diags(posP.astype(np.double), format="csc")
            new_spPlist.append(posP)
            new_Lags.extend([Lags[2*i], Lags[2*i+1]])
        negP = np.logical_and(neg_violation,Psupport)
        if np.sum(negP)>0:
            negP = sp.diags(negP.astype(np.double), format="csc")
            new_spPlist.append(negP)
            new_Lags.extend([Lags[2*i], Lags[2*i+1]])
    if len(new_spPlist)==len(spPlist):
        raise ValueError('no splitting achieved')
    return new_Lags, new_spPlist


def plot_sparseP_distribution(Nx, Ny, design_mask, spPlist, name='test.png'):
    """
    plots the distribution of the local constraints on a rectangular grid
    """
    Pregions = np.zeros((Nx,Ny))
    
    for i in range(len(spPlist)):
        Pregions[design_mask] += spPlist[i].diagonal()*i

    fig = plt.figure()
    plt.imshow(Pregions, cmap='viridis')
    plt.savefig(name)


def get_max_violation_sparseP(violation, align=1.0+0j):
    """
    violation is 1D array over the primal degrees of freedom
    representing spatial distribution of violation of the
    complex power constraint
    get new constraint by setting a new projector P with complex
    entries on the diagonals such that violation_i * P_i all have
    the same phase with align
    for now let every |P_i| = 1 unless violation_i=0 and then P_i = 0
    """
    Pdiag = align / violation
    Pdiag = Pdiag / np.abs(Pdiag) #normalization
    Pdiag[np.isnan(Pdiag)] = 0.0
    return sp.diags(Pdiag, format='csc')

def get_max_violation_sparseP_2(violation, align=1.0+0j):
    """
    violation is 1D array over the primal degrees of freedom
    representing spatial distribution of violation of the
    complex power constraint
    get new constraint by setting a new projector P with complex
    entries on the diagonals such that violation_i * P_i all have
    the same phase with align
    for now let every |P_i| = 1 unless violation_i=0 and then P_i = 0
    """
    Pdiag = align / violation
    Pdiag = Pdiag / np.abs(Pdiag)  #normalization
    violP = violation[np.isnan(Pdiag,where=False)]
    Pdiag = Pdiag * np.abs(violation) / np.sum(np.abs(violP)) * len(violP) 
    Pdiag[np.isnan(Pdiag)] = 0.0
    return sp.diags(Pdiag, format='csc')


def normalize_P(P):
    """
    normalize P such that its Frobenius norm is equal to that of identity
    """
    if sp.issparse(P):
        #normalization = np.sqrt(P.shape[0]) / spla.norm(P, ord='fro')
        normalization = spla.norm(P)
    else:
        #normalization = np.sqrt(P.shape[0]) / la.norm(P, ord='fro')
        normalization = la.norm(P)
    
    P /= normalization
    return P, normalization


def reduce_Plist(Plist,optLags):
    """
    merge the last two projections in Plist together such that the Lagrangian remains the same
    the initial multiplier values for the new P is 0 for the real multiplier and 1 for the old multiplier
    normalize P such that its Frobenius norm is equal to that of Id, i.e., sqrt(n)
    """
    Plist2 = Plist.copy()

    Plist2[-2] = Plist[-2]*(optLags[-4]-(1j)*optLags[-3]) + Plist[-1]*(optLags[-2]-1j*optLags[-1])
    Plist2[-2], normalization = normalize_P(Plist2[-2])

    Plist2 = Plist2[:-1]
    optLags2 = optLags.copy()
    optLags2[-4] = normalization #the real constraint multiplier starts at 1.0 * P normalization
    optLags2[-3] = 0.0 #the imag constraint multiplier starts at 0
    
    optLags2 = optLags2[:-2]
    return Plist2,optLags2


def split_generalP_arb_domains(Pdof, reg_list, n_S=1):
    split_reg_list=[]
    split_Pdof=[]
    flag=False #indicates whether any region has been split or not
    for i in range(len(reg_list)):
        reg = reg_list[i]
        
        projLagnum = 2*n_S**2
        if int(np.sum(reg)) > 1:
            sub_reg_1 = np.zeros(reg.shape,dtype=np.bool)
            sub_reg_2 = sub_reg_1.copy()
            indices = np.arange(len(reg))
            true_indices = indices[reg]
            sub_reg_1[true_indices[len(true_indices)//2:]] = True
            sub_reg_2[true_indices[:len(true_indices)//2]] = True
            
            split_reg_list.append(sub_reg_1)
            split_reg_list.append(sub_reg_2)

            split_Pdof.extend(Pdof[i*projLagnum:(i+1)*projLagnum].tolist() * 2)
            flag=True
        elif int(np.sum(reg)) == 1:
            split_reg_list.append(reg)
            split_Pdof.extend(Pdof[i*projLagnum:(i+1)*projLagnum].tolist() * 1)
        else: #no split was possible
            raise ValueError('nonexistent region')

    if not flag:
        raise ValueError('no more subdivisions possible')

    return np.array(split_Pdof), split_reg_list



def get_arb_Ideslist(design_reg, proj_reg_list, sparse=False, pol='TM'):
    Ideslist = []
    for i in range(len(proj_reg_list)):
        proj_reg = proj_reg_list[i]
        Proj_bool = get_arb_proj_in_rect_region(design_reg, proj_reg)
        Proj_mat = np.zeros_like(Proj_bool, dtype=int)
        Proj_mat[Proj_bool] = 1
        if sparse:
            Proj_mat = sp.diags(Proj_mat, format="csc")
            if pol=='TE':
                Proj_mat = sp.kron(sp.eye(2), Proj_mat, format='csc')
        else:
            Proj_mat = np.diag(Proj_mat)
            if pol=='TE':
                Proj_mat = np.kron(np.eye(2), Proj_mat)

        Ideslist.append(Proj_mat)
        
    return Ideslist


def dualopt_Msparse_iterative_splitting_general(n_S, design_mask, chi, Ginv, dualoptFunc, outputFunc=None, pol='TM'):
    """
    runs dual optimization while iteratively increasing the number of projection constraints, seeding later optimizations with solution of prior optimizations
    dualoptfunc(n_S, initLags, GinvdagPdaglist, UPlist, include) returns dual optimum, optimizing from initLags based on Msparse formulation
    """

    design_reg = design_mask.flatten()[design_mask.flatten()]
    subproj_reg_list = [design_reg.copy()] #start with just the global constraints
    num_iters = 0
    while True:
        num_iters += 1
        proj_reg_list = [design_reg] + subproj_reg_list
        
        Plist = get_arb_Ideslist(design_reg, proj_reg_list, sparse=True, pol=pol)
        num_region = len(Plist)
        print('Total # of subregions', num_region-1, flush=True)
        GinvdagPdaglist = []
        UPlist = []
        include = np.array([True] * (num_region * 2 * n_S**2))
        include[2*n_S**2:4*n_S**2] = False #since we include global constraints, leave out first subregion to prevent linear dependence
        del_items = []
        del_index = -1
        for i in range(num_region):
            if int(np.sum(np.sum(Plist[i]))) == 0:
                del_items += [i*2*n_S**2,i*2*n_S**2+1]
                subproj_reg_list = subproj_reg_list[:del_index]+subproj_reg_list[del_index+1:]
                del_index -= 1
                continue
            del_index += 1
            GinvdagPdag = Ginv.conj().T @ Plist[i].conj().T
            UMP = (Ginv.conj().T @ GinvdagPdag.conj().T)/np.conj(chi) - GinvdagPdag.conj().T
            GinvdagPdaglist.append(GinvdagPdag)
            UPlist.append(UMP)

        #####SET THE INCLUDE LIST#####
        
        if num_iters==1: #start of iteration, initialize Lags randomly
            #Lags = np.zeros(len(include))
            #Lags[include] = np.random.rand(int(np.sum(include)))
            Lags = None

        if del_items:
            del_items = np.array(del_items)
            Lags = np.delete(Lags,del_items)
            include = np.delete(include,del_items)
        print('Begin optimization', flush=True)
        optLags, optgrad, optdual, optobj = dualoptFunc(n_S, Lags, GinvdagPdaglist, UPlist, include)

        if not (outputFunc is None):
            outputFunc(optLags, optgrad, optdual, optobj) #print-out function specific to the problem

        try:
            projLagnum = 2*n_S**2
            subLags = optLags[projLagnum:] #just split the multipliers corresponding to local constraints
            subLags, subproj_reg_list = split_generalP_arb_domains(subLags, subproj_reg_list, n_S=n_S)
            Lags = np.zeros(len(subLags)+projLagnum)
            Lags[:projLagnum] = optLags[:projLagnum]
            Lags[projLagnum:] = subLags[:] #always keep a copy of the global constraints
        except ValueError:
            print('subregions down to pixel level, iterations complete')
            break

    return 0

def dualopt_Msparse_iterative_splitting_general_Lags_input(n_S, Mx, My, design_mask, chi, Ginv, dualoptFunc, outputFunc=None, pol='TM', Lags_input='Lags_var.txt'):
    """
    runs dual optimization while iteratively increasing the number of projection constraints, seeding later optimizations with solution of prior optimizations
    dualoptfunc(n_S, initLags, GinvdagPdaglist, UPlist, include) returns dual optimum, optimizing from initLags based on Msparse formulation
    """

    design_reg = design_mask.flatten()[design_mask.flatten()]
    subproj_reg_list = [design_reg.copy()] #start with just the global constraints
    num_iters = 0
    while True:
        num_iters += 1
        proj_reg_list = [design_reg] + subproj_reg_list
        
        Plist = get_arb_Ideslist(design_reg, proj_reg_list, sparse=True, pol=pol)
        num_region = len(Plist)
        print('Total # of subregions', num_region-1, flush=True)
        GinvdagPdaglist = []
        UPlist = []
        include = np.array([True] * (num_region * 2 * n_S**2))
        include[2*n_S**2:4*n_S**2] = False #since we include global constraints, leave out first subregion to prevent linear dependence
        del_items = []
        del_index = -1
        for i in range(num_region):
            if int(np.sum(np.sum(Plist[i]))) == 0:
                del_items += [i*2*n_S**2,i*2*n_S**2+1]
                subproj_reg_list = subproj_reg_list[:del_index]+subproj_reg_list[del_index+1:]
                del_index -= 1
                continue
            del_index += 1
            GinvdagPdag = Ginv.conj().T @ Plist[i].conj().T
            UMP = (Ginv.conj().T @ GinvdagPdag.conj().T)/np.conj(chi) - GinvdagPdag.conj().T
            GinvdagPdaglist.append(GinvdagPdag)
            UPlist.append(UMP)

        #####SET THE INCLUDE LIST#####
        
        if num_iters==1: #start of iteration, initialize Lags randomly
            Lags = np.zeros(len(include))
            Lags[include] = np.random.rand(int(np.sum(include)))
            try:
                Lags = np.loadtxt(Lags_input)
                Lags = Lags[:len(include)]
            except:
                pass

        if del_items:
            del_items = np.array(del_items)
            Lags = np.delete(Lags,del_items)
            include = np.delete(include,del_items)
        print('Begin optimization', flush=True)
        optLags, optgrad, optdual, optobj = dualoptFunc(n_S, Lags, GinvdagPdaglist, UPlist, include)
        if num_iters==1: #start of iteration, initialize Lags randomly
            np.savetxt(Lags_input,optLags)

        if not (outputFunc is None):
            outputFunc(optLags, optgrad, optdual, optobj) #print-out function specific to the problem

        try:
            projLagnum = 2*n_S**2
            subLags = optLags[projLagnum:] #just split the multipliers corresponding to local constraints
            subLags, subproj_reg_list = split_generalP_arb_domains(subLags, subproj_reg_list, n_S=n_S)
            Lags = np.zeros(len(subLags)+projLagnum)
            Lags[:projLagnum] = optLags[:projLagnum]
            Lags[projLagnum:] = subLags[:] #always keep a copy of the global constraints
        except ValueError:
            print('subregions down to pixel level, iterations complete')
            break

    return 0

