import numpy as np 
import scipy.sparse as sp
import sys 

import dualbound.Maxwell.TM_FDFD as TM 
import design_tools as dt
import json, os
import copy 
import matplotlib.pyplot as plt 
from dualbound.Constraints.off_diagonal_P_save_struct import dual_space_reduction_iteration_Msparse_align_mineig_maxviol as gcd

# Implements scraping for naive_extraction_gcd
# Basically just take yn -> yn + sigma * (previous optimal T) until you get strong duality 
# Strong duality will be checked by the condition that the compact constraint holds

normalize = True
sigma = float(sys.argv[1])
Lsize = float(sys.argv[2])
gpr = int(sys.argv[3])
gcd_max_iter = eval(str(sys.argv[4]))
start_viter = int(sys.argv[5])
problem_type = sys.argv[6]
dist_x = float(sys.argv[7])
assert problem_type in ['oneside', 'center_square', 'center_circle']

# from naive_extraction_gcd import get_TM_LDOS_oneside_gcd_extract_struct as bound
chi = 5+1e-6j
chilabel = '5+1e-6j'
wvlgth = 1.0
dl = 1/gpr 
des_x = Lsize
des_y = Lsize
#sigma = 0.05 # scraping size
delta = 1e-4 # strong duality tolerance
#gcd_max_iter = np.inf
gcd_iter_period = 5
pnum = 5
viol_switch = True

def vn(value): # vn = value_name
    value_str = f"{value}"
    return value_str.replace('.','p')

pml_sep = 0.5
pml_thick=0.5

print("Paramterers")
print(f"design_x: {des_x}")
print(f"design_y: {des_y}")
print(f"dist_x: {dist_x}")
print(f"gpr: {gpr}")
print(f"chi: {chi}")
print(f"sigma: {sigma}")
print(f"delta: {delta}")
print(f"vswitch: {viol_switch}")
print(f"normalize: {normalize}")
print(f'pnum: {pnum}')
print(f'gcd_max_iter: {gcd_max_iter}')
print(f'gcd_iter_period: {gcd_iter_period}', flush=True)

name = f'TM_LDOS_{problem_type}_chi{chilabel}_des{vn(des_x)}x{vn(des_y)}_dist{vn(dist_x)}_gpr_{gpr}_gcd_iters_{gcd_max_iter}_pnum_{pnum}_sigma_{vn(sigma)}_delta_{vn(delta)}_vswitch_{viol_switch}'
folder = f'scraping_results/{name}'

verlan_params = {}

zinv = np.imag(chi) / np.real(chi*np.conj(chi))
dl = 1.0 / gpr
omega = 2*np.pi / wvlgth

Mx = int(np.round(des_x / dl))
My = int(np.round(des_y / dl))
Npml = int(np.round(pml_thick / dl))
Npml_sep = int(np.round(pml_sep / dl))

if problem_type == 'oneside':
    Dx = int(np.round(dist_x / dl))
    nonpmlNx = Mx + Dx + 2*Npml_sep
    nonpmlNy = My + 2*Npml_sep
    Nx = nonpmlNx + 2*Npml
    Ny = nonpmlNy + 2*Npml

    des_ulx = Npml+Npml_sep+Dx
    des_uly = (Ny-My)//2

    des_mask = np.zeros((Nx,Ny), dtype=bool)
    des_mask[des_ulx:des_ulx+Mx , des_uly:des_uly+My] = True

    #get Green's  function
    Ginv, _ = TM.get_Gddinv(omega, dl, Nx, Ny, Npml, des_mask)

    #get dipole source and vacuum ldos
    cx = Npml+Npml_sep
    cy = Ny // 2
else:
    circle = True if problem_type == 'center_circle' else False

    Mi = int(np.round(dist_x / dl)) # dist is the distance to the design, so it is a radius!
    nonpmlNx = Mx + 2*Npml_sep
    nonpmlNy = My + 2*Npml_sep
    Nx = nonpmlNx + 2*Npml
    Ny = nonpmlNy + 2*Npml

    des_mask, ndof = dt.init_cavity(Nx, Ny, Mx, My, Mi, Npml, Npml_sep, nonpmlNx, nonpmlNy, circle)
    #get dipole source and vacuum ldos
    cx = Nx // 2
    cy = Ny // 2
    # design_mask[cx, cy] = True
    #get Green's  function
    Ginv, _ = TM.get_Gddinv(omega, dl, Nx, Ny, Npml, des_mask)


Evac = TM.get_TM_dipole_field(omega, dl, Nx, Ny, cx, cy, Npml)
vacLDOS = -0.5 * np.real(Evac[cx,cy])
print('vacLDOS', vacLDOS, flush=True)
Si_desvec = Evac[des_mask]
O_lin = (-1j*omega/4) * (Ginv.conj().T @ Si_desvec.conj()) * dl**2
O_lin_dense = (-1j*omega/4) * (Si_desvec.conj()) * dl**2
O_lin_dense_norm = np.linalg.norm(O_lin_dense)

O_quad = sp.csc_matrix(Ginv.shape, dtype=complex)
dualconst = vacLDOS

def structFunc(optstruct, iternum, mindual, lags=None, verlan_iter=0, plot_struct=True):
    viter_folder = f"{folder}/viter={verlan_iter}/"
    os.makedirs(viter_folder, exist_ok=True)
    np.save(viter_folder+f'extracted_struct_{iternum}.npy', optstruct)
    if not (lags is None):
        np.save(viter_folder+f'extracted_lags_{iternum}.npy', lags)

    np.save(viter_folder+f'bound_{iternum}.npy', mindual)

    if plot_struct:
        large_struct = np.zeros((Nx, Ny), dtype=complex)
        large_struct[des_mask] = optstruct
        large_struct = large_struct[Npml+Npml_sep:-Npml-Npml_sep, Npml+Npml_sep:-Npml-Npml_sep]

        plt.figure()
        optStruct_truncate_re = np.clip(np.real(large_struct), 0, np.real(chi))
        plt.imshow(optStruct_truncate_re)
        plt.colorbar()
        plt.savefig(viter_folder+f'extracted_struct_{iternum}_trunc.png')
        plt.close()

        plt.figure()
        plt.imshow(np.real(large_struct))
        plt.colorbar()
        plt.savefig(viter_folder+f'extracted_struct_{iternum}.png')
        plt.close()

verlan_result = None 
def dual_func(yn, verlan_iter, plot_struct, only_viol=False):
    viter_folder = f"{folder}/viter={verlan_iter}/"
    prev_viter_folder = f"{folder}/viter={verlan_iter-1}/"

    savefunc = lambda optstruct, iternum, mindual, lags=None: structFunc(optstruct, iternum, mindual, lags, verlan_iter, plot_struct)

    if verlan_iter == 0 and not normalize:
        prev_Pdatalist = None
        prev_optLags = None
    elif verlan_iter == 0 and normalize:
        # Get the data from the un-normalized run (which we already have) viter = 0. This will greatly speed up convergence of viter = 0, which should be identical to unnormalized. 
        prev_Pdatalist = None
        prev_optLags = None
        for sig in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
            print(f'Trying to grab previous viter from sigma={sig}', flush=True)
            
            name_any = f'TM_LDOS_{problem_type}_chi{chilabel}_des{vn(des_x)}x{vn(des_y)}_dist{vn(dist_x)}_gpr_{gpr}_gcd_iters_{gcd_max_iter}_pnum_{pnum}_sigma_{vn(sig)}_delta_{vn(delta)}_vswitch_{viol_switch}'
            if os.path.exists(f'scraping_results/{name_any}/viter=0/final_struct_Pdatalist.npy'):
                print(f'Grabbing previous viter=0 result from name = {name_any}', flush=True)
                prev_Pdatalist = np.load(f'scraping_results/{name_any}/viter=0/final_struct_Pdatalist.npy')
                prev_optLags = np.load(f'scraping_results/{name_any}/viter=0/final_struct_optLags.npy')
                break
        if prev_Pdatalist is None:
            print(f'Could not find any sigma, starting from scratch', flush=True)
    else:
        prev_Pdatalist = np.load(prev_viter_folder+f'final_struct_Pdatalist.npy')
        prev_optLags = np.load(prev_viter_folder+f'final_struct_optLags.npy')

    Pdatalist, optLags, mindual, mingrad, optGT, time_data = gcd(chi, Si_desvec, Ginv, yn, O_quad, dualconst=dualconst, structFunc=savefunc, outputFunc=None, Pnum=pnum, save_period=5, gcd_max_iter=gcd_max_iter, gcd_iter_period=gcd_iter_period, verbose=0, Plist_start=prev_Pdatalist, Lags_start=prev_optLags, only_viol=only_viol)
    Pdatalist = np.array(Pdatalist)

    optT = Ginv @ optGT
    result = {
        'dualval': mindual,
        'compact_viol': mingrad[1],
        'GT': optGT,
        'T': optT
    }

    viter_folder = f"{folder}/viter={verlan_iter}/"
    np.save(viter_folder+f'final_struct_Pdatalist.npy', Pdatalist)
    np.save(viter_folder+f'final_struct_optLags.npy', optLags)

    return result

def _check_strong_duality(grad_compact, delta):
    return np.allclose(grad_compact, 0, atol=delta)
    
def scrape_until_strong_duality(oyn_dense, sigma, delta, start_viter=0):
    if start_viter == 0:
        yn_dense = oyn_dense
    else:
        yn_dense = np.load(f"{folder}/viter={start_viter-1}/yn_dense.npy")
    yn = Ginv.conj().T @ yn_dense
    
    verlan_iter = start_viter
    plot_struct = True #if verlan_iter % 5 == 0 else False
    only_viol = False
    while(True):
        viter_folder = f"{folder}/viter={verlan_iter}/"
        os.makedirs(viter_folder, exist_ok=True)
        if start_viter != verlan_iter or verlan_iter == 0: 
            # if start_viter = 0, then we should save from the beginning _and_ after every step
            # if start_viter != 0, then we shouldn't save the initial step 
            np.save(viter_folder+f'yn.npy', yn)
            np.save(viter_folder+f'yn_dense.npy', yn_dense)
            np.save(viter_folder+f'yn_overlap.npy', np.abs(oyn_dense.conj() @ yn_dense) / np.linalg.norm(oyn_dense) / np.linalg.norm(yn_dense))

        result = dual_func(yn, verlan_iter, plot_struct, only_viol=only_viol)
        if viol_switch: only_viol = True

        print(f"Viter {verlan_iter}. Dual value: {result['dualval']}", flush=True)
        print(f"Overlap is {np.abs(oyn_dense.conj() @ yn_dense) / np.linalg.norm(oyn_dense) / np.linalg.norm(yn_dense)}")
        verlan_iter += 1
        if not _check_strong_duality(result['compact_viol'], delta):
            print(f"Strong duality not found. Scraping with sigma = {sigma}.", flush=True)
            # if not normalize:
            #     yn_dense = yn_dense + sigma * result['T'] * (dl * dl)
            if normalize:
                result['T'] *= O_lin_dense_norm / np.linalg.norm(result['T'])
                yn_dense = yn_dense + sigma * result['T']
                
            yn_dense *= O_lin_dense_norm / np.linalg.norm(yn_dense)
            yn = Ginv.conj().T @ yn_dense
        else:
            print("Strong duality found.", flush=True)
            break

scrape_until_strong_duality(O_lin_dense, sigma, delta, start_viter)
    
