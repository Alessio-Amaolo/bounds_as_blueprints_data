import numpy as np 
from skimage.draw import disk

def init_cavity(Nx, Ny, Mx, My, Mi, Npml, Npmlsep, nonpmlNx, nonpmlNy, circle):
    '''
    Tool to draw domains for the optimization problem.
    '''
    design_mask = np.zeros((Nx, Ny), dtype=bool) 

    if circle:
        rr1, cc1 = disk((Nx//2, Ny//2), Mx//2, shape=design_mask.shape)
        design_mask[rr1, cc1] = 1
        if Mi > 0:
            rr2, cc2 = disk((Nx//2, Ny//2), Mi, shape=design_mask.shape) 
            design_mask[rr2, cc2] = 0
    else:
        design_mask[Npml+Npmlsep:Npml+Npmlsep+Mx, Npml+Npmlsep:Npml+Npmlsep+My] = True
        if Mi > 0:
            design_mask[Npml+nonpmlNx//2 - Mi : Npml+nonpmlNx//2 + 1 + Mi,  Npml+nonpmlNy//2 - Mi : Npml+nonpmlNy//2 + 1 + Mi] = False
    ndof = np.sum(design_mask)
    return design_mask, ndof 

def init_box(Nx, Ny, Mx, My, Dx, Npml, Npmlsep):
    '''
    Tool to draw domains for the optimization problem.
    '''
    design_mask = np.zeros((Nx, Ny), dtype=bool) 
    design_mask[Npml+Npmlsep+Dx:Npml+Npmlsep+Dx+Mx, Npml+Npmlsep:Npml+Npmlsep+My] = True
    ndof = np.sum(design_mask)
    return design_mask, ndof

def init_box_strips(Nx, Ny, Mx, My, Dx, Npml, Npmlsep, nstrips):
    strip_width = Mx // nstrips
    design_masks = []
    for i in range(nstrips):
        design_mask = np.zeros((Nx, Ny), dtype=bool) 
        if i == nstrips-1:
            design_mask[Npml+Npmlsep+Dx+i*strip_width:Npml+Npmlsep+Dx+Mx, Npml+Npmlsep:Npml+Npmlsep+My] = True
        else:
            design_mask[Npml+Npmlsep+Dx+i*strip_width:Npml+Npmlsep+Dx+(i+1)*strip_width, Npml+Npmlsep:Npml+Npmlsep+My] = True
        design_masks.append(design_mask)
    
    return design_masks

def getchi(rho, chi, chibkg, Nx, Ny, Dmask):
    rho = np.clip(rho, 0, 1)   
    chigrid = chibkg + (chi-chibkg) * rho
    bigchigrid = np.zeros((Nx, Ny), dtype=complex)
    bigchigrid[Dmask] += chigrid
    return bigchigrid

def getrho(chir, chib, chi):
    # invert getchi
    rho = np.real((chir - chib) / (chi - chib))
    return np.clip(rho, 0, 1)

def vn(value): # vn = value_name
    value_str = f"{value}"
    return value_str.replace('.','p')