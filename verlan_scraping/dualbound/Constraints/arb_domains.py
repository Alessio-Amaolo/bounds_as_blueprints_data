import numpy as np


def get_arb_proj_in_rect_region(Dreg, Preg):
    """
    get 1D boolean array mask for arbitrary projection region Preg in a larger rectangular region 
    """
    if np.sum(Dreg*Preg) > np.sum(Preg):
        raise ValueError('projection region outside of domain')

    Proj1D = np.zeros(Dreg.shape,dtype=bool)
    Proj1D[Preg] = True
    return Proj1D


def divide_rect_region(Rulco, Rbrco, Dx, Dy):
    #Dx, Dy are the row and column # of the rectangular subdivisions
    Nx = Rbrco[0] - Rulco[0]
    Ny = Rbrco[1] - Rulco[1]
    lxcoords = Rulco[0] + int(np.floor(Nx/Dx))*np.arange(Dx)
    rxcoords = np.zeros_like(lxcoords, dtype=int)
    rxcoords[:-1] = lxcoords[1:]
    rxcoords[-1] = Rbrco[0]

    uycoords = Rulco[1] + int(np.floor(Ny/Dy))*np.arange(Dy)
    bycoords = np.zeros_like(uycoords, dtype=int)
    bycoords[:-1] = uycoords[1:]
    bycoords[-1] = Rbrco[1]

    ulco_list = []
    brco_list = []
    for i in range(Dx):
        for j in range(Dy):
            ulco = (lxcoords[i],uycoords[j])
            brco = (rxcoords[i],bycoords[j])
            ulco_list.append(ulco)
            brco_list.append(brco)

    return ulco_list, brco_list
