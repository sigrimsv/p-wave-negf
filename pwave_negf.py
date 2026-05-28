"""
p-wave magnetic lattice Hamiltonians and lesser Green's functions
within the non-equilibrium Green's function (NEGF) formalism.

Author: -----
"""



## ============================================================
# LIBRARIES
## ============================================================

import numpy as np




## ============================================================
# GLOBAL CONSTANTS
## ============================================================

on_site_energy = 0.0      # Clean-system onsite energy
lattice_const = 1         # Lattice constant a (lattice units)
k_B = 1                   # Boltzmann constant (natural units)

# Pauli spin matrices: sigma_x, sigma_y, sigma_z
pauli_matrices = np.array([
    [[0, 1], [1, 0]],
    [[0, -1j], [1j, 0]],
    [[1, 0], [0, -1]]
], dtype=complex)




## ============================================================
## LATTICE INDEXING FUNCTIONS
## ============================================================

def coord(i, N_x):
    """
    Convert lattice site index to the corresponding spatial coordinate.

    Parameters 
    ----------
    - i (int): Lattice site index.
    - N_x (int): Total number of lattice sites along the x-direction of the system.

    Return 
    ------
    - (x, y) (tuple[int, int]): Spatial lattice coordinate corresponding to site i.

    Notes
    -----
    - Lattice indexing convention: i = 1 <--> (x,y) = (0,0).
    - The indexing increases first along the x-direction, then along the y-direction.
    """

    x = (i - 1) % N_x
    y = (i - 1) // N_x
    return (x, y)


def index(x, y, N_x):
    """
    Convert spatial coordinate to the corresponding lattice site index.

    Parameters
    ----------
    - x (int): Spatial lattice coordinate along the x-direction.
    - y (int): Spatial lattice coordinate along the y-direction.
    - N_x (int): Total number of lattice sites along the x-direction of the system.

    Return 
    ------
    - i (int): Lattice site index corresponding to spatial coordinate (x,y).

    Notes
    -----
    - Lattice indexing convention: i = 1 <--> (x,y) = (0,0)
    - The indexing increases first along the x-direction, then along the y-direction.
    """
    return x + y * N_x + 1

    


## ============================================================
## ITINERANT NON-COLLINEAR P-WAVE MODEL
## ============================================================

def hopping_itinerant(m1, m2, t_s, t_x, t_y, phi):
    """
    Construct the spin-dependent nearest-neighbor hopping matrix 
    for the itinerant non-collinear p-wave model.

    Parameters
    ----------
    - m1 (tuple[int, int]): Spatial lattice coordinate of the initial site.
    - m2 (tuple[int, int]): Spatial lattice coordinate of the neighboring site.
    - t_s (float): Spin-independent hopping amplitude.
    - t_x (float): Spin-dependent hopping amplitude along the x-direction.
    - t_y (float): Spin-dependent hopping amplitude along the y-direction.
    - phi (float): Non-collinear magnetic phase angle.

    Return
    ------
    - (2,2) complex ndarray: Spin-space hopping matrix between lattice sites m1 and m2.
    """

    dx = m2[0] - m1[0]
    dy = m2[1] - m1[1]

    # +x hopping
    if (dx, dy) == (1, 0):
        return -(t_s/2) * np.eye(2) + (t_x/2) * np.exp(-1j*phi) * pauli_matrices[2]
    # -x hopping
    elif (dx, dy) == (-1, 0):
        return -(t_s/2) * np.eye(2) + (t_x/2) * np.exp(1j*phi) * pauli_matrices[2]
    # +y hopping
    elif (dx, dy) == (0, 1):
        return -(t_s/2) * np.eye(2) + (t_y/2) * np.exp(-1j*phi) * pauli_matrices[2]
    # -y hopping
    elif (dx, dy) == (0, -1):
        return -(t_s/2) * np.eye(2) + (t_y/2) * np.exp(1j*phi) * pauli_matrices[2]
    else:
        return np.zeros((2,2), dtype=complex)


def H_itinerant(N, N_x, t_s, t_x, t_y, phi):
    # returns the sample hamiltonian (2Nx2N matrix)
    """
    Construct the lattice Hamiltonian for the itinerant
    non-collinear p-wave model.

    Parameters
    ----------
    - N (int): Total number of lattice sites in the system.
    - N_x (int): Total number of lattice sites along the x-direction of the system.
    - t_s (float): Spin-independent hopping amplitude.
    - t_x (float): Spin-dependent hopping amplitude along the x-direction.
    - t_y (float): Spin-dependent hopping amplitude along the y-direction.
    - phi (float): Non-collinear magnetic phase angle.

    Return
    ------
    - (2N,2N) complex ndarray: Full system Hamiltonian matrix in spin space.
    """

    H_S = np.zeros((N, N, 2, 2), dtype=complex)

    for i in range(N):
        for j in range(N):
            if i == j:
                H_S[i][j] = on_site_energy * np.eye(2)
            else:
                H_S[i][j] = hopping_itinerant( coord(i+1, N_x), coord(j+1, N_x), t_s, t_x, t_y, phi )
    return H_S.transpose(0, 2, 1, 3).reshape(N*2, N*2)




## ============================================================
## LOCALIZED NON-COLLINEAR P-WAVE MODEL
## ============================================================

def hopping_localized(m1, m2, t_s):
    """
    Construct the nearest-neighbor hopping matrix for the
    localized non-collinear p-wave model.

    Parameters
    ----------
    - m1 (tuple[int, int]): Spatial lattice coordinate of the initial site.
    - m2 (tuple[int, int]): Spatial lattice coordinate of the neighboring site.
    - t_s (float): Spin-independent hopping amplitude.

    Return
    ------
    - (2,2) ndarray: Spin-space hopping matrix between lattice sites m1 and m2.
    """

    dx = m2[0] - m1[0]
    dy = m2[1] - m1[1]

    if (dx, dy) in [(1,0), (-1,0), (0,1), (0,-1)]:
        return -t_s * np.eye(2) 
    else:
        return np.zeros((2,2))


def helimagnetic_exchange(x, y, J_sd, phi, eta):
    """
    Construct the local helimagnetic exchange-interaction matrix 
    for the localized non-collinear p-wave model.

    Parameters
    ----------
    - x (int): Spatial lattice coordinate along the x-direction.
    - y (int): Spatial lattice coordinate along the y-direction.
    - J_sd (float): Exchange coupling strength.
    - phi (float): Spin-spiral magnetic phase angle.
    - eta (int): Helimagnetic staggering parameter.

    Return
    -------
    - (2,2) complex ndarray: Local spin-space exchange-interaction matrix 
    at lattice coordinate (x,y).

    Note
    ----
    - eta = 0 (1) yields ferromagnetic (antiferromagnetic) alignment 
    between neighboring chains along the y-direction.
    """

    phase = (-1)**(eta * y)
    S_y = phase * np.cos(phi * x)
    S_z = phase * np.sin(phi * x)
    return J_sd * (S_y * pauli_matrices[1] + S_z * pauli_matrices[2])


def helix_pitch(N_x_uc):
    """
    Construct the helimagnetic pitch angle.

    Parameter
    ---------
    - N_x_uc (int): Number of lattice sites within one magnetic unit cell 
    along the x-direction.

    Return
    ------
    - float: Helimagnetic pitch angle.

    Note
    ----
    - The spin texture performs one full 2π rotation over N_x_uc + 1 lattice sites.
    """

    return 2 * np.pi / (N_x_uc + 1)


def H_localized(N, N_x, t_s, J_sd, phi, eta):
    """
    Construct the lattice Hamiltonian for the localized 
    non-collinear p-wave model.

    Parameters
    ----------
    - N (int): Total number of lattice sites in the system.
    - N_x (int): Total number of lattice sites along the x-direction of the system.
    - t_s (float): Spin-independent hopping amplitude.
    - J_sd (float): Exchange coupling strength.
    - phi (float): Spin-spiral magnetic phase angle.
    - eta (int): Helimagnetic staggering parameter.

    Return
    ------
    - (2N,2N) complex ndarray: Full system Hamiltonian matrix in spin space.

    Note
    ----
    - eta = 0 (1) yields ferromagnetic (antiferromagnetic) alignment
    between neighboring chains along the y-direction.
    """

    H_S = np.zeros((N, N, 2, 2), dtype=complex)
    for i in range(N):
        xi, yi = coord(i+1, N_x)
        for j in range(N):
            if i == j:
                H_S[i][j] = (on_site_energy * np.eye(2) + helimagnetic_exchange(xi, yi, J_sd, phi, eta))
            else:
                H_S[i][j] = hopping_localized( coord(i+1, N_x), coord(j+1, N_x), t_s )
    return H_S.transpose(0, 2, 1, 3).reshape(N*2, N*2)




## ============================================================
## COLLINEAR P-WAVE MODEL
## ============================================================

def hopping_collinear(m1, m2, t_s, phi):
    """
    Construct the hopping matrix for the collinear p-wave model.

    Parameters
    ----------
    - m1 (tuple[int, int]): Spatial lattice coordinate of the initial site.
    - m2 (tuple[int, int]): Spatial lattice coordinate of the neighboring site.
    - t_s (float): Spin-independent hopping amplitude.
    - phi (float): Collinear magnetic phase angle.

    Return
    ------
    - (2,2) complex ndarray: Spin-space hopping matrix 
    between lattice sites m1 and m2.

    Note
    ----
    - The lattice consists of two sublattices:
        A : spin-up sublattice consisting of all sites with odd x + y
        B : spin-down sublattice consisting of all sites with even x + y
    """

    x1, y1 = m1
    x2, y2 = m2

    dx = x2 - x1
    dy = y2 - y1

    sub1 = (x1 + y1) % 2
    sub2 = (x2 + y2) % 2

    if (dx, dy) in [(1,0), (-1,0), (0,1), (0,-1)] or (dx, dy) in [(1,-1), (-1,1)]:
        return -t_s * np.eye(2)
    
    elif sub1 == sub2 and (dx, dy) == (1,1):
        if sub1 == 0:   # B sublattice
            return -t_s * np.exp(-1j * phi/2) * np.eye(2)
        else:           # A sublattice
            return -t_s * np.exp(1j * phi/2) * np.eye(2) 

    elif sub1 == sub2 and (dx, dy) == (-1,-1):
        if sub1 == 0:   # B sublattice
            return -t_s * np.exp(1j * phi/2) * np.eye(2)
        else:           # A sublattice
            return -t_s * np.exp(-1j * phi/2) * np.eye(2) 

    else:
        return np.zeros((2,2))


def H_collinear(N, N_x, m_AFM, t_s, phi):
    """
    Construct the lattice Hamiltonian for the collinear p-wave model.

    Parameters
    ----------
    - N (int): Total number of lattice sites in the system.
    - N_x (int): Total number of lattice sites along the x-direction of the system.
    - m_AFM (float): Antiferromagnetic exchange strength.
    - t_s (float): Spin-independent hopping amplitude.
    - phi (float): Collinear magnetic phase angle.

    Return
    ------
    - (2N,2N) complex ndarray: Full system Hamiltonian matrix in spin space.

    Note
    ----
    - The lattice consists of two sublattices:
        A : spin-up sublattice consisting of all sites with odd x + y
        B : spin-down sublattice consisting of all sites with even x + y
    """

    H_S = np.zeros((N, N, 2, 2), dtype=complex)

    for i in range(N):
        for j in range(N):

            if i == j:
                x, y = coord(i+1, N_x)
                AFM_staggering = (-1)**(x + y)
                H_S[i][i] = on_site_energy*np.eye(2) - m_AFM * AFM_staggering * pauli_matrices[2]
            else:
                H_S[i][j] = hopping_collinear( coord(i+1, N_x), coord(j+1, N_x), t_s, phi )

    return H_S.transpose(0, 2, 1, 3).reshape(N*2, N*2)




## ============================================================
## NEGF LEAD SELF-ENERGIES AND LESSER GREEN'S FUNCTIONS
## ============================================================

def k(j, N_i):
    # j range from 1 to N_y (in case of leads situated rigth/left)
    """
    Construct the transverse lead wavevector.

    Parameters
    ----------
    - j (int): Transverse lead mode index.
    - N_i (int): Number of lattice sites along the i-direction of the lead.

    Return
    ------
    - float: Quantized transverse lead wavevector.

    Notes
    -----
    - N_i = N_y for left and right leads.
    - N_i = N_x for upper and lower leads.
    """

    return (np.pi * j) / (lattice_const * (N_i + 1))


def energy_disp(k_j, t_L):
    """
    Construct the lead energy dispersion relation.

    Parameters
    ----------
    - k_i (float): Transverse lead wavevector.
    - t_L (float): Lead hopping amplitude.

    Return
    ------
    - float: Lead energy dispersion corresponding to k_i.
    """
    return 2 * t_L * np.cos(k_j * lattice_const)


def self_energy_leads(E, l, N_x, N_y, t_L, t_C):
    """
    Construct the retarded self-energy matrix for a lead.

    Parameters
    ----------
    - E (float): Energy at which the self-energy is evaluated.
    - l (str): Lead label.
        'L' : left lead
        'R' : right lead
        'U' : upper lead
        'D' : lower lead
    - N_x (int): Total number of lattice sites along the x-direction of the system.
    - N_y (int): Total number of lattice sites along the y-direction of the system.
    - t_l (float): Lead hopping amplitude.
    - t_c (float): Coupling strength between lead and central-sample region

    Return
    ------
    - (2N,2N) complex ndarray: Retarded lead self-energy matrix in spin space.
    """

    N = N_x * N_y
    self_E_L = np.zeros((N, N, 2, 2), dtype=complex)

    if l == 'L':
        sites = [(0, my) for my in range(0, N_y)]
        N_i = N_y
        coord_idx = 1
    elif l == 'R':
        sites = [(N_x-1, my) for my in range(0, N_y)]
        N_i = N_y
        coord_idx = 1
    elif l == 'D':
        sites = [(mx, 0) for mx in range(0, N_x)]
        N_i = N_x
        coord_idx = 0
    elif l == 'U':
        sites = [(mx, N_y-1) for mx in range(0, N_x)]
        N_i = N_x
        coord_idx = 0
    else:
        raise ValueError("Lead must be one of 'L','R','U','D'")
        
    for m1 in sites:
        for m2 in sites:
            
            ssum = 0
            for j in range(1, N_i+1):
                k_i = k(j, N_i)
                E_J = E - energy_disp(k_i, t_L)

                if abs(E_J) <= 2.0 * t_L: 
                    ssum += np.sin(k_i * (m1[coord_idx]+1) * lattice_const) * np.sin(k_i * (m2[coord_idx]+1) * lattice_const) \
                        * (t_C**2 / (2*(t_L**2))) * (E_J - 1j * np.sqrt(4*(t_L**2) - E_J**2))
                else:
                    ssum += np.sin(k_i * (m1[coord_idx]+1) * lattice_const) * np.sin(k_i * (m2[coord_idx]+1) * lattice_const) \
                        * (t_C**2 / (2*(t_L**2))) * (E_J - np.sign(E_J) * np.sqrt(E_J**2 - 4*(t_L**2)))
            
            i1 = index(m1[0], m1[1], N_x)
            i2 = index(m2[0], m2[1], N_x)
            self_E_L[i1-1][i2-1] = (2 / (N_i + 1)) * ssum * np.eye(2, dtype=complex)
    
    return self_E_L.transpose(0, 2, 1, 3).reshape(N*2, N*2)


def f(E, mu_l, T_l):
    """
    Construct the Fermi-Dirac distribution function.

    Parameters
    ----------
    - E (float): Energy.
    - mu_l (float): Chemical potential of the lead.
    - T_l (float): Temperature of the lead.

    Return
    ------
    - float: Fermi-Dirac occupation probability.

    Note
    ----
    - In the zero-temperature limit, the Fermi-Dirac
    distribution reduces to a step function.
    """

    if T_l <= 1e-10:
        return 1.0 if E < mu_l else 0.0
    else:
        return 1 / (1 + np.exp((E - mu_l) / (T_l * k_B)))


def G_lesser(E, E_F, H_S, mu_L, mu_R, mu_U, mu_D, N_x, N_y, t_L, t_C, T_dict):
    """
    Construct the lesser Green's function of the system.

    Parameters
    ----------
    - E (float): Energy at which the lesser Green's function is evaluated.
    - E_F (float): Equilibrium Fermi energy.
    - H_S ((2N,2N) complex ndarray): Full system Hamiltonian matrix in spin space.
    - mu_L (float): Chemical potential of the left lead.
    - mu_R (float): Chemical potential of the right lead.
    - mu_U (float): Chemical potential of the upper lead.
    - mu_D (float): Chemical potential of the lower lead.
    - N_x (int): Total number of lattice sites along the x-direction of the system.
    - N_y (int): Total number of lattice sites along the y-direction of the system.
    - t_L (float): Lead hopping amplitude.
    - t_C (float): Coupling strength between lead and central region.
    - T_dict (dict): Dictionary containing the temperatures of each lead.

    Return
    -------
    - (2N,2N) complex ndarray: Full lesser Green's function matrix in spin space.

    Note
    ----
    - The lead temperatures are specified through
    T_dict = {'L': ..., 'R': ..., 'U': ..., 'D': ...}
    """

    N = N_x * N_y
    mu_dict = {'L': mu_L, 'R': mu_R, 'U': mu_U, 'D': mu_D}
    eV_dict = {'L': mu_L-E_F, 'R': mu_R-E_F, 'U': mu_U-E_F, 'D': mu_D-E_F}
    Sigma_l = np.zeros((2*N, 2*N), dtype=complex)
    tot_self_E_leads = np.zeros((2*N, 2*N), dtype=complex)

    for l in ['L', 'R', 'U', 'D']:
        self_E_l = self_energy_leads(E-eV_dict[l], l, N_x, N_y, t_L, t_C)
        tot_self_E_leads += self_E_l
        Gamma_l = 1j * (self_E_l - self_E_l.conj().T)
        Sigma_l +=  Gamma_l * f(E, mu_dict[l], T_dict[l])
    
    Sigma_lesser = Sigma_l * 1j
    G_S_R = np.linalg.inv(E * np.eye(2*N) - H_S - tot_self_E_leads)
    G_S_A = G_S_R.conj().T
    G_less = G_S_R @ Sigma_lesser @ G_S_A

    return G_less