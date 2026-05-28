"""
Example usage of the p-wave NEGF framework
==========================================

This script demonstrates how to:
    1. Construct each implemented p-wave Hamiltonian.
    2. Calculate the corresponding lesser Green's function.
"""

import numpy as np
from pwave_negf import *




## ============================================================
## SYSTEM PARAMETERS
## ============================================================

N_x = 6
N_y = 6
N = N_x * N_y

eV = 0.1
E_F = 0.0
E = np.random.uniform(-5, 5)

t_s = 1.0
t_L = 1.0
t_C = 1.0

T_dict = {
    'L': 0.0,
    'R': 0.0,
    'U': 0.0,
    'D': 0.0
}




## ============================================================
## ITINERANT NON-COLLINEAR P-WAVE MODEL
## ============================================================

H_it = H_itinerant(
    N=N,
    N_x=N_x,
    t_s=t_s,
    t_x=0.5,
    t_y=0.5,
    phi=np.pi/2
)

# Diagonal voltage bias
mu_L = E_F + eV/2
mu_R = E_F - eV/2
mu_U = E_F - eV/2
mu_D = E_F + eV/2

G_less_it = G_lesser(
    E, E_F, H_it,
    mu_L, mu_R, mu_U, mu_D,
    N_x, N_y,
    t_L, t_C,
    T_dict
)

print("\nItinerant p-wave model:")
print("----------------")
print("Hamiltonian shape:", H_it.shape)
print("Lesser Green's function shape:", G_less_it.shape)




## ============================================================
## LOCALIZED NON-COLLINEAR P-WAVE MODEL
## ============================================================

phi = helix_pitch(N_x_uc=4)

H_loc = H_localized(
    N=N,
    N_x=N_x,
    t_s=t_s,
    J_sd=0.5,
    phi=phi,
    eta=1
)

# Vertical voltage bias
mu_L = E_F
mu_R = E_F
mu_U = E_F - eV/2
mu_D = E_F + eV/2

G_less_loc = G_lesser(
    E, E_F, H_loc,
    mu_L, mu_R, mu_U, mu_D,
    N_x, N_y,
    t_L, t_C,
    T_dict
)

print("\nLocalized p-wave model:")
print("----------------")
print("Hamiltonian shape:", H_loc.shape)
print("Lesser Green's function shape:", G_less_loc.shape)




## ============================================================
## COLLINEAR P-WAVE MODEL
## ============================================================

H_col = H_collinear(
    N=N,
    N_x=N_x,
    m_AFM=0.9,
    t_s=t_s,
    phi=0.5
)

# Diagonal voltage bias
mu_L = E_F + eV/2
mu_R = E_F - eV/2
mu_U = E_F - eV/2
mu_D = E_F + eV/2

G_less_col = G_lesser(
    E, E_F, H_col,
    mu_L, mu_R, mu_U, mu_D,
    N_x, N_y,
    t_L, t_C,
    T_dict
)

print("\nCollinear p-wave model:")
print("----------------")
print("Hamiltonian shape:", H_col.shape)
print("Lesser Green's function shape:", G_less_col.shape)