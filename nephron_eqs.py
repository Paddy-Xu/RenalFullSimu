import numpy as np

from parameters import *

def C_protein(Q_A, P_GC, P_T0, x, C):
    return Glomerular.Kf/(Glomerular.L * Q_A * Glomerular.C_A) * C**2 * (
            P_GC - P_T0 - Glomerular.a * C - Glomerular.b * C**2)
def P_T_desc(sol_desc, z, P):
    Qz = sol_desc(z)[0]
    dP = - 1/60 * 1/133.322 * 1e-6 * 8 * Tubular.mu/(np.pi * Tubular.r_loop**4) * Qz
    return dP
def Q_T_proximal(Q_T0, z):
    C = Q_T0 - Tubular.keppa/Tubular.theta
    Q_T = C + Tubular.keppa/Tubular.theta * np.exp(-Tubular.theta * z)
    return Q_T

def Combined_desc(z, F):
    C_I = 150 + 150/0.3 * z
    Q_T, C = F
    dQ_T = - Tubular.Lv * Tubular.ns * (C_I - C) * 6e4 #  cm2 l osmol-1 s-1 * mmol =
    dC = (-C * dQ_T - 6e7 * Tubular.Ls * (C - C_I))/Q_T
    return dQ_T, dC


def Combined_desc_long_inter(z, F):
    C_I = 150 + 450/0.8 * z
    Q_T, C = F
    dQ_T = - Tubular.Lv * Tubular.ns * (C_I - C) * 6e4 #  cm2 l osmol-1 s-1 * mmol =
    dC = (-C * dQ_T - 6e7 * Tubular.Ls * (C - C_I))/Q_T
    return dQ_T, dC

# def Combined_desc_inter(z, F):
#     C_I = 150 + 400/0.8 * z
#     Q_T, C = F
#     dQ_T = - Tubular.Lv * Tubular.ns * (C_I - C) * 6e4 #  cm2 l osmol-1 s-1 * mmol =
#     dC = (-C * dQ_T - 6e7 * Tubular.Ls * (C - C_I))/Q_T
#
#     return dQ_T, dC

def Cs_asce(Q_T_desc_end, z, C, md_loc=0.5):
    C_I = (300 - 150/0.3 * z) if z < 0.3 else 150
    Vmax = Tubular.V_max1 if z < md_loc else Tubular.V_max2
    return - 6e7 * (Tubular.Ls * (C - C_I) + 1e3 * Vmax * C/(Tubular.Km + C)
             )/Q_T_desc_end

def Cs_asce_thin(Q_T_desc_end, z, C, long=True):
    # C_I = 275 - 75/0.5 * z

    C_I = 300
    Ls  = Tubular.Ls_long if long else Tubular.Ls_inter
    return - 6e7 * (Ls * (C - C_I))/Q_T_desc_end

# def Cs_asce_thin_inter(Q_T_desc_end, z, C):
#     C_I = 275 - 75/0.5 * z
#     # C_I = 275 - 75/0.25 * z
#
#     return - 6e7 * (Tubular.Ls_inter * (C - C_I))/Q_T_desc_end
#

def P_T_asce(Q_T_desc_end, P_end, z, asce_length=0.65):
    return (1/60 * 1/133.322 * 1e-6 * 8 * Tubular.mu/(np.pi * Tubular.r_loop**4) *
            Q_T_desc_end * (asce_length - z) + P_end)

def P_T_proximal(Q_T0, P_TZ, z):
    Z = 1
    # k1 = 1/60 * 1/133.322 * 1e-6 * 8 * Tubular.mu/(np.pi * Tubular.r_proximal**4
    #                                           ) * Tubular.keppa/Tubular.theta**2
    k2 = 1/60 * 1/133.322 * 1e-6 * 8 * Tubular.mu/(np.pi * Tubular.r_proximal**4
                                              )
    k1 = k2 * Tubular.keppa/Tubular.theta**2

    P_T = (k1 * (np.exp(-Tubular.theta * z) - np.exp(-Tubular.theta * Z)) +
           k2 * (Q_T0 - Tubular.keppa/Tubular.theta) * (Z - z) + P_TZ)

    return P_T

