
from nephron_solver import *
import scipy.optimize as so

def fx(x_tgf, x_myo):

    # x = x_tgf + x_myo

    x = x_myo

    fx = 3 * np.exp(x) / (np.exp(x) + 2 * np.exp(-0.5 * x))

    return fx

def h_v(r_v):
    r_0 = 10
    h_0 = 2
    return -r_v + np.sqrt(r_v**2 + 2 * h_0 * r_0 + h_0**2)

def AA_model(r_num, Cs_md, P_v, final=False, only_myo=False, type=0):

    if type == 0:
        x_tgf = 5 * 1 / Myo.lam * (TGF.ita_max -
                               TGF.phi / (1 + np.exp(TGF.k * (Cs_md - TGF.C_half)))
                               )
    elif type == 1:
        x_tgf = 5 * 1 / Myo.lam * (TGF_INTER.ita_max -
                               TGF_INTER.phi / (1 + np.exp(TGF_INTER.k * (Cs_md - TGF.C_half)))
                               )
    elif type == 2:
        x_tgf = 5 * 1 / Myo.lam * (TGF_LONG.ita_max -
                               TGF_LONG.phi / (1 + np.exp(TGF_LONG.k * (Cs_md - TGF.C_half)))
                               )
    else:
        raise Exception("Nephron type must be either 0, 1, or 2")

    x_tgf = 0 if only_myo else x_tgf

    x_myo = Myo.lam * Myo.G * (r_num * P_v / (1e3 / 133.32) - Myo.T0 *
                                   (1 - x_tgf))

    T_e = 1e-4 * AA.sigma_e * h_v(r_num) * (
               np.exp(1e-4 * AA.k_e * (r_num - AA.r_e)) - 1)  # dyn/cm2 * um = 10-4 dyn/cm
    T_m = 1e-4 * AA.sigma_m * h_v(r_num) * fx(x_tgf, x_myo) * np.exp(-1e-8 * AA.k_m * (r_num - AA.r_m) ** 2)

    if not final:
        return r_num * P_v / (1e3 / 133.32) - T_e - T_m
    else:
        return r_num * P_v / (1e3 / 133.32), x_myo, x_tgf, T_e, T_m
