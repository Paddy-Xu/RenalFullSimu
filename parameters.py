from dataclasses import dataclass

@dataclass
class Glomerular:
    C_A: float = 57 #g/l
    H_A: float = 0.5
    Kf: float = 2.5
    R_E: float = 0.209
    R_PC: float = 0.0702
    a: float = 0.1631 # mmHg l g-1
    b: float = 0.00294
    L: float = 10

@dataclass
class Tubular:
    Km: float = 20 #mM
    Lv: float = 2e-5 # cm2 l osmol-1 s-1
    ns: float = 2
    Ls: float = 3.4e-7 #cm2 s-1
    V_max1 = 1e-7 # mmol cm-1 s-1
    V_max2 = 0.65e-7
    alpha: float = 1.65e-2 # (nl min-1 mmHg-5)1/4
    beta: float = 0.924  # (nl min-1 mmHg-1)1/4
    theta: float = 1.3 # cm-1
    keppa: float = 33.6 #nl min-1 cm-1
    gamma: float = 1.33e-5 # cm mmHg-1
    # r0_proximal: float = 11e-3 # cm
    # r0_henle: float = 9.1e-4 # cm
    # Glomerular.L: float = 1 # cm
    mu: float = 7.2e-4 #g/(cm s) # Ps s
    r_proximal: float = 12e-4 #cm
    r_loop: float = 10e-4 #cm

    # Ls_long: float = 3.8e-7 #cm2 s-1
    # Ls_inter: float = 5.8e-7 #cm2 s-1

    Ls_long: float = 5.2e-7 #cm2 s-1
    Ls_inter: float = 5.6e-7 #cm2 s-1

    V_max1 = V_max1 * 1.22
    V_max2 = V_max2 * 1.22

@dataclass
class AA:

    sigma_e: float = 25.56 #dyn cm-2

    k_e: float = 17304 #cm-1
    r_e: float = 7 #micro-meter
    k_m: float = 8.75e6 # cm-2
    r_m: float = 12.5 #micro-meter


    sigma_m: float = 7.23e5 #dyn cm-2
    sigma_v: float = 1e6 # dyn s cm-2
    r_n: float = 10.04 # micro-meter no really usede
    h_init = 2 # micro-meter

    r_m = 14
    k_m = 3e6
    sigma_m = 10e5
    r_e = 5
    k_e = 1.2e4
    sigma_e = 6.4

@dataclass
class Myo:
    # lam: float = 0.2
    lam: float = 1

    # T0: float = 84.7 #dyn cm-1   #value from old paper
    T0: float = 81.1252  #dyn cm-1

    G: float = 0.06 # cm dyn -1 (in the range between 0.03 to 0.09)
    # G: float = 0.15 # cm dyn -1 (in the range between 0.03 to 0.09)

@dataclass
class TGF:

    C_half: float = 46 # mmol/l ### original value

    # C_half = 48   # mmol/l


    #C_half = 56 # mmol/l
    # k: float = 0.2 # l/mmol ###??????? 0 - 0.4 range?
    # ita_max: float = 0.057 # mmHg min nl-1
    # phi: float = 0.02 # mmHg min nl-1
    #
    #
    k: float = (70 + 100)/2 * 1e-3 # l/mmol
    ita_max: float = 0.091
    phi: float = 0.182

@dataclass
class TGF_INTER:
    k: float = 0.8328 * (70 + 100)/2 * 1e-3 # l/mmol
    ita_max: float = 0.1183
    phi: float = 0.2184


@dataclass
class TGF_LONG:
    k: float = 0.05/0.07 * (70 + 100)/2 * 1e-3 # l/mmol
    ita_max: float = 0.1456
    phi: float = 0.2583

