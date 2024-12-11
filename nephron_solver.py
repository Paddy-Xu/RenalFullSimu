
from nephron_eqs import *
import logging
from scipy.integrate import solve_ivp
from functools import partial
from scipy.optimize import fsolve

class NegativeFlowException(Exception):
    pass


def func_Q_T0(Q_T0, Q_A, P_T0, P_GC=None):

    if P_GC is None:
        P_GC = Glomerular.R_E * (Q_A / (1 - Glomerular.H_A) - Q_T0) + Glomerular.R_PC * Q_A / (1 - Glomerular.H_A)

    C_x = partial(C_protein, Q_A, P_GC, P_T0)

    sol = solve_ivp(C_x, [0, Glomerular.L], [Glomerular.C_A],  method='Radau', dense_output=True)
    if not sol.success:
        print(f'error in solving C_x, {sol.message = } with {Q_A = }, {P_GC = }, {P_T0 = }')
        logging.warning(f'error in solving C_x, {sol.message = } with {Q_A = }, {P_GC = }, {P_T0 = }')
        raise RuntimeWarning

    C_E = np.squeeze(sol.y[:, -1])

    Q_T0_new = (1 - Glomerular.C_A / C_E) * Q_A

    return Q_T0 - Q_T0_new



def func_glomerular(P_T0, Q_A, P_GC=None, final=False, long=False, inter=False, debug=False):

    assert not (long and inter)
    Cs_0 = 150
    Q_T0 = Q_A/3

    if not np.isscalar(Q_A):
        Q_T0 = np.ones(len(Q_A)) * Q_T0
        Cs_0 = np.ones(len(Q_A)) * Cs_0

    Cs_list = []
    t_list = []

    Q_T0 = fsolve(func_Q_T0, x0=Q_T0, args=(Q_A, P_T0, P_GC))

    if P_GC is None:
        P_GC = Glomerular.R_E * (Q_A / (1 - Glomerular.H_A) - Q_T0) + Glomerular.R_PC * Q_A / (1 - Glomerular.H_A)

    if np.isscalar(Q_A):
        assert len(Q_T0) == 1
        Q_T0 = Q_T0[0]
    else:
        assert len(Q_T0) == len(Q_A)

    ############    Proximal part    ############

    Q_T_proximal_end = Q_T_proximal(Q_T0, z=1)

    if Q_T_proximal_end < 0:
        raise NegativeFlowException('negative in Q_T_proximal_end')

    ############   descending limb    ############

    if long:
        sol = solve_ivp(Combined_desc_long_inter, [0, 0.8], [Q_T_proximal_end, Cs_0],
                        method='Radau', dense_output=True)
    elif inter:
        sol = solve_ivp(Combined_desc_long_inter, [0, 0.55], [Q_T_proximal_end, Cs_0],
                        method='Radau', dense_output=True)
    else:
        sol = solve_ivp(Combined_desc, [0, 0.3], [Q_T_proximal_end, Cs_0],
        method='Radau', dense_output=True)

    if not sol.success:
        print(f'error in solving desc, {sol.message = } with {Q_A = }, {P_GC = }, {Q_T_proximal_end = }')
        logging.warning(f'error in solving desc, {sol.message = } with {Q_A = }, {P_GC = }, {Q_T_proximal_end = }')
        raise RuntimeWarning

    z_desc_span = np.squeeze(sol.t)
    Q_T_desc_list = sol.y[0]
    if debug:
        Cs_desc_list = sol.y[1]
        Cs_list.append(Cs_desc_list)
        t_list.append(z_desc_span)

    Q_T_desc_end = np.squeeze(sol.y)[0, -1]
    Cs_desc_end = np.squeeze(sol.y)[1, -1]

    sol_desc = sol.sol # this is required to solve P_T backwards

    if Q_T_desc_end < 0:
        raise NegativeFlowException('negative in Q_T_desc_end')
    if Cs_desc_end < 0:
        raise NegativeFlowException('negative in Cs_desc_end')

    ############   ascending limb    ############

    if long or inter:

        Cs_asce_thin_cur = partial(Cs_asce_thin, Q_T_desc_end, long=long)

        asce_thin_len = 0.5 if long else 0.25

        sol = solve_ivp(Cs_asce_thin_cur, [0, asce_thin_len],
                        [Cs_desc_end],
                        method='Radau', dense_output=True
                        )

        if not sol.success:
            print(f'error in solving thin asce, {sol.message = } with {Q_A = }, {P_GC = }, {Q_T_proximal_end = }, '
                  f'{Cs_desc_end = }, {Cs_desc_end = }')
            logging.warning(f'error in solving thin desc, {sol.message = } with {Q_A = }, {P_GC = }, {Q_T_proximal_end = }, '
                            f'{Cs_desc_end = }, {Cs_desc_end = }')
            raise RuntimeWarning

        Cs_thin_end = np.squeeze(sol.y)[-1]

        if debug:
            Cs_list.append(np.squeeze(sol.y))
            t_list.append(np.squeeze(sol.t) + t_list[-1][-1])
        if long:
            z_range = np.concatenate([np.arange(0, 0.3, 0.001),
                                      np.arange(0.3, 0.32 - 1e-6, 0.001),
                                      np.arange(0.32, 0.47 - 1e-6, 0.001)]
                                     )
        else:
            z_range = np.concatenate([np.arange(0, 0.3, 0.001),
                                      np.arange(0.3, 0.4 - 1e-6, 0.001),
                                      np.arange(0.4, 0.55 - 1e-6, 0.001)]
                                     )

    else:
        z_range = np.concatenate([np.arange(0, 0.3, 0.001),
                                  np.arange(0.3, 0.5 - 1e-6, 0.001),
                                  np.arange(0.5, 0.65 - 1e-6, 0.001)]
                                 )

    if long or inter:
        asce_len = 0.47 if long else 0.55
        md_loc = 0.32 if long else 0.4

        Cs_asce_cur = partial(Cs_asce, Q_T_desc_end, md_loc=md_loc)

        sol = solve_ivp(Cs_asce_cur, [0, asce_len],
                        [Cs_thin_end], t_eval=z_range,
                        method='Radau', dense_output=True
                        # method='LSODA',
                        )
    else:
        Cs_asce_cur = partial(Cs_asce, Q_T_desc_end)
        sol = solve_ivp(Cs_asce_cur, [0, 0.65],
                        [Cs_desc_end], t_eval=z_range,
                        method='Radau', dense_output=True
                        # method='LSODA',
                        )


    if not sol.success:
        print(f'error in solving asce, {sol.message = } with {Q_A = }, {P_GC = }, {Q_T_proximal_end = }, '
              f'{Cs_desc_end = }, {Cs_desc_end = }')
        logging.warning(f'error in solving desc, {sol.message = } with {Q_A = }, {P_GC = }, {Q_T_proximal_end = }, '
                        f'{Cs_desc_end = }, {Cs_desc_end = }')
        raise RuntimeWarning

    if long:
        md_index = np.argmin(np.abs(z_range - 0.32))
    elif inter:
        md_index = np.argmin(np.abs(z_range - 0.4))
    else:
        md_index = np.argmin(np.abs(z_range - 0.5))

    y = np.squeeze(sol.y)
    t = np.squeeze(sol.t)
    assert len(y) == len(z_range)
    Cs_md = y[md_index]

    Cs_end = y[-1]

    if Cs_md < 0:
        raise NegativeFlowException('negative in Cs_md')


    if debug:
        off_set = t_list[-1][-1]
        Cs_list.append(y[:md_index])
        t_list.append(t[:md_index] + off_set)
        Cs_list.append(y[md_index+1:])
        t_list.append(t[md_index+1:] + off_set)


    logging.debug(f'{Cs_0 = }')
    logging.debug(f'{Cs_desc_end = }')
    logging.debug(f'{Cs_md = }')
    logging.debug(f'{Cs_end = }')

    ############   solve P_T at the end (note Q is constant at ascending, so Q_T_asce_end = Q_T_desc_end)     ############

    sol = fsolve(lambda x: x - Q_T_desc_end / (Tubular.alpha * x + Tubular.beta) ** 4, x0=7.2467)

    assert len(sol) == 1
    P_end = sol[0]

    logging.debug(f'{P_end = }')

    if long:
        asce_length = 0.97
    elif inter:
        asce_length = 0.8
    else:
        asce_length = 0.65

    ############   solve P_T at asce backwards    ############

    P_desc_end = P_T_asce(Q_T_desc_end, P_end, 0, asce_length=asce_length)

    logging.debug(f'{P_desc_end = }')

    ############   solve P_T at P desc  backwards    ############

    dP_desc_list = Q_T_desc_list[1:] * np.diff(z_desc_span) * 1 / 60 * 1 / 133.322 * 1e-6 * 8 * Tubular.mu / (
                np.pi * Tubular.r_loop ** 4)

    P_desc_list = P_desc_end + np.cumsum(dP_desc_list[::-1])
    P_proximal_end_est = P_desc_list[-1]
    logging.debug(f'{P_proximal_end_est = }')


    P_T_desc_cur = partial(P_T_desc, sol_desc)

    if long:
        desc_len = 0.8
    elif inter:
        desc_len = 0.55
    else:
        desc_len = 0.3

    sol = solve_ivp(P_T_desc_cur, [desc_len, 0],
                    [P_desc_end], method='LSODA',)

    P_proximal_end = np.squeeze(sol.y)[-1]

    logging.debug(f'{P_proximal_end = }')

    P_start = P_T_proximal(Q_T0, P_proximal_end, z=0)

    logging.debug(f'{P_start = }')

    # print(f'{P_T0 = }, {P_start = } {np.abs(P_start - P_T0)}')

    if not final:
        return P_T0 - P_start

    else:
        if debug:
            return P_T0, Cs_md, Q_T0, Cs_desc_end, Q_T_desc_end, Cs_list, t_list
        else:
            return P_T0, Cs_md, Q_T0, Cs_desc_end, Q_T_desc_end
