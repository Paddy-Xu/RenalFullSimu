import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import logging

def plot_save_results_before_iteration(df, _iter, path, P_in):

    all_terminal_radius, all_terminal_pressures, all_terminal_flow, T0_all, all_terminal_resistance = df.T

    if _iter == -1:

        plt.figure()
        plt.scatter(all_terminal_pressures, all_terminal_flow)
        plt.title(f'terminal pressure vs flow')
        plt.savefig(os.path.join(path, f"terminal pressure vs flow_0.png"), format="png", bbox_inches="tight")
        plt.clf()
        plt.figure()
        plt.hist(T0_all, 30)
        plt.title(f'T0 mean = {np.mean(T0_all)}, std = {np.std(T0_all)} dyn/cm')
        plt.savefig(os.path.join(path, f"T0_{0}.png"), format="png", bbox_inches="tight")
        plt.clf()
        print(f'T0 = {np.mean(T0_all):.4f} at P0 = {int(P_in / 133.322e-6)} at {_iter = }  dyn/cm')

        df = pd.DataFrame(df, columns=['radius', 'pressure', 'flow', 'T', 'resistance'])
        df.to_csv(os.path.join(path, 'before_iter.csv'), sep=',', index=False)
        QA0 = all_terminal_flow / 2 * 1e-6 * 60
        plt.figure()
        plt.hist(QA0, 30)
        plt.title(f'mean terminal_flow = {np.mean(QA0)}, std = {np.std(QA0)} nl/min')
        plt.savefig(os.path.join(path, f"QA_0.png"), format="png", bbox_inches="tight")
        plt.clf()

        Q_T00 = QA0 / 3

        plt.figure()
        plt.hist(Q_T00, 30)
        plt.title(f'mean QT0 = {np.mean(Q_T00)}, std = {np.std(Q_T00)} nl/min')
        plt.savefig(os.path.join(path, f"QT0_0.png"), format="png", bbox_inches="tight")
        plt.clf()

    plt.figure()
    plt.scatter(all_terminal_pressures, T0_all)
    plt.xlabel('terminal pressure')
    plt.ylabel('Tension')
    # plt.show()
    plt.savefig(os.path.join(path, f"tension_vs_pressure{_iter}.png"), format="png", bbox_inches="tight")
    plt.clf()

    all_terminal_flow1 = all_terminal_flow * 1e-6  # nl/s
    plt.figure()
    plt.hist(all_terminal_flow1, 30)
    plt.title(f'sum = {np.sum(all_terminal_flow1) / 1e6 * 60  :.4f} ml/min, '
              f'mean terminal_flow = {np.mean(all_terminal_flow1 * 60)}, std = {np.std(all_terminal_flow1 * 60)} nl/min')
    plt.savefig(os.path.join(path, f"flow_distribution_{_iter}.png"), format="png", bbox_inches="tight")
    plt.clf()

    plt.figure()
    plt.hist(all_terminal_resistance, 30)
    plt.title(f'mean = {np.mean(all_terminal_resistance):.4f}')
    plt.savefig(os.path.join(path, f"resistance_distribution_{_iter}.png"), format="png", bbox_inches="tight")
    plt.clf()

    plt.figure()
    plt.scatter(all_terminal_radius, all_terminal_pressures)
    plt.xlabel('radius')
    plt.ylabel('pressure')
    plt.title('all_terminal_pressure vs all_terminal_radius')
    plt.savefig(os.path.join(path, f"pressure_vs radius_{_iter}.png"), format="png", bbox_inches="tight")
    plt.clf()

    plt.figure()
    plt.scatter(all_terminal_radius, all_terminal_flow)
    plt.xlabel('radius')
    plt.ylabel('flow')
    plt.title('all_terminal_flow vs all_terminal_radius')
    plt.savefig(os.path.join(path, f"flow_vs_radius{_iter}.png"), format="png", bbox_inches="tight")
    plt.clf()

    plt.figure()
    plt.hist(all_terminal_pressures, 30)
    plt.title(f'mean = {np.mean(all_terminal_pressures):.4f}, std = {np.std(all_terminal_pressures)}')
    plt.savefig(os.path.join(path, f"pressure_distribution_{_iter}.png"), format="png", bbox_inches="tight")
    plt.clf()

    plt.figure()
    plt.scatter(all_terminal_flow, all_terminal_pressures)
    plt.title('all_terminal_pressure vs all_terminal_flow')
    plt.savefig(os.path.join(path, f"pressure vs flow_{_iter}.png"), format="png", bbox_inches="tight")
    plt.clf()

    plt.figure()
    plt.hist(all_terminal_radius, 30)
    plt.title(f'mean = {np.mean(all_terminal_radius):.4f}, std = {np.std(all_terminal_radius)}')
    plt.savefig(os.path.join(path, f"radius_distribution_{_iter}.png"), format="png", bbox_inches="tight")
    plt.clf()


def plot_save_results(df, in_flow, _iter, path):
    (all_terminal_radius, all_terminal_pressures, all_terminal_flow, all_terminal_T, all_terminal_resistance,
         all_terminal_filtration, all_terminal_x_myo, all_terminal_x_tgf, all_terminal_Cs_md, all_terminal_T_e,
         all_terminal_T_m) = df.T
    df = pd.DataFrame(df, columns=['radius', 'pressure', 'flow', 'T',
                                   'resistance', 'filtration', 'x_myo', 'x_tgf',
                                   'Cs_md', 'T_e', 'T_m'])

    df.to_csv(os.path.join(path, f'iter_{_iter}.csv'), sep=',', index=False)

    plt.figure()
    plt.scatter(all_terminal_T, all_terminal_Cs_md)
    plt.title('Cs_md vs T')
    plt.xlabel('T')
    plt.ylabel('Cs_md')
    plt.savefig(os.path.join(path,f"Cs_md_vs_Tension{_iter}.png"), format="png", bbox_inches="tight")
    plt.clf()


    plt.figure()
    plt.scatter(all_terminal_T, all_terminal_x_myo)
    plt.title('x_myo vs T')
    plt.xlabel('T')
    plt.ylabel('x_myo')
    plt.savefig(os.path.join(path,f"x_myo vs T_{_iter}.png"), format="png", bbox_inches="tight")
    plt.clf()


    plt.figure()
    plt.scatter(all_terminal_x_myo, all_terminal_Cs_md)
    plt.title('Cs_md vs x_myo')
    plt.xlabel('x_myo')
    plt.ylabel('Cs_md')
    plt.savefig(os.path.join(path,f"x_myo_vs_Cs_md{_iter}.png"), format="png", bbox_inches="tight")
    plt.clf()

    plt.figure()
    plt.scatter(all_terminal_Cs_md, all_terminal_x_tgf)
    plt.title('x_tgf vs Cs_md')
    plt.xlabel('Cs_md')
    plt.ylabel('x_tgf')
    plt.savefig(os.path.join(path,f"x_tgf vs Cs_md{_iter}.png"), format="png", bbox_inches="tight")
    plt.clf()

    plt.figure()
    plt.hist(all_terminal_T_e, 30)
    plt.title('T_e')
    plt.savefig(os.path.join(path,f"T_e_{_iter}.png"), format="png", bbox_inches="tight")
    plt.clf()

    plt.figure()
    plt.hist(all_terminal_T_m, 30)
    plt.title('T_m')
    plt.savefig(os.path.join(path,f"T_m_{_iter}.png"), format="png", bbox_inches="tight")
    plt.clf()

    plt.figure()
    plt.scatter(all_terminal_T_m, all_terminal_T_e)
    plt.title('Te vs Tm')
    plt.xlabel('Tm')
    plt.ylabel('Te')
    plt.savefig(os.path.join(path, f"Te vs Tm_{_iter}.png"), format="png", bbox_inches="tight")
    plt.clf()


    plt.figure()
    plt.scatter(all_terminal_flow, all_terminal_Cs_md)
    plt.xlabel('all_terminal_flow')
    plt.ylabel('all_terminal_Cs_md')
    plt.title('all_terminal_Cs_md vs all_terminal_flow')
    plt.savefig(os.path.join(path, f"all_terminal_Cs_md vs all_terminal_flow_{_iter}.png"), format="png", bbox_inches="tight")
    plt.clf()

    plt.figure()
    plt.scatter(all_terminal_radius, all_terminal_Cs_md)
    plt.title('all_terminal_Cs_md vs all_terminal_radius')
    plt.xlabel('all_terminal_radius')
    plt.ylabel('all_terminal_Cs_md')
    plt.savefig(os.path.join(path, f"all_terminal_Cs_md vs all_terminal_radius_{_iter}.png"), format="png", bbox_inches="tight")
    plt.clf()

    plt.figure()
    plt.scatter(all_terminal_x_myo, all_terminal_x_tgf)
    plt.xlabel('x_myo')
    plt.ylabel('x_tgf')
    plt.savefig(os.path.join(path,f"tgf_vs_myo_{_iter}.png"), format="png", bbox_inches="tight")
    plt.clf()

    plt.figure()
    plt.hist(all_terminal_x_tgf, 30)
    plt.title('tgf hist')
    plt.savefig(os.path.join(path,f"tgf_hist_{_iter}.png"), format="png", bbox_inches="tight")
    plt.clf()

    plt.figure()
    plt.hist(all_terminal_x_myo, 30)
    plt.title('myo hist')
    plt.savefig(os.path.join(path, f"myo_hist_{_iter}.png"), format="png", bbox_inches="tight")
    plt.clf()

    plt.figure()
    plt.hist(all_terminal_Cs_md, 30)
    plt.title(f'mean Cs_md = {np.mean(all_terminal_Cs_md):.4f}')
    plt.savefig(os.path.join(path,f"all_terminal_Cs_md_{_iter}.png"), format="png", bbox_inches="tight")
    plt.clf()


    plt.figure()
    plt.hist(all_terminal_filtration, 30)
    plt.savefig(os.path.join(path,f"filtration_distribution_{_iter}.png"), format="png", bbox_inches="tight")
    plt.clf()

    total_filtration = np.sum(all_terminal_filtration) * 1e-6  # ml/min

    r_mean = np.mean(all_terminal_radius)
    Cs_md_mean = np.mean(all_terminal_Cs_md)
      # mm3/s
    # in_flow = in_flow / 1e3 * 60  # 7 ml min -1
    print(f'in_flow = {in_flow:.4f} ml/min, r = {r_mean:.4f} Cs_md = {Cs_md_mean:.4f}'
          f' total_filtration = {total_filtration:.4f} ml/s at {_iter = }')

    logging.warning(f'in_flow = {in_flow:.4f} ml/min, r = {r_mean:.4f} Cs_md = {Cs_md_mean:.4f}'
          f' total_filtration = {total_filtration:.4f} ml/s at {_iter = }')


    print(f'**************** iter {_iter} finished ************************')
    logging.warning(f'**************** iter {_iter} finished ************************')

