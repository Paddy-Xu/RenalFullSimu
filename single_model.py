from vascular_tree_model import *
from afferent_arteriole import *
import warnings
import sys
from tree_model import simu
warnings.filterwarnings("error")

plt.rcParams.update({'figure.max_open_warning': 0})
#
# def simu(Q, r_v, P_t_in, P_GC, type=0, only_myo=False):
#
#     Q_A = 6 * 1e-5 * Q * (1 - Glomerular.H_A)  # micro-meter3 s-1 to nanolitter min-1
#     P_v = (P_t_in + P_GC)/2
#
#     if type == 0:
#         func_glomerular_cur = func_glomerular
#         AA_cur = AA_model
#
#     elif type == 1:
#         func_glomerular_cur = partial(func_glomerular,
#                                   inter=True
#                                   )
#         AA_cur = partial(AA_model, type=type)
#     elif type == 2:
#         func_glomerular_cur = partial(func_glomerular,
#                                   long=True
#                                   )
#         AA_cur = partial(AA_model, type=type)
#     else:
#         raise "this should never happen, nephron type needs to be within 3 types"
#
#     try:
#         root = fsolve(func_glomerular_cur, x0=10, args=(Q_A, P_GC))
#         assert len(root) == 1
#         P_0_final = root[0]
#         P_0_final_again, Cs_md_final, Q_T0, Cs_desc_end, Q_T_desc_end = func_glomerular_cur(P_0_final, Q_A, P_GC=P_GC,
#                                                                                             final=True)
#
#     except NegativeFlowException as e:
#
#         logging.info(f"FlowException in finding P0: with {Q_A = }  {P_GC = } Cs_md is set to be 0")
#         print(f"FlowException in finding P0: with {Q_A = }  {P_GC = } Cs_md is set to be 0", end=', ')
#         P_0_final, P_0_final_again, Cs_md_final, Q_T0, Cs_desc_end, Q_T_desc_end = 0, 0, 0, 0, 0, 0
#         print(e)
#
#     except RuntimeWarning as e:
#         print(f"RuntimeWarning in finding P0: with {Q_A = }  {P_GC = } no regulation happen here", end=', ')
#         print(e)
#         logging.warning(f"RuntimeWarning in finding P0: with {Q_A = }  {P_GC = } no regulation happen here")
#         logging.warning(e)
#
#         return r_v, 1/3 * Q_A, 0, 0, 0, 0, 0, 0, 1/3, 0, 0
#
#     ratio = Q_T0/(Q_A*2)
#
#     try:
#         r_opt = so.brentq(AA_cur, a=1e-1, b=20, args=(Cs_md_final, P_v, False, only_myo))
#         T1, x_myo, x_tgf, T_e, T_m = AA_cur(r_opt,  Cs_md_final, P_v, final=True, only_myo=only_myo)
#
#     except RuntimeWarning as e:
#         print(f"RuntimeWarning in finding r_new: with {Q_A = } {P_GC = } {Cs_md_final = } {P_v = }")
#         print(e)
#         logging.warning(e)
#
#         T1, x_myo, x_tgf, T_e, T_m = AA_cur(r_v, Cs_md_final, P_v, final=True, only_myo=only_myo)
#         return r_v, Q_T0, 0, 0, Cs_md_final, P_0_final, Cs_desc_end, Q_T_desc_end, ratio, T_e, T_m
#
#     except Exception as e:
#         print('this should never happen, other exception not RuntimeWarnings ', end=', ')
#         logging.warning(f"Warning in finding r_new: with {Q_A = } {P_GC = } {Cs_md_final = } {P_v = }")
#         print(e)
#         sys.exit()
#
#     logging.info(f'{P_0_final_again:.2f}, {Cs_md_final:.2f}, {r_opt:.2f}')
#
#
#     return r_opt, Q_T0, x_myo, x_tgf, Cs_md_final, P_0_final, Cs_desc_end, Q_T_desc_end, ratio, T_e, T_m

if __name__ == '__main__':

    '''
    after = pd.read_csv('/Users/px/q100, lr = 0.2/iter_10.csv')

    radius_1 = after['radius'].to_numpy()
    pressure_1 = after['pressure'].to_numpy()
    flow_1 = after['flow'].to_numpy()
    T_1 = after['T'].to_numpy()
    resistance_1 = after['resistance'].to_numpy()
    filtration_1 = after['filtration'].to_numpy()
    x_myo_1 = after['x_myo'].to_numpy()
    x_tgf_1 = after['x_tgf'].to_numpy()
    Cs_md_1 = after['Cs_md'].to_numpy()
    T_e_1 = after['T_e'].to_numpy()
    T_m_1 = after['T_m'].to_numpy()
    point1 = np.argmin(np.sum(np.abs(np.vstack([radius_1, pressure_1]).T - np.array([5.75, 70.59])), axis=1))
    point2 = np.argmin(np.sum(np.abs(np.vstack([radius_1, pressure_1]).T - np.array([6.43, 61.88])), axis=1))
    point3 = np.argmin(np.sum(np.abs(np.vstack([radius_1, pressure_1]).T - np.array([9.31, 57.92])), axis=1))
    point4 = np.argmin(np.sum(np.abs(np.vstack([radius_1, pressure_1]).T - np.array([13.36, 53.34])), axis=1))
    point5 = np.argmin(np.sum(np.abs(np.vstack([radius_1, pressure_1]).T - np.array([13.91, 48.52])), axis=1))
    '''

    ### indices that code above produces can be hard-coded.

    point1, point2, point3, point4, point5 = [27859, 19920,  7265,  7727,  6769]
    ### choose the current single nephron position

    cur_node = point4

    lr = 0.25
    only_myo = False

    num_iter = 20

    mu = 3.6e-15  # micro-meter3 s-1 # N s micro-meter-2
    vspace = 22.6
    root_loc = [588, 217, 650]

    R_e = 0.209 * 60 * 133.322 * 1e-3  # Ns mm-5 # around 1.67
    R_PC = 0.0702 * 60 * 133.322 * 1e-3  # Ns mm-5 around 0.56

    P_in_range = np.arange(80, 201, 10)

    relTol = 2e-5

    for P_in in P_in_range:
        print(f'{P_in = }')
        print('**************************************************************************************')

        pt_file = os.path.join('data', f'kirchhoff_p_in = {P_in}.vtk')

        if not os.path.exists(pt_file):
            pt_file = 'final_tree.vtk'

        vt = VtkNetworkAnalysis(pt_file, root_loc, vsize=vspace, mu=mu)
        vt.build()

        vt.to_directed()
        vt.tree = vt.di_tree

        P_in = P_in * 133.322e-6 # N/mm2

        vt.label_pressure_drop()
        vt.label_pressure_from_root_di(
            root_pressure=P_in * 1e-6
        )

        if 'kir' not in pt_file:
            vt.Kirchoff_law_PC(P_in=P_in, Q0=None)  # N/mm2
            vt.save(f'kirchhoff_p_in = {int(P_in/133.322e-6)}.vtk')

        log_path = f'{int(P_in / 133.322e-6)}, {lr = }_only_myo_{only_myo}'

        if not os.path.exists(log_path):
            os.mkdir(log_path)

        logging.basicConfig(filename=os.path.join(log_path, 'app_tree_only.log'),
                            level=logging.WARNING,
                            filemode="w",
                            # format='%(asctime)s - %(levelname)s - %(message)s'
                            )


        print(f'P in = {int(P_in/133.322e-6)}')
        logging.warning(f'P in = {int(P_in/133.322e-6)}')
        logging.warning(f'{lr = }')
        print(f'{lr = }')

        all_terminals = [i for i in vt.tree.nodes if len(list(vt.tree.successors(i))) == 0]

        N = len(all_terminals)

        all_terminal_edges = [(list(vt.tree.predecessors(i))[0], i) for i in all_terminals]

        root = [i for i in vt.tree.nodes if len(list(vt.tree.predecessors(i))) == 0]
        assert len(root) == 1
        root = root[0]
        root_edge = [root, list(vt.tree.successors(root))[0]]

        all_terminal_flow = np.array([vt.tree[i][j]['flow'] for (i, j) in all_terminal_edges])

        assert np.all(all_terminal_flow == all_terminal_flow[0])

        in_flow = vt.tree[root_edge[0]][root_edge[1]]['flow_from_Kirchhoff']  # mm3/s
        in_flow = in_flow / 1e3 * 60


        print(f'in_flow = {in_flow:.4f} ml/min before iterations')
        logging.warning(f'in_flow = {in_flow:.4f} ml/min before iterations')

        all_terminal_flow = np.array([vt.tree[i][j]['flow_from_Kirchhoff'] for (i, j) in all_terminal_edges])
        all_terminal_radius = np.array([vt.tree[i][j]['radius'] for (i, j) in all_terminal_edges])
        all_terminal_pressures = np.array([vt.tree.nodes[j]['pressure_from_Kirchhoff_mmHg'] for (i, j) in all_terminal_edges])
        all_terminal_pressures_in = np.array([vt.tree.nodes[i]['pressure_from_Kirchhoff_mmHg'] for (i, j) in all_terminal_edges])
        all_terminal_resistance = np.array([vt.tree[i][j]['resistance_mm'] for (i, j) in all_terminal_edges])
        all_terminal_length = np.array([np.linalg.norm(vt.tree.nodes[i]['loc'] - vt.tree.nodes[j]['loc']) * vspace
                                        for (i, j) in all_terminal_edges])


        l = all_terminal_length[cur_node]

        pressures = all_terminal_pressures[cur_node]
        pressures_in = all_terminal_pressures_in[cur_node]
        radius = all_terminal_radius[cur_node]
        flow = all_terminal_flow[cur_node] #* 1e-9
        resistance = all_terminal_resistance[cur_node]
        # N s mm -5
        assert np.isclose(resistance,
                          (pressures_in - pressures) * 133.322e-6 / (flow))

        resistance_pre = (P_in - pressures_in * 133.322e-6) / (flow)

        print(N * flow *60 /1e3)

        in_flow_cur = N * flow / 1e3 * 60  # 7 ml min -1


        for _iter in range(num_iter):

            res = np.array(simu(flow/1e-9, radius, pressures_in, pressures,type=0, only_myo=only_myo))

            (r, q_t_0, x_myo_t, x_tgf_t, cs_md, p0, cs_d_end, qt_d_end, ratio, t_e, t_m, *_) = res

            radius = radius + (r - radius) * lr

            resistance = 8 * mu * l * 1e15 / (np.pi * radius ** 4)  # N s mm -5

            resistance_all = resistance_pre + resistance + R_e + R_PC
            # P in  N/mm2,  R in N s mm -5, Q_t0 in mm3/s
            q_t_0 = q_t_0 / 60 * 1e-3  # nl/min to mm3/s

            # Q_t represents the equivalent flow of the two parts,
            # the real flow Q associated with the current afferent arteriole can thus be
            # recovered by equal pressure drop property
            # Q_t = P_in/resistance_all, Q = Q_t + R_e * q_t_0/resistance_all
            # This gives ===> Q = (P_in + R_e * q_t_0)/(resistance_all)
            flow = (P_in + R_e * q_t_0)/(resistance_all)
            # flow in mm3/s

            P_GC = P_in - flow * (resistance_pre + resistance)

            assert np.isclose(P_GC, R_e * (flow - q_t_0) + R_PC * flow)
            # calculate P_GC from backward or inward should be the same

            pressures = P_GC/(133.322e-6)
            pressures_in = (P_in - flow * resistance_pre)/(133.322e-6)
            # Q_nl_s = Q * 1e3

            in_flow = N * flow / 1e3 * 60  # 7 ml min -1

            cur_flow_change = abs(in_flow - in_flow_cur) / (abs(in_flow) + 1e-8)

            print(f'relative flow change at {_iter = } cur_flow_change = {cur_flow_change:.5f}')
            logging.warning(f'relative flow change at {_iter = } cur_flow_change = {cur_flow_change:.5f}')


            print(f'pressures = {pressures:.4f}', end=', ')
            print(f'flow = {flow*1e3:.4f}', end=', ')
            print(f'in_flow = {in_flow:.4f}', end=', ')

            print(f'radius = {radius:.4f}')

            print(f'**************** iter {_iter} finished ************************')


            if cur_flow_change < relTol:
                print(f'converged at {_iter = }')
                logging.warning(f'converged at {_iter = }')
                break
            else:
                in_flow_cur = in_flow



