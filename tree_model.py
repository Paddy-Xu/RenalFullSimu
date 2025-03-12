import os.path

from afferent_arteriole import *
from vascular_tree_model import *
from helper_plot_funcs import *
import warnings
import nibabel as nib
import concurrent.futures
import sys
from tqdm import tqdm
# import multiprocessing
warnings.filterwarnings("error")

plt.rcParams.update({'figure.max_open_warning': 0})


def simu(Q, r_v, P_t_in, P_GC, type=0, only_myo=False):

    Q_A = 6 * 1e-5 * Q * (1 - Glomerular.H_A)  # micro-meter3 s-1 to nanolitter min-1
    P_v = (P_t_in + P_GC)/2

    if type == 0:
        func_glomerular_cur = func_glomerular
        AA_cur = AA_model

    elif type == 1:
        func_glomerular_cur = partial(func_glomerular,
                                  inter=True
                                  )
        AA_cur = partial(AA_model, type=type)
    elif type == 2:
        func_glomerular_cur = partial(func_glomerular,
                                  long=True
                                  )
        AA_cur = partial(AA_model, type=type)
    else:
        raise "this should never happen, nephron type needs to be within 3 types"

    try:
        root = fsolve(func_glomerular_cur, x0=10, args=(Q_A, P_GC))
        assert len(root) == 1
        P_0_final = root[0]
        P_0_final_again, Cs_md_final, Q_T0, Cs_desc_end, Q_T_desc_end, P_end, P_md = func_glomerular_cur(P_0_final, Q_A, P_GC=P_GC,
                                                                                            final=True)

    except NegativeFlowException as e:
        logging.info(f"FlowException in finding P0: with {Q_A = }  {P_GC = } Cs_md is set to be 0")
        print(f"FlowException in finding P0: with {Q_A = }  {P_GC = } Cs_md is set to be 0", end=', ')
        P_0_final, P_0_final_again, Cs_md_final, Q_T0, Cs_desc_end, Q_T_desc_end, P_end, P_md  = 0, 0, 0, 0, 0, 0, 0, 0
        print(e)

    except RuntimeWarning as e:
        print(f"RuntimeWarning in finding P0: with {Q_A = }  {P_GC = } no regulation happen here", end=', ')
        print(e)
        logging.warning(f"RuntimeWarning in finding P0: with {Q_A = }  {P_GC = } no regulation happen here")
        logging.warning(e)

        return r_v, 1/3 * Q_A, 0, 0, 0, 0, 0, 0, 1/3, 0, 0, 0, 0

    ratio = Q_T0/(Q_A*2)

    try:
        r_opt = so.brentq(AA_cur, a=1e-1, b=20, args=(Cs_md_final, P_v, False, only_myo))
        T1, x_myo, x_tgf, T_e, T_m = AA_cur(r_opt,  Cs_md_final, P_v, final=True, only_myo=only_myo)

    except RuntimeWarning as e:
        print(f"RuntimeWarning in finding r_new: with {Q_A = } {P_GC = } {Cs_md_final = } {P_v = }")
        print(e)
        logging.warning(e)

        T1, x_myo, x_tgf, T_e, T_m = AA_cur(r_v, Cs_md_final, P_v, final=True, only_myo=only_myo)
        
        return r_v, Q_T0, 0, 0, Cs_md_final, P_0_final, Cs_desc_end, Q_T_desc_end, ratio, T_e, T_m, P_end, P_md

    except Exception as e:
        print('This should never happen, other exception not RuntimeWarnings ', end=', ')
        logging.warning(f"Warning in finding r_new: with {Q_A = } {P_GC = } {Cs_md_final = } {P_v = }")
        print(e)
        sys.exit()

    logging.info(f'{P_0_final_again:.2f}, {Cs_md_final:.2f}, {r_opt:.2f}')


    return r_opt, Q_T0, x_myo, x_tgf, Cs_md_final, P_0_final, Cs_desc_end, Q_T_desc_end, ratio, T_e, T_m, P_end, P_md


class AutoRegulation:
    def __init__(self, P_in=100, lr=0.3, only_myo=False, num_iter=20, path=None, pop=True, relTol=1e-6):
        self.P_in = P_in * 133.322e-6
        self.lr = lr
        self.only_myo = only_myo
        self.num_iter = num_iter
        self.pt_file = None
        self.path = path
        self.pop = pop
        self.relTol = relTol

    def set_up_tree(self, root_loc=[588, 217, 650], t_flow=1.167e11 / 3e4, mu=3.6e-15, vspace=22.6, pt_file=None,
                    surface_name='surface.nii.gz'):

        self.surface_name = surface_name

        if pt_file is None:
            pt_file = os.path.join('data', f'kirchhoff_p_in = {self.P_in}.vtk')

        if not os.path.exists(pt_file):
            pt_file = 'final_tree.vtk'

        self.pt_file = pt_file

        vt = VtkNetworkAnalysis(pt_file, root_loc, vsize=vspace, mu=mu, t_flow=t_flow)
        vt.build()
        vt.to_directed()
        vt.tree = vt.di_tree

        vt.label_pressure_drop()
        vt.label_pressure_from_root_di(
            root_pressure=self.P_in * 1e-6
        )


        if 'kir' not in pt_file:
            vt.Kirchoff_law_PC(P_in=self.P_in, Q0=None)  # N/mm2
            vt.save(f'kirchhoff_p_in = {int(self.P_in/133.322e-6)}.vtk')

        if self.path is None:
            path = (f'path_new_fix_C_params_{int(self.P_in/133.322e-6)}, '
                    f'{self.lr = }_only_myo_{self.only_myo}_pop_{self.pop}')
        else:
            path = self.path

        if not os.path.exists(path):
            os.mkdir(path)
        logging.basicConfig(filename=os.path.join(path, 'app_tree_only.log'),
                            level=logging.WARNING,
                            filemode="w",
                            # format='%(asctime)s - %(levelname)s - %(message)s'
                            )


        print(f'P in = {int(self.P_in/133.322e-6)}')
        logging.warning(f'P in = {int(self.P_in/133.322e-6)}')
        logging.warning(f'{self.lr = }')
        print(f'{self.lr = }')

        all_terminals = [i for i in vt.tree.nodes if len(list(vt.tree.successors(i))) == 0]
        all_terminal_edges = [(list(vt.tree.predecessors(i))[0], i) for i in all_terminals]
        root = [i for i in vt.tree.nodes if len(list(vt.tree.predecessors(i))) == 0]
        assert len(root) == 1
        root = root[0]
        root_edge = [root, list(vt.tree.successors(root))[0]]
        all_terminal_flow = np.array([vt.tree[i][j]['flow'] for (i, j) in all_terminal_edges])
        assert np.all(all_terminal_flow == all_terminal_flow[0])
        # micro-meter3 s-1 # N s micro-meter-2

        all_terminal_pressures = np.array([vt.tree.nodes[i]['pressure_mmhg'] for i in all_terminals])
        all_terminal_pressures_in = np.array([vt.tree.nodes[i]['pressure_mmhg'] for (i, _) in all_terminal_edges])

        plt.figure()
        plt.hist(all_terminal_pressures, 30)
        plt.title(f'mean pressure = {np.mean(all_terminal_pressures)}, std = {np.std(all_terminal_pressures)}')
        plt.savefig(os.path.join(path, f"pressure_distribution_before_kirchhoff.png"), format="png", bbox_inches="tight")
        plt.clf()

        plt.figure()
        plt.hist(all_terminal_flow, 30)
        plt.title(f'mean terminal_flow = {np.mean(all_terminal_flow)}, std = {np.std(all_terminal_flow)}')
        plt.savefig(os.path.join(path, f"flow_distribution_before_kirchhoff.png"), format="png", bbox_inches="tight")
        plt.clf()

        logging.info('P_0_final, Cs_md_final, r_opt')

        in_flow = vt.tree[root_edge[0]][root_edge[1]]['flow_from_Kirchhoff']  # mm3/s
        in_flow = in_flow / 1e3 * 60

        for i in vt.tree.nodes:
            vt.tree.nodes[i]['pressure'] = vt.tree.nodes[i]['pressure_from_Kirchhoff'] * 1e-6  #
            vt.tree.nodes[i]['pressure_mmHg'] = vt.tree.nodes[i]['pressure_from_Kirchhoff_mmHg']

        print(f'in_flow = {in_flow:.4f} ml/min before iterations')
        logging.warning(f'in_flow = {in_flow:.4f} ml/min before iterations')

        all_terminal_type = np.zeros(len(all_terminals))
        if self.pop:
            surface = nib.load(surface_name).get_fdata()
            surface = np.array(np.where(surface)).T
            all_terminal_pos = np.array([vt.tree.nodes[i]['loc'] for i in all_terminals])

            sb = scipy.spatial.KDTree(surface)
            all_terminal_dist_to_surface = sb.query(all_terminal_pos)
            all_terminal_dist_to_surface = all_terminal_dist_to_surface[0]
            all_terminal_dist_to_surface *= 22.6

            quant_1 = np.quantile(all_terminal_dist_to_surface, 0.5)
            quant_2 = np.quantile(all_terminal_dist_to_surface, 0.7)


            all_terminal_type[np.logical_and(all_terminal_dist_to_surface > quant_1,
                                             all_terminal_dist_to_surface < quant_2)] = 1
            all_terminal_type[all_terminal_dist_to_surface > quant_2] = 2


        self.all_terminals = all_terminals
        self.all_terminal_edges = all_terminal_edges
        self.root = root
        self.root_edge = root_edge
        self.path = path
        self.logging = logging
        self.vt = vt
        self.all_terminal_type = all_terminal_type

    def re_label_tree(self):

        all_terminals = [i for i in self.vt.tree.nodes if len(list(self.vt.tree.successors(i))) == 0]
        all_terminal_edges = [(list(self.vt.tree.predecessors(i))[0], i) for i in all_terminals]
        root = [i for i in self.vt.tree.nodes if len(list(self.vt.tree.predecessors(i))) == 0]
        assert len(root) == 1
        root = root[0]
        root_edge = [root, list(self.vt.tree.successors(root))[0]]

        self.all_terminals = all_terminals
        self.all_terminal_edges = all_terminal_edges
        self.root = root
        self.root_edge = root_edge

    def auto_reg(self):

        in_flow_before = self.vt.tree[self.root_edge[0]][self.root_edge[1]]['flow_from_Kirchhoff']  # mm3/s
        in_flow_before = in_flow_before / 1e3 * 60

        in_flow_cur = in_flow_before

        for i in self.vt.tree.nodes:
            self.vt.tree.nodes[i]['pressure'] = self.vt.tree.nodes[i]['pressure_from_Kirchhoff'] * 1e-6  #
            self.vt.tree.nodes[i]['pressure_mmHg'] = self.vt.tree.nodes[i]['pressure_from_Kirchhoff_mmHg']

        self.logging.warning(f'in_flow = {in_flow_before:.4f} ml/min before iterations')

        for _iter in range(self.num_iter):
            all_terminal_radius = np.array([self.vt.tree[i][j]['radius'] for (i, j) in self.all_terminal_edges])
            all_terminal_pressures = np.array([self.vt.tree.nodes[i]['pressure_from_Kirchhoff_mmHg'] for i in self.all_terminals])
            all_terminal_pressures_in = np.array(
                [self.vt.tree.nodes[i]['pressure_from_Kirchhoff_mmHg'] for (i, _) in self.all_terminal_edges])
            all_terminal_flow = np.array([self.vt.tree[i][j]['flow_from_Kirchhoff'] for (i, j) in self.all_terminal_edges]) * 1e9
            # N mm-2/(N s mm -5) = mm3/s = 10^9  micro-meter3/s
            all_terminal_resistance = np.array([self.vt.tree[i][j]['resistance_mm'] for (i, j) in self.all_terminal_edges])

            P_v_all = (all_terminal_pressures_in + all_terminal_pressures) / 2
            T0_all = P_v_all * all_terminal_radius / (1e3 / 133.32)

            logging.warning(f'T0 = {np.mean(T0_all):.4f} at P0 = {int(self.P_in/133.322e-6)} at {_iter = }')
            df = np.array([all_terminal_radius, all_terminal_pressures, all_terminal_flow, T0_all,
                           all_terminal_resistance]).T
            if _iter == 0:
                plot_save_results_before_iteration(df, -1, self.path, self.P_in)

            else:
                plot_save_results_before_iteration(df, _iter, self.path, self.P_in)

            radius_before = all_terminal_radius

            #
            # res = [simu(all_terminal_flow[i], all_terminal_radius[i], all_terminal_pressures_in[i],
            #             all_terminal_pressures[i], self.all_terminal_type[i],
            #             only_myo=self.only_myo) for i in range(len(all_terminal_flow))]



            inputs = (all_terminal_flow, all_terminal_radius, all_terminal_pressures_in,
                         all_terminal_pressures, self.all_terminal_type,
                      np.array([self.only_myo] * len(all_terminal_flow)))

            with concurrent.futures.ProcessPoolExecutor() as executor:
                res = executor.map(simu, *inputs)

            executor = None

            res = np.array(list(res))

            res[:, 0] = radius_before + (res[:, 0] - radius_before) * self.lr

            for (r, q_t_0, x_myo_t, x_tgf_t, cs_md, p0, cs_d_end, qt_d_end, ratio, t_e, t_m, P_end, P_md, *_), (parent, i) in (
                    zip(res, self.all_terminal_edges)):
                self.vt.tree[parent][i]['radius'] = r
                self.vt.tree[parent][i]['filtration_rate'] = q_t_0
                self.vt.tree[parent][i]['x_myo'] = x_myo_t
                self.vt.tree[parent][i]['x_tgf'] = x_tgf_t
                self.vt.tree[parent][i]['Cs_md'] = cs_md
                self.vt.tree[parent][i]['P0'] = p0
                self.vt.tree[parent][i]['cs_d_end'] = cs_d_end
                self.vt.tree[parent][i]['qt_d_end'] = qt_d_end
                self.vt.tree[parent][i]['ratio'] = ratio
                self.vt.tree[parent][i]['T_e'] = t_e
                self.vt.tree[parent][i]['T_m'] = t_m
                self.vt.tree[parent][i]['P_end'] = P_end
                self.vt.tree[parent][i]['P_md'] = P_md

            self.vt.label_resistance()

            all_terminal_ratio = np.array([self.vt.tree[i][j]['ratio'] for (i, j) in self.all_terminal_edges])

            self.vt.Kirchoff_law_PC(P_in=self.P_in, ratio=all_terminal_ratio)  # N/mm2

            in_flow = self.vt.tree[self.root_edge[0]][self.root_edge[1]]['flow_from_Kirchhoff']  # mm3/s
            in_flow = in_flow / 1e3 * 60  # 7 ml min -1

            print(f'in_flow = {in_flow:.4f} ml/min at {_iter = }')
            logging.warning(f'in_flow = {in_flow:.4f} ml/min at {_iter = }')


            all_terminal_filtration = np.array([self.vt.tree[i][j]['filtration_rate'] for (i, j) in self.all_terminal_edges])
            all_terminal_x_myo = np.array([self.vt.tree[i][j]['x_myo'] for (i, j) in self.all_terminal_edges])
            all_terminal_x_tgf = np.array([self.vt.tree[i][j]['x_tgf'] for (i, j) in self.all_terminal_edges])
            all_terminal_pt0 = np.array([self.vt.tree[i][j]['P0'] for (i, j) in self.all_terminal_edges])
            all_terminal_Cs_md = np.array([self.vt.tree[i][j]['Cs_md'] for (i, j) in self.all_terminal_edges])
            all_terminal_Cs_d_end = np.array([self.vt.tree[i][j]['cs_d_end'] for (i, j) in self.all_terminal_edges])
            all_terminal_q_d_end = np.array([self.vt.tree[i][j]['qt_d_end'] for (i, j) in self.all_terminal_edges])
            all_terminal_radius = np.array([self.vt.tree[i][j]['radius'] for (i, j) in self.all_terminal_edges])
            all_terminal_T_e = np.array([self.vt.tree[i][j]['T_e'] for (i, j) in self.all_terminal_edges])
            all_terminal_T_m = np.array([self.vt.tree[i][j]['T_m'] for (i, j) in self.all_terminal_edges])
            all_terminal_T = all_terminal_T_e + all_terminal_T_m

            all_terminal_P_end =np.array([self.vt.tree[i][j]['P_end'] for (i, j) in self.all_terminal_edges])
            all_terminal_P_md =np.array([self.vt.tree[i][j]['P_md'] for (i, j) in self.all_terminal_edges])

            df = np.array([all_terminal_radius, all_terminal_pressures, all_terminal_flow, all_terminal_T,
                           all_terminal_resistance, all_terminal_filtration, all_terminal_x_myo, all_terminal_x_tgf,
                           all_terminal_Cs_md, all_terminal_T_e, all_terminal_T_m,
                           all_terminal_q_d_end, all_terminal_P_end, all_terminal_P_md]).T

            plot_save_results(df, in_flow, _iter, self.path)

            cur_flow_change = abs(in_flow - in_flow_cur) / (abs(in_flow) + 1e-8)

            print(f'relative flow change at {_iter = } cur_flow_change = {cur_flow_change:.5f}')
            logging.warning(f'relative flow change at {_iter = } cur_flow_change = {cur_flow_change:.5f}')

            if cur_flow_change < self.relTol:
                print(f'converged at {_iter = }')
                logging.warning(f'converged at {_iter = }')
                break

            else:
                in_flow_cur = in_flow
        # Converged

        return in_flow_before, in_flow


if __name__ == '__main__':

    surface_name = 'surface.nii.gz'
    num_iter = 25
    lr = 0.2
    only_myo = False
    P_in_range = np.arange(100, 201, 10)
    # P_in_range = (140, )

    pop = False

    root_loc = [588, 217, 650]
    vspace = 22.6

    relTol = 2e-5

    for P_in in P_in_range:
        pt_file = os.path.join('data', f'kirchhoff_p_in = {P_in}.vtk')
        if not os.path.exists(pt_file):
            pt_file = 'final_tree.vtk'

        auto = AutoRegulation(num_iter=num_iter, P_in=P_in, lr=lr,
                              only_myo=only_myo,
                              pop=pop,
                              # path=f'2_{P_in}_{1}',
                              relTol=relTol)
        auto.set_up_tree(surface_name=surface_name, pt_file=pt_file)

        before, after = auto.auto_reg()
        in_flow = auto.vt.tree[auto.root_edge[0]][auto.root_edge[1]]['flow_from_Kirchhoff']  # mm3/s
        in_flow = in_flow / 1e3 * 60

        logging.warning(f'in_flow = {in_flow:.4f} ml/min at {P_in = }, ')