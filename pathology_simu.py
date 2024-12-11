from tree_model import *

def stenosis(P_in=140, start_ratio = 0.2):

    auto = AutoRegulation(num_iter=15, P_in=P_in, lr=0.15, path=f'ratio_{P_in}_{start_ratio}')
    auto.set_up_tree()

    root_radius_orig = auto.vt.tree[auto.root_edge[0]][auto.root_edge[1]]['radius']
    in_flow = auto.vt.tree[auto.root_edge[0]][auto.root_edge[1]]['flow_from_Kirchhoff']  # mm3/s
    in_flow = in_flow / 1e3 * 60
    print(f'in_flow = {in_flow:.4f} ml/min before iterations with radius = {root_radius_orig:.4f} micr-meter')

    plt.figure()
    plt.scatter(1, in_flow)
    for ratio in np.arange(start_ratio, 0, -0.1):
        auto.vt.tree[auto.root_edge[0]][auto.root_edge[1]]['radius'] = root_radius_orig * ratio
        auto.vt.label_resistance()
        auto.vt.Kirchoff_law_PC(P_in=auto.P_in)  # N/mm2
        # vt.Kirchoff_law_efferent(P_in=P_in)  # N/mm2
        before, after = auto.auto_reg()
        in_flow = auto.vt.tree[auto.root_edge[0]][auto.root_edge[1]]['flow_from_Kirchhoff']  # mm3/s
        in_flow = in_flow / 1e3 * 60
        print(f'in_flow = {in_flow:.4f} ml/min at ratio = {ratio}, radius = {(root_radius_orig * ratio):.4f}'
              ,end=',')
        print(f'before = {before:.4f}')

        logging.warning(f'in_flow = {in_flow:.4f} ml/min at ratio = {ratio}, '
                        f'radius = {(root_radius_orig * ratio):.4f}, before = {before:.4f}')

        plt.scatter(ratio, in_flow)

def remove_terminals(P_in=140, start_ratio = 0.2):

    auto = AutoRegulation(num_iter=15, P_in=P_in, lr=0.2, path=f'ratio_{P_in}_{start_ratio}_remove')
    auto.set_up_tree()
    in_flow = auto.vt.tree[auto.root_edge[0]][auto.root_edge[1]]['flow_from_Kirchhoff']  # mm3/s
    in_flow = in_flow / 1e3 * 60
    print(f'in_flow = {in_flow:.4f} ml/min before iterations ')

    tree_copy = copy.deepcopy(auto.vt.tree)


    for ratio in np.arange(start_ratio, 1.01, 0.1):
        if ratio == 0:
            continue

        auto.vt.tree = tree_copy
        all_terminals = [i for i in auto.vt.tree.nodes if len(list(auto.vt.tree.successors(i))) == 0]
        # all_terminal_edges = [(list(auto.vt.tree.predecessors(i))[0], i) for i in all_terminals]
        nodes_to_delete = np.random.choice(all_terminals,  round(len(all_terminals) * ratio), replace=False)
        auto.vt.tree.remove_nodes_from(nodes_to_delete)
        auto.vt.reorder_nodes()
        assert len(auto.vt.tree.nodes) == np.max(auto.vt.tree.nodes) + 1

        auto.re_label_tree()

        auto.vt.label_resistance()
        auto.vt.Kirchoff_law_PC(P_in=auto.P_in)  # N/mm2
        # vt.Kirchoff_law_efferent(P_in=P_in)  # N/mm2
        before, after = auto.auto_reg()

        # in_flow = auto.vt.tree[root_edge[0]][root_edge[1]]['flow_from_Kirchhoff']  # mm3/s
        # in_flow = in_flow / 1e3 * 60

        print(f'in_flow = {after:.4f} ml/min at {int(ratio*100)}% terminals removed')
        print(f'before = {before:.4f}')

        logging.warning(f'in_flow = {after:.4f} ml/min at ratio = {ratio}, '
                        f'at {int(ratio*100)}% terminals removed, before = {before:.4f}')




if __name__ == '__main__':
    P_in_range = np.arange(140, 201, 10)
    for P_in in P_in_range:
        stenosis(P_in=140, start_ratio=0.9)