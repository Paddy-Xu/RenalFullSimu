import sys

# from matplotlib import rcParams
# rcParams.update({'figure.autolayout': True})
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

import scipy
import matplotlib.pyplot as plt
import numpy as np
import os
import networkx as nx
import copy
import pyvista

from helper_funcs import *

class VtkNetwork:

    def __init__(self, pt_file=None, root_loc=None, vsize=22.6):

        self.pt_file = pt_file
        self.root_loc = root_loc
        self.vsize = vsize

    def build(self):
        mesh = pyvista.read(self.pt_file,
                            # force_ext='.vtk'
                                      )
        points = mesh.points

        line = mesh.lines

        if 'radius' in mesh.point_data:
            node_radius = mesh.point_data['radius']
        elif 'node_radius' in mesh.point_data:
            node_radius = mesh.point_data['node_radius']
        elif 'radius' in mesh.cell_data:
            node_radius = None

        self.tree = nx.Graph()

        for i, p in enumerate(points):
            self.tree.add_node(i, loc=np.array(p), fixed=False, root=False,
                               node_radius=node_radius[i] if node_radius is not None else None
                               )

        i = 1
        while i < len(line):
            node1, node2 = line[i], line[i + 1]
            cur_dict = {k: v[i//3] for k, v in mesh.cell_data.items()}
            self.tree.add_edge(node1, node2, **cur_dict)
            i += 3

        for edge in self.tree.edges:
            a, b = edge
            self.tree[a][b]['radius_neg'] = - self.tree[a][b]['radius']

        if self.root_loc is not None:
            self.root = self.find_n_with_coord(self.root_loc)
            self.tree.nodes[self.root]['fixed'] = True
            self.tree.nodes[self.root]['root'] = True

        Gcc = sorted(nx.connected_components(self.tree), key=len, reverse=True)
        self.tree = nx.Graph(self.tree.subgraph(Gcc[0]))

        self.update_order()

    def mst(self):

        tree = nx.minimum_spanning_tree(self.tree, weight='radius_neg')
        root = self.root

        self.tree = tree
        self.root = self.find_n_with_coord(self.root_loc)

        self.tree.nodes[root]['fixed'] = True
        self.tree.nodes[root]['root'] = True

        self.update_order()
        self.to_directed()


        # di_tree = tree.to_directed()

    def to_directed(self):
        def bfs_to_direct(node):
            neighbors = list(self.tree.neighbors(node))

            if self.tree.nodes[node]['root']:
                children = neighbors
            else:
                neighbor_order = np.array([self.tree.nodes[n]['level'] for n in neighbors])
                if -1 in neighbor_order:
                    root_idx = np.where(neighbor_order == -1)[0][0]
                else:
                    root_idx = np.argmax(neighbor_order)

                root_idx = neighbors[root_idx]

                children = [i for i in neighbors if i != root_idx]

            if len(children) == 0:
                return

            for child in children:
                attrs = self.tree[node][child]
                self.di_tree.add_edge(node, child, **attrs)

            for child in children:
                bfs_to_direct(child)

        self.di_tree = nx.DiGraph()

        for n in self.tree.nodes:
            attrs = self.tree.nodes[n]
            self.di_tree.add_node(n, **attrs)

        root = [i for i in self.tree.nodes if self.tree.nodes[i]['root']][0]

        bfs_to_direct(root)


    def get_depth(self, root=None):
        if root is None:
            all_nodes = np.array([i for i in self.tree.nodes])

            orders = np.array([self.tree.nodes[n]['level'] for n in all_nodes])

            root = all_nodes[np.where(orders == -1)[0][0]]

            neighbors = np.array(list(self.tree.neighbors(root)))

            depth = 1 + max([self.get_depth(i) for i in neighbors])
            self.tree.nodes[root]['depth'] = depth
            return depth

        neighbors = np.array(list(self.tree.neighbors(root)))
        if len(neighbors) == 1:
            self.tree.nodes[root]['depth'] = 1
            return 1
        else:
            neighbor_orders = np.array([self.tree.nodes[n]['level'] for n in neighbors])
            if -1 in neighbor_orders:
                root_idx = np.where(neighbor_orders == -1)[0][0]
            else:
                root_idx = np.argmax(neighbor_orders)

            depth = 1 + max([self.get_depth(self.tree, i) for i in neighbors if i != neighbors[root_idx]])
            self.tree.nodes[root]['depth'] = depth
            return depth

    def save(self, file='sanity_check3.vtk'):

        for i in self.tree.nodes:
            self.tree.nodes[i]['ind'] = i

        nodes_inds = self.tree.nodes
        indices = sorted(nodes_inds)

        # points = [self.tree.nodes[i]['loc'] for i in wrong]

        points = [nodes_inds[i]['loc'] for i in indices]
        points = np.array(points)

        # points = points - np.mean(points, 0)
        # points *= self.vsize

        edges_inds = self.tree.edges
        edges_inds = np.array(edges_inds)

        radius_list = [self.tree[n1][n2]['radius'] for n1, n2 in edges_inds]
        radius_list = np.array(radius_list)

        if np.max(edges_inds) >= len(nodes_inds):
            map = dict(zip(indices, np.arange(len(indices))))
            edges_inds_new = np.vectorize(map.get)(edges_inds)
            # radius_list = np.vectorize(map.get)(radius_list)
        else:
            edges_inds_new = edges_inds

        lines = np.hstack([np.ones((len(edges_inds_new), 1), dtype=np.uint8) * 2, edges_inds_new])
        lines = np.hstack(lines)
        mesh = pyvista.PolyData(points,
                                lines=lines
                                )

        for key in self.tree[edges_inds[0][0]][edges_inds[0][1]]:
            edge_feature = [self.tree[n1][n2][key] for n1, n2 in edges_inds]
            edge_feature = np.array(edge_feature)

            assert len(edge_feature) > 0 and len(edge_feature) == mesh.n_lines
            mesh.cell_data[key] = edge_feature

        for key in self.tree.nodes[indices[0]]:
            if key in ('loc', 'fixed', 'root'):
                continue
            node_feature = [self.tree.nodes[i][key] for i in indices]
            assert len(node_feature) > 0 and len(node_feature) == mesh.n_points
            mesh.point_data[key] = np.array(node_feature)

        mesh.save(file)

    def update_order(self, mode='level'):
        for n in self.tree.nodes:
            self.tree.nodes[n][mode] = 1 if self.tree.degree[n] <= 1 and not self.tree.nodes[n]['fixed'] else 0
            if self.tree.nodes[n]['fixed']:
                self.tree.nodes[n][mode] = -1
        count_no_label = len(self.tree.nodes)
        cur_order = 1
        while count_no_label != 0:
            for node in self.tree.nodes:
                if self.tree.nodes[node][mode] != 0 or len(list(self.tree.neighbors(node))) == 0:
                    continue
                neighbor_orders = np.array([self.tree.nodes[n][mode] for n in self.tree.neighbors(node)])
                if -1 in neighbor_orders and np.count_nonzero(neighbor_orders == 0) >= 1:
                    continue
                if -1 not in neighbor_orders and np.count_nonzero(neighbor_orders == 0) > 1:
                    continue
                max_order = np.max(neighbor_orders)
                max_count = np.count_nonzero(neighbor_orders == max_order)


                self.tree.nodes[node][mode] = max_order if max_count == 1 and mode == 'HS' else max_order + 1

            count_no_label = np.count_nonzero(np.array([self.tree.nodes[n][mode] for n in self.tree.nodes]) == 0)

    def update_final_order(self, mode='HS'):
        """
        Update order when no further operations will be applied.
        """

        for n in self.tree.nodes:
            self.tree.nodes[n][mode] = 1 if self.tree.degree[n] == 1 else 0
        count_no_label = len(self.tree.nodes)
        while count_no_label != 0:
            for node in self.tree.nodes:
                if self.tree.nodes[node][mode] != 0 or len(list(self.tree.neighbors(node))) == 0:
                    continue
                neighbor_orders = np.array([self.tree.nodes[n][mode] for n in self.tree.neighbors(node)])

                if np.count_nonzero(neighbor_orders == 0) > 1:  # has to be second endpoint
                    continue
                max_order = np.max(neighbor_orders)
                max_count = np.count_nonzero(neighbor_orders == max_order)
                self.tree.nodes[node][mode] = max_order if max_count == 1 and mode == 'HS' else max_order + 1
            count_no_label = np.count_nonzero(np.array([self.tree.nodes[n][mode] for n in self.tree.nodes]) == 0)

        node = [n for n in self.tree.nodes if self.tree.nodes[n][mode] == -1][0]

        neighbor_orders = np.array([self.tree.nodes[n][mode] for n in self.tree.neighbors(node)])
        max_order = np.max(neighbor_orders)
        max_count = np.count_nonzero(neighbor_orders == max_order)
        self.tree.nodes[node][mode] = max_order if max_count == 1 and mode == 'HS' else max_order + 1

    def sum_children_distances(self, n):

        neighbors = np.array(list(self.tree.neighbors(n)))
        neighbor_orders = np.array([self.tree.nodes[n]['level'] for n in neighbors])
        root = -1 if -1 in neighbor_orders else np.max(neighbor_orders)
        children = np.array([neighbors[i] for i in range(len(neighbors)) if neighbor_orders[i] != root])

        root = neighbors[np.where(neighbor_orders == root)[0][0]]
        if len(neighbors) == 1:
            return np.linalg.norm(self.tree.nodes[root]['loc'] - self.tree.nodes[n]['loc'])

        cur = 0

        for i in children:
            cur += self.sum_children_distances(i)

        return cur

    def label_dist_from_root(self, node=None):
        if node is None:
            node = [n for n in self.tree.nodes if self.tree.nodes[n]['root']][0]
            self.tree.nodes[node]['sub_length_root'] = 0
            children = np.array(list(self.tree.neighbors(node)))
        else:
            neighbors = np.array(list(self.tree.neighbors(node)))
            neighbor_orders = np.array([self.tree.nodes[n]['level'] for n in neighbors])
            root = -1 if -1 in neighbor_orders else np.max(neighbor_orders)
            children = np.array([neighbors[i] for i in range(len(neighbors)) if neighbor_orders[i] != root])
            root = neighbors[np.where(neighbor_orders == root)[0][0]]
            self.tree.nodes[node]['sub_length_root'] = self.tree.nodes[root]['sub_length_root'] + np.linalg.norm(
                self.tree.nodes[root]['loc'] - self.tree.nodes[node]['loc'])

        for i in children:
            self.label_dist_from_root(i)

    def label_dist_to_end(self, node=None):
        if node is None:
            node = [n for n in self.tree.nodes if self.tree.nodes[n]['root']][0]
            self.tree.nodes[node]['sub_length_end'] = 0
            children = np.array(list(self.tree.neighbors(node)))
        else:
            neighbors = np.array(list(self.tree.neighbors(node)))
            neighbor_orders = np.array([self.tree.nodes[n]['level'] for n in neighbors])
            root = -1 if -1 in neighbor_orders else np.max(neighbor_orders)
            children = np.array([neighbors[i] for i in range(len(neighbors)) if neighbor_orders[i] != root])
            root = neighbors[np.where(neighbor_orders == root)[0][0]]
            self.tree.nodes[node]['sub_length_root'] = self.tree.nodes[root]['sub_length_root'] + np.linalg.norm(
                self.tree.nodes[root]['loc'] - self.tree.nodes[node]['loc'])

        for i in children:
            self.label_dist_from_root(i)


    def max_children_distances(self, n):

        neighbors = np.array(list(self.tree.neighbors(n)))
        neighbor_orders = np.array([self.tree.nodes[n]['level'] for n in neighbors])
        root = -1 if -1 in neighbor_orders else np.max(neighbor_orders)
        children = np.array([neighbors[i] for i in range(len(neighbors)) if neighbor_orders[i] != root])

        root = neighbors[np.where(neighbor_orders == root)[0][0]]
        if len(neighbors) == 1:
            return np.linalg.norm(self.tree.nodes[root]['loc'] - self.tree.nodes[n]['loc'])

        cur = np.max([self.max_children_distances(i) for i in children]) + np.linalg.norm(
            self.tree.nodes[root]['loc'] - self.tree.nodes[n]['loc'])

        return cur

    def save_results(self, work_dir):
        """
        Stores the resulting network structure.
        """

        if not os.path.exists(work_dir):
            os.mkdir(work_dir)

        coord_file = os.path.join(work_dir, 'test_1_result_coords.npy')
        connection_file = os.path.join(work_dir, 'test_1_result_connections.npy')
        radius_file = os.path.join(work_dir, 'test_1_result_radii.npy')
        order_file = os.path.join(work_dir, 'test_1_result_HS_order.npy')
        level_file = os.path.join(work_dir, 'test_1_result_level_order.npy')

        nodes = dict()
        coords = list()
        connections = list()
        radii = list()
        order = list()
        l_order = list()
        # self.update_final_order('HS')
        # self.update_final_order('level')

        for edge in list(self.tree.edges):
            node1, node2 = edge
            for node in edge:
                if not node in nodes:
                    nodes[node] = len(coords)
                    coords.append(self.tree.nodes[node]['loc'])
                    order.append(self.tree.nodes[node]['HS'])
                    l_order.append(self.tree.nodes[node]['level'])
            connections.append([nodes[node1], nodes[node2]])
            radii.append(abs(self.tree[node1][node2]['radius']))

        np.save(coord_file, coords)
        np.save(connection_file, connections)
        np.save(radius_file, radii)
        print("Save coords, edges and radius.")
        np.save(order_file, order)
        np.save(level_file, l_order)
        print("Save orders.")

    def find_n_with_coord(self, coords=np.array([1, 2, 3])):
        all_nodes = [i for i in self.tree.nodes]
        all_loc = np.array([self.tree.nodes[i]['loc'] for i in all_nodes])
        all_dist = np.sum((all_loc - np.array(coords)) ** 2, axis=1)
        return all_nodes[np.argmin(all_dist)]

    def remove_intermediate(self):

        max_orders = np.max(np.array([self.tree.nodes[n]['level'] for n in self.tree.nodes]))
        root = [n for n in self.tree.nodes if self.tree.nodes[n]['level'] == max_orders]
        assert len(root) == 1
        root = root[0]
        all_nodes = [n for n in self.tree.nodes]
        for n in all_nodes:
            neighbors = list(self.tree.neighbors(n))
            if len(neighbors) == 2 and n != root:
                left = neighbors[0]
                right = neighbors[1]

                dis = np.linalg.norm(self.tree.nodes[left]['loc'] - self.tree.nodes[right]['loc'])
                r = np.mean([self.tree[left][n]['radius'], self.tree[n][right]['radius']])
                self.tree.add_edge(left, right, radius=r, radius_neg=-r, length=dis)

                self.tree.remove_node(n)

        self.update_order()

        return


    def make_node_radius(self):

        for n in self.tree.nodes:
            self.tree.nodes[n]['node_radius'] = 0

        for n in self.tree.edges:
            a, b = n[0], n[1]
            self.tree.nodes[b]['node_radius'] = max(self.tree[a][b]['radius'], self.tree.nodes[b]['node_radius'])
            self.tree.nodes[a]['node_radius'] = max(self.tree[a][b]['radius'], self.tree.nodes[a]['node_radius'])

        root = [i for i in self.tree.nodes if self.tree.nodes[i]['node_radius'] == 0]
        print(len(root))

        for n in self.tree.edges:
            for i in n:
                self.tree.nodes[i]['node_radius_scaled'] = self.tree.nodes[i]['node_radius'] / self.vsize

        # if len(root) == 1:
        #     self.tree.nodes[root[0]]['radius'] = np.max(mesh.cell_data['radius'])

    def rescale_new_terminals(self, mu=10.04, sigma=0.14, flow=None):
        for n in self.tree.nodes:
            if self.tree.nodes[n]['level'] != 0 and len(list(self.tree.neighbors(n))) == 0:
                root = list(self.tree.predecessors(n))[0]
                self.tree[root][n]['radius'] = np.random.normal(loc=mu, scale=sigma)
                self.tree[root][n]['flow'] = flow

    def rescale_fixed_radius_flow(self, mode='level'):

        for n in self.tree.nodes:
            self.tree.nodes[n][mode] = 1 if self.tree.degree[n] <= 1 else 0
            if self.tree.nodes[n]['root']:
                self.tree.nodes[n][mode] = -1

        # count_no_label = self.node_count
        count_no_label = len(self.tree.nodes)


        while count_no_label != 0:
            for node in self.tree.nodes:
                if self.tree.nodes[node][mode] != 0 or len(list(self.tree.neighbors(node))) == 0:
                    continue

                neighbor_orders = np.array([self.tree.nodes[n][mode] for n in self.tree.neighbors(node)])

                root = list(self.tree.predecessors(node))[0]

                if np.count_nonzero(neighbor_orders == 0) >= 1:
                    continue

                max_order = np.max(neighbor_orders)
                self.tree.nodes[node][mode] = max_order + 1

                r_child = np.array([self.tree[node][n]['radius'] for n in self.tree.neighbors(node)])

                f_child = np.array([self.tree[node][n]['flow'] for n in self.tree.neighbors(node)])

                r_root = np.power(np.sum(r_child ** 3), 1 / 3)
                f_root = np.sum(f_child)

                self.tree[root][node]['radius'] = r_root
                self.tree[root][node]['flow'] = f_root

            count_no_label = np.count_nonzero(np.array([self.tree.nodes[n][mode] for n in self.tree.nodes]) == 0)


class VtkNetworkAnalysis(VtkNetwork):

    def __init__(self, pt_file=None, root_loc=None, vsize=22.6, mu=0, t_flow=0):
        super().__init__(pt_file, root_loc, vsize)

        self.mu = mu

    def build(self):

        mesh = pyvista.read(self.pt_file)

        points = mesh.points
        line = mesh.lines

        self.tree = nx.Graph()

        point_data = mesh.point_data
        cell_data = mesh.cell_data

        point_feature_all = {key: value for (key, value) in point_data.items()}

        for i, p in enumerate(points):
            # point_feature = {}
            # for key in point_data.keys():
            #     point_feature[key] = point_data[key][i]
            point_feature = {key: value[i] for (key, value) in point_feature_all.items()}

            self.tree.add_node(i, loc=np.array(p),
                               root=False, fixed=False,
                               **point_feature
                               )

        i = 1
        while i < len(line):
            node1, node2 = line[i], line[i + 1]

            edge_feature = {}
            for key in cell_data.keys():
                edge_feature[key] = cell_data[key][i // 3]
            # radius = mesh.cell_data['radius'][i // 3]

            self.tree.add_edge(node1, node2,
                               **edge_feature,
                               )

            i += 3

        for edge in self.tree.edges:
            a, b = edge
            self.tree[a][b]['radius_neg'] = - self.tree[a][b]['radius']

        all_nodes = [i for i in self.tree.nodes]

        if 'level' in self.tree[all_nodes[0]]:
            all_levels = np.array([self.tree.nodes[i]['level'] for i in all_nodes])
            if -1 in all_levels:
                root_idx = np.where(all_levels == -1)[0][0]
            else:
                rooot_idx = np.argmax(all_levels)

            self.root = all_nodes[root_idx]

        elif self.root_loc is not None:
            self.root = self.find_n_with_coord(self.root_loc)

        self.tree.nodes[self.root]['root'] = True
        self.tree.nodes[self.root]['fixed'] = True

        Gcc = sorted(nx.connected_components(self.tree), key=len, reverse=True)
        self.tree = nx.Graph(self.tree.subgraph(Gcc[0]))

        self.remove_intermediate()

        self.update_order()

        self.node_count = len(list(self.tree.nodes))

    def add_vessel(self, node, neighbor_node, radius=None, flow=None, sub=-1):
        """
        Adds a vessel between two nodes.

        Parameters
        --------------------
        node -- one endpoint
        neighbor_node -- the other endpoint
        """

        r = radius
        f = flow
        dis = np.linalg.norm(self.tree.nodes[node]['loc'] - self.tree.nodes[neighbor_node]['loc']
                             ) * self.vsize

        self.tree.add_edge(node, neighbor_node, radius=r, flow=f, length=dis, sub=sub)

    def remove_intermediate(self):
        max_orders = np.max(np.array([self.tree.nodes[n]['level'] for n in self.tree.nodes]))
        root = [n for n in self.tree.nodes if self.tree.nodes[n]['level'] == max_orders]
        assert len(root) == 1
        root = root[0]
        all_nodes = [n for n in self.tree.nodes]
        for n in all_nodes:
            neighbors = list(self.tree.neighbors(n))
            if len(neighbors) == 2 and n != root:
                left = neighbors[0]
                right = neighbors[1]

                edge_feature = {}
                for key in self.tree[left][n].keys():
                    edge_feature[key] = self.tree[left][n][key]

                self.tree.add_edge(left, right, **edge_feature)

                self.tree.remove_node(n)

    def remove_intermediate_di(self):

        all_nodes = [n for n in self.tree.nodes]
        nodes_to_remove = []
        for n in all_nodes:
            neighbors = list(self.tree.successors(n))

            if len(neighbors) == 1 and n != self.root:
                # left = neighbors[0]
                # right = neighbors[1]
                left = list(self.tree.predecessors(n))[0]
                right = neighbors[0]
                edge_feature = {}
                for key in self.tree[left][n].keys():
                    edge_feature[key] = self.tree[left][n][key]

                self.tree.add_edge(left, right, **edge_feature)

                nodes_to_remove.append(n)

            self.tree.remove_nodes_from(nodes_to_remove)

    def update_order_di(self, mode='level'):
        for n in self.tree.nodes:
            # self.tree.nodes[n][mode] = 1 if self.tree.degree[n] <= 1 and not self.tree.nodes[n]['fixed'] else 0

            self.tree.nodes[n][mode] = 1 if self.tree.degree[n] <= 1 else 0

            if self.tree.nodes[n]['fixed'] and self.tree.nodes[n]['root']:
                self.tree.nodes[n][mode] = -1

        count_no_label = self.node_count
        cur_order = 1
        while count_no_label != 0:
            for node in self.tree.nodes:
                if self.tree.nodes[node][mode] != 0 or len(list(self.tree.neighbors(node))) == 0:
                    continue
                neighbor_orders = np.array([self.tree.nodes[n][mode] for n in self.tree.neighbors(node)])

                root = list(self.tree.predecessors(node))[0]
                root_order = self.tree.nodes[root][mode]

                if np.count_nonzero(neighbor_orders == 0) >= 1:
                    continue

                max_order = np.max(neighbor_orders)
                max_count = np.count_nonzero(neighbor_orders == max_order)

                if len(neighbor_orders) == 1:
                    self.tree.nodes[node][mode] = max_order
                else:
                    self.tree.nodes[node][mode] = max_order if max_count == 1 and mode == 'HS' else max_order + 1

            count_no_label = np.count_nonzero(np.array([self.tree.nodes[n][mode] for n in self.tree.nodes]) == 0)

    def depth_order(self):
        for n in self.tree.nodes:
            self.tree.nodes[n]['depth'] = -1
        for n in self.tree.nodes:
            if self.tree.nodes[n]['depth'] != -1:
                continue
            else:
                self.tree.nodes[n]['depth'] = self.find_depth(n)

    def find_depth(self, n):
        root = list(self.tree.predecessors(n))
        if len(root) == 0:
            return 0
        # elif self.tree.nodes[n]['depth'] != -1:
        #     return self.tree.nodes[n]['depth']
        else:
            root = root[0]
            return 1 + self.find_depth(root)



    def update_final_order_di(self, mode='HS'):
        """
        Update order when no further operations will be applied.
        """

        for n in self.tree.nodes:
            self.tree.nodes[n][mode] = 1 if self.tree.degree[n] == 1 else 0
        count_no_label = self.node_count
        cur_order = 1
        while count_no_label != 0:
            for node in self.tree.nodes:
                if self.tree.nodes[node][mode] != 0 or len(list(self.tree.neighbors(node))) == 0:
                    continue
                neighbor_orders = np.array([self.tree.nodes[n][mode] for n in self.tree.neighbors(node)])

                if np.count_nonzero(neighbor_orders == 0) >= 1:  # has to be second endpoint
                    continue
                max_order = np.max(neighbor_orders)
                max_count = np.count_nonzero(neighbor_orders == max_order)
                self.tree.nodes[node][mode] = max_order if max_count == 1 and mode == 'HS' else max_order + 1
            count_no_label = np.count_nonzero(np.array([self.tree.nodes[n][mode] for n in self.tree.nodes]) == 0)

        node = [n for n in self.tree.nodes if self.tree.nodes[n]['root']]
        if len(node) == 0:
            return
        node = node[0]

        neighbor_orders = np.array([self.tree.nodes[n][mode] for n in self.tree.neighbors(node)])
        max_order = np.max(neighbor_orders)
        max_count = np.count_nonzero(neighbor_orders == max_order)
        self.tree.nodes[node][mode] = max_order if max_count == 1 and mode == 'HS' else max_order + 1

    def label_pressure_drop(self):
        for n in self.tree.edges:
            a, b = n[0], n[1]
            r = self.tree[a][b]['radius']

            l = np.linalg.norm(self.tree.nodes[a]['loc'] - self.tree.nodes[b]['loc']) * self.vsize
            self.tree[a][b]['length'] = l

            # r_mm = r/1e3
            # diameter = r_mm * 2
            # mu = (1 + (6 * np.exp(-0.085 * diameter) + 2.2 - 2.44 * np.exp(-0.06 * (diameter ** 0.645))) *
            #         ((diameter / (diameter - 1.1)) ** 2)) * ((diameter / (diameter - 1.1)) ** 2) / 1000
            #
            # mu = mu * 1e-12

            mu = self.mu

            self.tree[a][b]['resistance'] = 8 * mu * l / (np.pi * r ** 4)  # N s micrometer -5
            self.tree[a][b]['resistance_mm'] = 8 * mu * l * 1e15 / (np.pi * r ** 4)  # N s mm -5

            self.tree[a][b]['pressure_drop'] = self.tree[a][b]['resistance'] * self.tree[a][b]['flow']
            self.tree[a][b]['viscosity'] = mu

    def label_resistance(self):
        for n in self.tree.edges:
            a, b = n[0], n[1]
            r = self.tree[a][b]['radius']

            l = np.linalg.norm(self.tree.nodes[a]['loc'] - self.tree.nodes[b]['loc']) * self.vsize
            self.tree[a][b]['length'] = l

            mu = self.mu

            self.tree[a][b]['resistance'] = 8 * mu * l / (np.pi * r ** 4)  # N s micrometer -5
            self.tree[a][b]['resistance_mm'] = 8 * mu * l * 1e15 / (np.pi * r ** 4)  # N s mm -5

    def label_wss(self):
        for n in self.tree.edges:
            a, b = n[0], n[1]
            r = self.tree[a][b]['radius']

            l = np.linalg.norm(self.tree.nodes[a]['loc'] - self.tree.nodes[b]['loc']) * self.vsize
            self.tree[a][b]['length'] = l

            mu = self.mu

            self.tree[a][b]['wall_shear_stress'] = 4 * mu * self.tree[a][b]['flow'] / (np.pi * r ** 3)
            self.tree[a][b]['wall_shear_stress_Pa'] = self.tree[a][b]['wall_shear_stress'] * 1e12
            self.tree[a][b]['wall_shear_rate'] = 4 * self.tree[a][b]['flow'] / (np.pi * r ** 3)

    def label_pressure_from_root(self, node=None, root_pressure=0.0):

        if node is None:
            node = [n for n in self.tree.nodes if self.tree.nodes[n]['root']][0]
            self.tree.nodes[node]['pressure'] = root_pressure
            self.tree.nodes[node]['pressure_mmhg'] = self.tree.nodes[node]['pressure'] * 1e12 / 133.322

            children = np.array(list(self.tree.neighbors(node)))

        else:
            neighbors = np.array(list(self.tree.neighbors(node)))
            neighbor_orders = np.array([self.tree.nodes[n]['level'] for n in neighbors])
            root = -1 if -1 in neighbor_orders else np.max(neighbor_orders)
            children = np.array([neighbors[i] for i in range(len(neighbors)) if neighbor_orders[i] != root])
            root = neighbors[np.where(neighbor_orders == root)[0][0]]
            self.tree.nodes[node]['pressure'] = \
                self.tree.nodes[root]['pressure'] - self.tree[root][node]['pressure_drop']

            self.tree.nodes[node]['pressure_mmhg'] = self.tree.nodes[node]['pressure'] * 1e12 / 133.322

        for i in children:
            self.label_pressure_from_root(i)

    def label_pressure_from_root_di(self, node=None, root_pressure=0.0):

        if node is None:
            node = [n for n in self.tree.nodes if self.tree.nodes[n]['root']][0]
            self.tree.nodes[node]['pressure'] = root_pressure
            self.tree.nodes[node]['pressure_mmhg'] = self.tree.nodes[node]['pressure'] * 1e12 / 133.322

            children = np.array(list(self.tree.successors(node)))

        else:
            children = np.array(list(self.tree.successors(node)))
            root = list(self.tree.predecessors(node))[0]
            self.tree.nodes[node]['pressure'] = \
                self.tree.nodes[root]['pressure'] - self.tree[root][node]['pressure_drop']

            self.tree.nodes[node]['pressure_mmhg'] = self.tree.nodes[node]['pressure'] * 1e12 / 133.322

        for i in children:
            self.label_pressure_from_root_di(i)


    def save_terminal_only(self, file='sanity_check3.vtk'):

        tree_copy = copy.deepcopy(self.tree)

        all_terminals = [i for i in self.tree.nodes if len(list(self.tree.successors(i))) == 0]
        all_terminal_edges = np.array([(list(self.tree.predecessors(i))[0], i) for i in all_terminals])

        nodes_to_remove = np.setdiff1d(list(self.tree.nodes), np.reshape(all_terminal_edges, -1))

        all_edges = np.array(self.tree.edges)
        remaining_edges = (all_edges[:, None] == all_terminal_edges).all(-1).any(-1)
        edge_to_remove = all_edges[~remaining_edges]

        tree_copy.remove_nodes_from(nodes_to_remove)
        tree_copy.remove_edges_from(edge_to_remove)

        node_list = list(tree_copy.nodes)
        level_idices = np.argsort(node_list)
        level_map = {}
        for idx, i in enumerate(level_idices):
            level_map[node_list[i]] = idx

        tree_copy = nx.relabel_nodes(tree_copy, level_map)

        nodes_inds = tree_copy.nodes
        indices = sorted(nodes_inds)

        # points = [self.tree.nodes[i]['loc'] for i in wrong]

        points = [nodes_inds[i]['loc'] for i in indices]
        points = np.array(points)

        # points = points - np.mean(points, 0)
        # points *= self.vsize

        edges_inds = tree_copy.edges
        edges_inds = np.array(edges_inds)

        radius_list = [tree_copy[n1][n2]['radius'] for n1, n2 in edges_inds]
        radius_list = np.array(radius_list)

        if np.max(edges_inds) >= len(nodes_inds):
            map = dict(zip(indices, np.arange(len(indices))))
            edges_inds_new = np.vectorize(map.get)(edges_inds)
            # radius_list = np.vectorize(map.get)(radius_list)
        else:
            edges_inds_new = edges_inds

        lines = np.hstack([np.ones((len(edges_inds_new), 1), dtype=np.uint8) * 2, edges_inds_new])
        lines = np.hstack(lines)
        mesh = pyvista.PolyData(points,
                                lines=lines
                                )

        for key in tree_copy[edges_inds[0][0]][edges_inds[0][1]]:
            edge_feature = [tree_copy[n1][n2][key] for n1, n2 in edges_inds]
            edge_feature = np.array(edge_feature)

            assert len(edge_feature) > 0 and len(edge_feature) == mesh.n_lines
            mesh.cell_data[key] = edge_feature

        for key in tree_copy.nodes[indices[0]]:
            if key in ('loc', 'fixed', 'root'):
                continue
            node_feature = [tree_copy.nodes[i][key] for i in indices]
            assert len(node_feature) > 0 and len(node_feature) == mesh.n_points
            mesh.point_data[key] = np.array(node_feature)

        mesh.save(file)

    def label_total_resistance(self, node=None):

        if node is None:
            node = [n for n in self.tree.nodes if self.tree.nodes[n]['root']][0]

        children = np.array(list(self.tree.successors(node)))
        if len(children) == 0:
            return 0

        eq = 1 / np.sum([1 / self.tree[node][i]['resistance'] for i in children])


        return eq + np.sum([self.label_total_resistance(i) for i in children])

    def reorder_nodes(self):

        all_nodes = list(self.tree.nodes)

        level_map = {all_nodes[i]: i for i in range(len(all_nodes))}

        self.tree = nx.relabel_nodes(self.tree, level_map)

    def reorder_removed_leaves(self, n):
        """
        Reorder nodes by level.
        """

        level_map = {}
        j = 0
        for i in self.tree.nodes:

            while j < len(n) and i > n[j]:
                j += 1

            level_map[i] = i - j

        assert len(level_map) == len(self.tree.nodes)

        self.tree = nx.relabel_nodes(self.tree, level_map)

    def Kirchoff_law_efferent(self, P_in=100 * 133.322e-6, P_out=15 * 133.322e-6):

        n_nodes = len(self.tree.nodes)

        assert np.max(self.tree.nodes) == n_nodes - 1
        A = np.zeros((n_nodes, n_nodes))
        b = np.zeros(n_nodes)

        R_e = 0.209 * 60 * 133.322 * 1e-3  # Ns mm-5

        for i in range(n_nodes):
            if len(list(self.tree.predecessors(i))) == 0:
                A[i, i] = 1
                b[i] = P_in
            elif len(list(self.tree.successors(i))) == 0:
                A[i, i] = 1
                b[i] = P_out

            else:
                assert len(list(self.tree.predecessors(i))) == 1
                n_in = list(self.tree.predecessors(i))[0]
                # R_in = self.tree[n_in][i]['resistance']         # N s mm -5
                R_in = self.tree[n_in][i]['resistance_mm']
                A[i, n_in] = - 1 / R_in
                coefficient_cur = 1 / R_in

                for n_out in self.tree.successors(i):
                    # R_out = self.tree[i][n_out]['resistance']

                    if len(list(self.tree.successors(n_out))) == 0:
                        R_out = self.tree[i][n_out]['resistance_mm'] + R_e
                    else:
                        R_out = self.tree[i][n_out]['resistance_mm']
                    coefficient_cur += 1 / R_out
                    A[i, n_out] = - 1 / R_out

                A[i, i] = coefficient_cur

        res = np.linalg.solve(A, b)  # micro-meter3 s-1 = 1e-6 nl/s = 6e-5 nl/min
        # ##### mm3 s-1 = 1e-15 nl/s = 6e-14 nl/min

        for i in range(n_nodes):
            self.tree.nodes[i]['pressure_from_Kirchhoff'] = res[i]  #
            self.tree.nodes[i]['pressure_from_Kirchhoff_mmHg'] = res[i] / (133.322e-6)

        all_terminals = [i for i in self.tree.nodes if len(list(self.tree.successors(i))) == 0]
        all_terminal_edges = [(list(self.tree.predecessors(i))[0], i) for i in all_terminals]

        for (i, j) in all_terminal_edges:
            P1 = self.tree.nodes[i]['pressure_from_Kirchhoff']
            P2 = self.tree.nodes[j]['pressure_from_Kirchhoff']
            assert np.isclose(P2, P_out)
            R1 = self.tree[i][j]['resistance_mm']
            R2 = R_e

            AA_pressure = (P1 / R1 + P2 / R2) / (1 / R1 + 1 / R2)

            self.tree.nodes[j]['pressure_from_Kirchhoff'] = AA_pressure
            self.tree.nodes[j]['pressure_from_Kirchhoff_mmHg'] = AA_pressure / (133.322e-6)

        for (i, j) in self.tree.edges:

            self.tree[i][j]['flow_from_Kirchhoff'] = (self.tree.nodes[i]['pressure_from_Kirchhoff'] -
                                                      self.tree.nodes[j]['pressure_from_Kirchhoff']) / self.tree[i][j][
                                                         'resistance_mm']
            # N mm-2/(N s mm -5) = mm3/s
            self.tree[i][j]['flow_from_Kirchhoff_nl/s'] = self.tree[i][j]['flow_from_Kirchhoff'] * 1e3

        # self.save('kirchhoff_efferent.vtk')

    def Kirchoff_law_efferent_flow(self, F_in=7 * 1e3 / 60, P_out=10 * 133.322e-6):

        n_nodes = len(self.tree.nodes)

        assert np.max(self.tree.nodes) == n_nodes - 1
        root = [i for i in self.tree.nodes if len(list(self.tree.predecessors(i))) == 0]
        assert len(root) == 1
        root = root[0]
        root_edge = [root, list(self.tree.successors(root))[0]]

        A = np.zeros((n_nodes, n_nodes))
        b = np.zeros(n_nodes)

        R_e = 0.209 * 60 * 133.322 * 1e-3  # Ns mm-5

        for i in range(n_nodes):
            if i == root:
                A[i, i] = 1
                A[i, root_edge[1]] = -1
                b[i] = self.tree[root][root_edge[1]]['resistance_mm'] * F_in

            elif len(list(self.tree.successors(i))) == 0:
                A[i, i] = 1
                b[i] = P_out

            else:
                assert len(list(self.tree.predecessors(i))) == 1
                n_in = list(self.tree.predecessors(i))[0]
                # R_in = self.tree[n_in][i]['resistance']         # N s mm -5
                R_in = self.tree[n_in][i]['resistance_mm']
                A[i, n_in] = - 1 / R_in
                coefficient_cur = 1 / R_in

                for n_out in self.tree.successors(i):
                    # R_out = self.tree[i][n_out]['resistance']

                    if len(list(self.tree.successors(n_out))) == 0:
                        R_out = self.tree[i][n_out]['resistance_mm'] + R_e
                    else:
                        R_out = self.tree[i][n_out]['resistance_mm']
                    coefficient_cur += 1 / R_out
                    A[i, n_out] = - 1 / R_out

                A[i, i] = coefficient_cur

        res = np.linalg.solve(A, b)  # micro-meter3 s-1 = 1e-6 nl/s = 6e-5 nl/min
        # ##### mm3 s-1 = 1e-15 nl/s = 6e-14 nl/min
        #
        # b_new = A.dot(res)
        # print(np.average(np.isclose(b, b_new)))

        for i in range(n_nodes):
            self.tree.nodes[i]['pressure_from_Kirchhoff'] = res[i]  #
            # self.tree.nodes[i]['pressure_from_Kirchhoff_mmHg'] = res[i]/(133.322e-12)
            self.tree.nodes[i]['pressure_from_Kirchhoff_mmHg'] = res[i] / (133.322e-6)

        all_terminals = [i for i in self.tree.nodes if len(list(self.tree.successors(i))) == 0]
        all_terminal_edges = [(list(self.tree.predecessors(i))[0], i) for i in all_terminals]

        for (i, j) in all_terminal_edges:
            P1 = self.tree.nodes[i]['pressure_from_Kirchhoff']
            P2 = self.tree.nodes[j]['pressure_from_Kirchhoff']
            assert np.isclose(P2, P_out)
            R1 = self.tree[i][j]['resistance_mm']
            R2 = R_e

            AA_pressure = (P1 / R1 + P2 / R2) / (1 / R1 + 1 / R2)

            self.tree.nodes[j]['pressure_from_Kirchhoff'] = AA_pressure
            self.tree.nodes[j]['pressure_from_Kirchhoff_mmHg'] = AA_pressure / (133.322e-6)

        for (i, j) in self.tree.edges:

            self.tree[i][j]['flow_from_Kirchhoff'] = (self.tree.nodes[i]['pressure_from_Kirchhoff'] -
                                                      self.tree.nodes[j]['pressure_from_Kirchhoff']) / self.tree[i][j][
                                                         'resistance_mm']
            # N mm-2/(N s mm -5) = mm3/s
            self.tree[i][j]['flow_from_Kirchhoff_nl/s'] = self.tree[i][j]['flow_from_Kirchhoff'] * 1e3



    def Kirchoff_law_PC(self, P_in=100 * 133.322e-6, P_out=0, Q0=None, ratio=None):
        n_nodes = len(self.tree.nodes)

        assert np.max(self.tree.nodes) == n_nodes - 1

        all_terminals = [i for i in self.tree.nodes if len(list(self.tree.successors(i))) == 0]
        all_terminal_edges = [(list(self.tree.predecessors(i))[0], i) for i in all_terminals]

        # for (parent, i) in all_terminal_edges:
        #     self.tree[parent][i]['radius'] = 8
        # self.label_resistance()

        A = np.zeros((n_nodes, n_nodes))
        b = np.zeros(n_nodes)

        R_e = 0.209 * 60 * 133.322 * 1e-3  # Ns mm-5 # around 1.67
        R_PC = 0.0702 * 60 * 133.322 * 1e-3  # Ns mm-5 around 0.56
        # ratio around 0.7
        for i in range(n_nodes):
            if len(list(self.tree.predecessors(i))) == 0:
                A[i, i] = 1
                b[i] = P_in
            elif len(list(self.tree.successors(i))) == 0:
                A[i, i] = 1
                b[i] = P_out

            else:
                assert len(list(self.tree.predecessors(i))) == 1
                n_in = list(self.tree.predecessors(i))[0]
                # R_in = self.tree[n_in][i]['resistance']         # N s mm -5
                R_in = self.tree[n_in][i]['resistance_mm']
                A[i, n_in] = - 1 / R_in
                coefficient_cur = 1 / R_in

                for n_out in self.tree.successors(i):
                    # R_out = self.tree[i][n_out]['resistance']

                    if len(list(self.tree.successors(n_out))) == 0:
                        R_out = self.tree[i][n_out]['resistance_mm'] + R_e + R_PC
                    else:
                        R_out = self.tree[i][n_out]['resistance_mm']
                    coefficient_cur += 1 / R_out
                    A[i, n_out] = - 1 / R_out

                A[i, i] = coefficient_cur



        res = np.linalg.solve(A, b)  # micro-meter3 s-1 = 1e-6 nl/s = 6e-5 nl/min

        for i in range(n_nodes):
            self.tree.nodes[i]['pressure_from_Kirchhoff'] = res[i]  #
            # self.tree.nodes[i]['pressure_from_Kirchhoff_mmHg'] = res[i]/(133.322e-12)
            self.tree.nodes[i]['pressure_from_Kirchhoff_mmHg'] = res[i] / (133.322e-6)

        all_terminal_flow = np.array([(self.tree.nodes[i]['pressure_from_Kirchhoff'] -
                                       self.tree.nodes[j]['pressure_from_Kirchhoff']) /
                                      (self.tree[i][j]['resistance_mm'] + R_e + R_PC) for (i, j) in all_terminal_edges])
        if Q0 is None and ratio is None:
            Q0 = all_terminal_flow / 6
        else:
            if Q0 is not None:
                Q0 = Q0 / 60 * 1e-3  # nl/min to mm3/s
            else:
                Q0 = all_terminal_flow * ratio

        for q0, qt, (i, j) in zip(Q0, all_terminal_flow, all_terminal_edges):
            Q_aa = qt + q0 * R_e / (R_e + self.tree[i][j]['resistance_mm'] + R_PC)
            P1 = self.tree.nodes[i]['pressure_from_Kirchhoff']
            P2 = self.tree.nodes[j]['pressure_from_Kirchhoff']
            assert np.isclose(P2, 0)
            AA_pressure = P1 - self.tree[i][j]['resistance_mm'] * Q_aa

            self.tree.nodes[j]['pressure_from_Kirchhoff'] = AA_pressure
            self.tree.nodes[j]['pressure_from_Kirchhoff_mmHg'] = AA_pressure / (133.322e-6)
            self.tree[i][j]['flow_from_Kirchhoff'] = Q_aa
            # N mm-2/(N s mm -5) = mm3/s
            self.tree[i][j]['flow_from_Kirchhoff_nl/s'] = Q_aa * 1e3
            # mm3 / s

        all_terminal_flow = [self.tree[i][j]['flow_from_Kirchhoff'] for (i, j) in all_terminal_edges]

        self.rescale_flow_Kirchhoff()
        root = [i for i in self.tree.nodes if len(list(self.tree.predecessors(i))) == 0]
        assert len(root) == 1
        root = root[0]
        root_edge = [root, list(self.tree.successors(root))[0]]
        in_flow = self.tree[root_edge[0]][root_edge[1]]['flow_from_Kirchhoff']  # mm3/s
        in_flow = in_flow / 1e3 * 60  # ml/min
        # print(f'{in_flow = } after Kirchhoff')

        # self.save('kirchhoff_pc.vtk')

    def Kirchoff_law_PC_flow(self, F_in=7 * 1e3 / 60, P_out=0, Q0=None, ratio=None):

        # ml/min to mm3/s

        n_nodes = len(self.tree.nodes)

        assert np.max(self.tree.nodes) == n_nodes - 1
        root = [i for i in self.tree.nodes if len(list(self.tree.predecessors(i))) == 0]
        assert len(root) == 1
        root = root[0]
        root_edge = [root, list(self.tree.successors(root))[0]]

        all_terminals = [i for i in self.tree.nodes if len(list(self.tree.successors(i))) == 0]
        all_terminal_edges = [(list(self.tree.predecessors(i))[0], i) for i in all_terminals]

        A = np.zeros((n_nodes, n_nodes))
        b = np.zeros(n_nodes)

        R_e = 0.209 * 60 * 133.322 * 1e-3  # Ns mm-5 # around 1.67
        R_PC = 0.0702 * 60 * 133.322 * 1e-3  # Ns mm-5 around 0.56
        # ratio around 0.7
        for i in range(n_nodes):
            if i == root:
                A[i, i] = 1
                A[i, root_edge[1]] = -1
                b[i] = self.tree[root][root_edge[1]]['resistance_mm'] * F_in

            elif len(list(self.tree.successors(i))) == 0:
                A[i, i] = 1
                b[i] = 0

            else:
                assert len(list(self.tree.predecessors(i))) == 1
                n_in = list(self.tree.predecessors(i))[0]
                # R_in = self.tree[n_in][i]['resistance']         # N s mm -5
                R_in = self.tree[n_in][i]['resistance_mm']
                A[i, n_in] = - 1 / R_in
                coefficient_cur = 1 / R_in

                for n_out in self.tree.successors(i):
                    # R_out = self.tree[i][n_out]['resistance']

                    if len(list(self.tree.successors(n_out))) == 0:
                        R_out = self.tree[i][n_out]['resistance_mm'] + R_e + R_PC
                    else:
                        R_out = self.tree[i][n_out]['resistance_mm']
                    coefficient_cur += 1 / R_out
                    A[i, n_out] = - 1 / R_out

                A[i, i] = coefficient_cur

        res = np.linalg.solve(A, b)  # micro-meter3 s-1 = 1e-6 nl/s = 6e-5 nl/min

        for i in range(n_nodes):
            self.tree.nodes[i]['pressure_from_Kirchhoff'] = res[i]  #
            # self.tree.nodes[i]['pressure_from_Kirchhoff_mmHg'] = res[i]/(133.322e-12)
            self.tree.nodes[i]['pressure_from_Kirchhoff_mmHg'] = res[i] / (133.322e-6)

        all_terminal_flow = np.array([(self.tree.nodes[i]['pressure_from_Kirchhoff'] -
                                       self.tree.nodes[j]['pressure_from_Kirchhoff']) /
                                      (self.tree[i][j]['resistance_mm'] + R_e + R_PC) for (i, j) in all_terminal_edges])
        if Q0 is None and ratio is None:
            Q0 = all_terminal_flow / 6
        else:
            if Q0 is not None:
                Q0 = Q0 / 60 * 1e-3  # nl/min to mm3/s
            else:
                Q0 = all_terminal_flow * ratio

        for q0, qt, (i, j) in zip(Q0, all_terminal_flow, all_terminal_edges):
            Q_aa = qt + q0 * R_e / (R_e + self.tree[i][j]['resistance_mm'] + R_PC)
            P1 = self.tree.nodes[i]['pressure_from_Kirchhoff']
            P2 = self.tree.nodes[j]['pressure_from_Kirchhoff']
            assert np.isclose(P2, 0)
            AA_pressure = P1 - self.tree[i][j]['resistance_mm'] * Q_aa

            self.tree.nodes[j]['pressure_from_Kirchhoff'] = AA_pressure
            self.tree.nodes[j]['pressure_from_Kirchhoff_mmHg'] = AA_pressure / (133.322e-6)
            self.tree[i][j]['flow_from_Kirchhoff'] = Q_aa
            # N mm-2/(N s mm -5) = mm3/s
            self.tree[i][j]['flow_from_Kirchhoff_nl/s'] = Q_aa * 1e3

        self.rescale_flow_Kirchhoff()

        in_flow = self.tree[root_edge[0]][root_edge[1]]['flow_from_Kirchhoff']  # mm3/s
        in_flow = in_flow / 1e3 * 60  # ml/min
        # print(f'{in_flow = } after Kirchhoff')

        # self.save('kirchhoff_pc.vtk')

    def rescale_flow_Kirchhoff(self, mode='level'):

        for n in self.tree.nodes:
            self.tree.nodes[n][mode] = 1 if self.tree.degree[n] <= 1 else 0

            if self.tree.nodes[n]['root']:
                assert len(list(self.tree.predecessors(n))) == 0
                self.tree.nodes[n][mode] = -1

        count_no_label = len(self.tree.nodes)

        while count_no_label != 0:
            for node in self.tree.nodes:
                if self.tree.nodes[node][mode] != 0 or len(list(self.tree.neighbors(node))) == 0:
                    continue

                neighbor_orders = np.array([self.tree.nodes[n][mode] for n in self.tree.neighbors(node)])

                root = list(self.tree.predecessors(node))[0]

                if np.count_nonzero(neighbor_orders == 0) >= 1:
                    continue

                max_order = np.max(neighbor_orders)
                self.tree.nodes[node][mode] = max_order + 1

                self.tree[root][node]['flow_from_Kirchhoff'] = np.sum(
                    [self.tree[node][n]['flow_from_Kirchhoff'] for n in self.tree.neighbors(node)])

                self.tree[root][node]['flow_from_Kirchhoff_nl/s'] = np.sum(
                    [self.tree[node][n]['flow_from_Kirchhoff_nl/s'] for n in self.tree.neighbors(node)])

            count_no_label = np.count_nonzero(np.array([self.tree.nodes[n][mode] for n in self.tree.nodes]) == 0)

    def Kirchoff_law(self, P_in=100 * 133.322e-6):

        n_nodes = len(self.tree.nodes)

        # P_in = 100 * 133.322e-12  # N/um2
        # P_out = 55 * 133.322e-12

        # P_in = 100 * 133.322e-6  # N/mm2
        # P_out = 55 * 133.322e-6

        assert np.max(self.tree.nodes) == n_nodes - 1
        A = np.zeros((n_nodes, n_nodes))
        b = np.zeros(n_nodes)

        for i in range(n_nodes):
            if len(list(self.tree.predecessors(i))) == 0:
                A[i, i] = 1
                b[i] = P_in
            elif len(list(self.tree.successors(i))) == 0:
                A[i, i] = 1
                # b[i] = P_out #self.tree.nodes[i]['pressure']

                b[i] = self.tree.nodes[i]['pressure'] * 1e6

            else:
                assert len(list(self.tree.predecessors(i))) == 1
                n_in = list(self.tree.predecessors(i))[0]
                # R_in = self.tree[n_in][i]['resistance']         # N s mm -5
                R_in = self.tree[n_in][i]['resistance_mm']
                A[i, n_in] = - 1 / R_in
                coefficient_cur = 1 / R_in

                for n_out in self.tree.successors(i):
                    # R_out = self.tree[i][n_out]['resistance']
                    R_out = self.tree[i][n_out]['resistance_mm']
                    coefficient_cur += 1 / R_out
                    A[i, n_out] = - 1 / R_out

                A[i, i] = coefficient_cur

        res = np.linalg.solve(A, b)  # micro-meter3 s-1 = 1e-6 nl/s = 6e-5 nl/min
        ##### mm3 s-1 = 1e-15 nl/s = 6e-14 nl/min

        b_new = A.dot(res)
        print(np.average(np.isclose(b, b_new)))

        # print(f'is close{np.all(np.isclose(res, res2))}')

        for i in range(n_nodes):
            self.tree.nodes[i]['pressure_from_Kirchhoff'] = res[i]  #
            # self.tree.nodes[i]['pressure_from_Kirchhoff_mmHg'] = res[i]/(133.322e-12)
            self.tree.nodes[i]['pressure_from_Kirchhoff_mmHg'] = res[i] / (133.322e-6)

        for (i, j) in self.tree.edges:
            # self.tree[i][j]['flow_from_Kirchhoff'] = (self.tree.nodes[i]['pressure_from_Kirchhoff'] -
            #                                           self.tree.nodes[j]['pressure_from_Kirchhoff'])/self.tree[i][j]['resistance']

            self.tree[i][j]['flow_from_Kirchhoff'] = (self.tree.nodes[i]['pressure_from_Kirchhoff'] -
                                                      self.tree.nodes[j]['pressure_from_Kirchhoff']) / self.tree[i][j][
                                                         'resistance_mm']
            # N mm-2/(N s mm -5) = mm3/s
            self.tree[i][j]['flow_from_Kirchhoff_nl/s'] = self.tree[i][j]['flow_from_Kirchhoff'] * 1e3

            # self.tree[i][j]['flow_from_Kirchhoff_nl/s'] = self.tree[i][j]['flow_from_Kirchhoff']/1e6

    def re_compute_length(self):
        for (i, j) in self.tree.edges:
            length = np.linalg.norm(self.tree.nodes[i]['loc'] - self.tree.nodes[j]['loc']) * self.vsize
            self.tree[i][j]['length'] = length

    def find_terminal_all_neighbors(self, node):
        """
        Update order when no further operations will be applied.
        """
        parent = list(self.tree.predecessors(node))[0]
        all_children = list(self.tree.successors(parent))
        all_neighbors = [i for i in all_children if i != node]
        return np.array(all_neighbors)

    def back_prop_terminal_flow(self):
        pass

        plt.show()


    def find_dist_to_all_neighbors(self, n):

        # Note that n is index from 0, not index in the tree
        if n % 1000 == 0:
            print(f'{n} nodes done')
        assert self.UnDi_tree is not None and self.all_terminals is not None

        node = self.all_terminals[n]

        all_dist = [0]

        path = nx.single_source_dijkstra_path_length(vt.UnDi_tree, node, cutoff=1e4, weight='length')

        for other_leaf in self.all_terminals[n+1:]:

            # all_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
            # all_length = [self.UnDi_tree[i][j]['length'] for (i, j) in all_edges]
            # cur_dist = np.sum(all_length)
            cur_dist = path.get(other_leaf, 0)
            all_dist.append(cur_dist)

        return all_dist



if __name__ == '__main__':

    root_loc = [588, 217, 650]
    t_flow, mu = 1.167e11 / 3e4, 3.6e-15  # micro-meter3 s-1 # N s micro-meter-2
    vspace = 22.6


    pt_file = os.path.join('data', 'kirchhoff_p_in = 100.vtk')

    save_file = pt_file[:-4] + '_w_pressure.vtk'

    vt = VtkNetworkAnalysis(pt_file, root_loc, vsize=vspace, mu=mu, t_flow=t_flow)
    vt.build()

    vt.to_directed()
    vt.tree = vt.di_tree

    all_terminals = np.array([i for i in vt.tree.nodes if len(list(vt.tree.successors(i))) == 0])

    all_terminal_edges = [(list(vt.tree.predecessors(i))[0], i) for i in all_terminals]

    all_neighbors = vt.find_terminal_all_neighbors(all_terminals[0])

    all_terminal_length = np.array([np.linalg.norm(vt.tree.nodes[i]['loc'] - vt.tree.nodes[j]['loc']) * vspace
                                    for (i, j) in all_terminal_edges])

    plt.hist(all_terminal_length, 30)
    plt.show()

