import pandas as pd
import numpy as np
from typing import Iterable, List, Callable, Tuple, Dict, Union, Set
from bisect import bisect_right, bisect_left
import graph_tool as gt
# from graph_tool.topology import all_circuits
from copy import deepcopy


class DWGraph(gt.Graph):
    """
    This class extends graph_tool.Graph to perform reading/writing from file
    and other useful algorithms that not implemented in graph-tool library.
    """

    EdgesMatchingType = Callable[[Tuple[int, int, float], Tuple[int, int, float]], bool]

    def __init__(self, directed=True):
        super().__init__(directed=directed)
        self.is_weighted_prop = False
        self.is_directed_prop = directed
        self._labels = {}
        self.info_memoization = {}

        self.orig_nodes = None
        self.nodes_orig_new_mapping = None

    def is_weighted(self):
        return self.is_weighted_prop

    def is_directed(self):
        return self.is_directed_prop

    def copy(self) -> 'DWGraph':
        return DWGraph.from_graph(self, self.is_directed(), self.is_weighted())

    @staticmethod
    def read_graph_edges(file_path: str,
                         weighted=True,
                         skip_first_line=True,
                         col_types=(int, int, float)):
        """
        Returns generator reading edges from file line-by-line
        :param file_path: Path to edges list file
        :param weighted: Are edge weights specified?
        :param skip_first_line: Is first line a header?
        :param col_types: Types of identifiers and weights columns.
        :return: Generator
        """

        n_entries = 3 if weighted else 2

        with open(file_path, "r") as f:
            if skip_first_line:
                f.readline()
            while True:
                line = f.readline()
                if line == '':
                    break
                entries = line.split()
                if len(entries) != n_entries:
                    continue
                res = []
                for i, entry in enumerate(entries):
                    res.append(col_types[i](entry))
                yield tuple(res)

    @staticmethod
    def from_graph(gt_graph: gt.Graph, is_directed, is_weighted):
        g = DWGraph()

        v_ind = gt_graph.vertex_index
        e_ind = gt_graph.edge_index
        old_to_new_v = {}
        old_to_new_e = {}

        s = set()

        for e in gt_graph.edges():
            u, v = e
            s.add(v_ind[u])
            s.add(v_ind[v])

        for v in gt_graph.vertices():
            if v_ind[v] in s:
                old_to_new_v[v_ind[v]] = g.add_vertex()

        for e in gt_graph.edges():
            u, v = e
            old_to_new_e[e_ind[e]] = g.add_edge(old_to_new_v[v_ind[u]], old_to_new_v[v_ind[v]])

        for p_type, vp_name in gt_graph.vp.properties:
            if p_type != 'v':
                continue
            old_vp = gt_graph.vp[vp_name]
            g.vp[vp_name] = g.new_vp(old_vp.value_type())
            new_vp = g.vp[vp_name]
            for v in gt_graph.vertices():
                if v_ind[v] in s:
                    new_vp[old_to_new_v[v_ind[v]]] = deepcopy(old_vp[v])

        for p_type, ep_name in gt_graph.ep.properties:
            if p_type != 'e':
                continue
            old_ep = gt_graph.ep[ep_name]
            g.ep[ep_name] = g.new_ep(old_ep.value_type())
            new_ep = g.ep[ep_name]
            for e in gt_graph.edges():
                new_ep[old_to_new_e[e_ind[e]]] = deepcopy(old_ep[e])

        g.is_weighted_prop = is_weighted
        g.is_directed_prop = is_directed

        # g.orig_nodes = list(vertices.keys())
        # g.nodes_orig_new_mapping = vertices

        return g

    @staticmethod
    def from_file_edge_list(file_path, is_weighted=True, skip_first_line=True, make_undirected=False):
        """
        Initialize graph from the list of its (un)weighted edges.
        Each line of input file should has the format
        vertex1 vertex2 (edge_weight)
        """

        file_reader = DWGraph.read_graph_edges(file_path, is_weighted, skip_first_line)

        w_list = []
        for e in file_reader:
            w_list.append(e)
            # if make_undirected:
            #     if is_weighted:
            #         e = (e[1], e[0], e[2])
            #     else:
            #         e = (e[1], e[0])
            #     w_list.append(e)

        g = DWGraph(directed=False)

        weight_property = g.new_ep('float')
        g.ep['w'] = weight_property

        old_label_property = g.new_vp('int')
        g.vp['old'] = old_label_property

        vertices = {}

        for e in w_list:
            for i in (0, 1):
                if e[i] not in vertices:
                    _v = g.add_vertex()
                    vertices[e[i]] = _v
                    old_label_property[_v] = e[i]

            gt_edge = g.add_edge(vertices[e[0]], vertices[e[1]], False)

            if is_weighted:
                weight_property[gt_edge] = e[2]

        g.is_weighted_prop = is_weighted
        g.is_directed_prop = not make_undirected

        g.orig_nodes = list(vertices.keys())
        g.nodes_orig_new_mapping = vertices

        return g

    @staticmethod
    def lcc_r(g: 'DWGraph', g0: 'DWGraph', fr: Dict[int, Set[int]], k_max: int,
              edges_matching: EdgesMatchingType = None) \
            -> Tuple[Dict[int, Set[int]], bool, bool]:
        """
        Performs Local constraint checking algorithm
        :param g: background graph
        :param g0: template graph
        :param fr: specific fr-function in extended Q -> 2^V format
        :param k_max: maximal number of iterations
        :param edges_matching: function for matching edges instead of matching vertices
        :return: new fr-function, exit flag and effectiveness
        """

        # Initialization
        g_labels = g.get_labels()
        g0_labels = g0.get_labels()
        exit_flag = False
        is_effective = False

        # The main loop
        for k in range(k_max):
            d_t = 0

            for e0 in g0.edges():
                q0, q = e0
                matched = False
                for v0 in list(fr[q0]):
                    flag = False
                    v0_neighbors = g.get_out_neighbors(v0)
                    for v in v0_neighbors:
                        if v not in fr[q]:
                            continue
                        if edges_matching is None:
                            flag = True
                        else:
                            g0_e = (g0_labels[q0], g0_labels[q], g0.get_max_weight(q0, q))
                            for gw in g.get_all_weights(v0, v):
                                g_e = (g_labels[v0], g_labels[v], gw)
                                if edges_matching(g_e, g0_e):
                                    flag = True
                                    break
                        if flag:
                            break

                    if not flag:
                        fr[q0].remove(v0)
                        d_t += 1
                        is_effective = True
                        if not fr[q0]:
                            exit_flag = True
                            break
                    else:
                        matched = True

                exit_flag = exit_flag or not matched
                if exit_flag:
                    break

            if d_t == 0 or exit_flag:
                break

        return fr, exit_flag, is_effective

    @staticmethod
    def cc_r(g: 'DWGraph', g0: 'DWGraph', fr: Dict[int, Set[int]], k0: Iterable[List[int]],
             edges_matching: EdgesMatchingType = None) \
            -> Tuple[Dict[int, Set[int]], bool, bool]:
        """
        Performs cycle checking algorithm
        :param g: background graph
        :param g0: template graph
        :param fr: function f from LCC algorithm
        :param k0: cycles templates
        :param edges_matching: function for matching edges instead of matching vertices
        :return: new T (subset of vertices of g)
        """

        exit_flag = False
        is_effective = False

        # iterate through all cycles templates
        it = 0
        for c0 in k0:
            print(str(it) + " cycles proceeded")
            it += 1
            # initialize cycles
            # labels of background graph
            g_labels = g.get_labels()
            # labels of template graph
            g0_labels = g0.get_labels()

            # get the first edge of the template cycle
            if len(c0) == 1:
                continue
            q0, q1 = c0[0:2]

            for v0 in list(fr[q0]):
                a = [v0]
                for s in range(len(c0)):
                    # print(len(c0))
                    # qb is the s-th edge's start
                    qb = c0[s % len(c0)]
                    qb_label = g0_labels[qb]

                    # qe is the s-th edge's end
                    qe = c0[(s + 1) % len(c0)]
                    qe_label = g0_labels[qe]

                    # get weight of (qb, qe) edge
                    w0 = None
                    if edges_matching is not None:
                        w0 = g0.get_max_weight(qb, qe)

                    b = []
                    # print(len(a))
                    for v in a:
                        for v1 in fr[qe]:
                            if not g.edge(v, v1):
                                continue
                            if s == len(c0) - 1 and v0 != v1:
                                continue
                            if edges_matching is None:
                                condition = g_labels[v1] == qe_label
                            else:
                                condition = False
                                g0_e = (qb_label, qe_label, w0)
                                for gw in g.get_all_weights(v, v1):
                                    g_e = (g_labels[v], g_labels[v1], gw)
                                    if edges_matching(g_e, g0_e):
                                        condition = True
                                        break
                            if condition:
                                b.append(v1)
                    a.clear()
                    a = b.copy()

                if v0 not in a:
                    is_effective = True
                    fr[q0].remove(v0)
                    if not fr[q0]:
                        # print(q0)
                        exit_flag = True
                if exit_flag:
                    break

            if exit_flag:
                break

        return fr, exit_flag, is_effective

    @staticmethod
    def vertex_elimination_r(g: 'DWGraph', g0: 'DWGraph', k_max: int = 2,
                             edges_matching: EdgesMatchingType = None) -> Tuple[Union[Dict[int, set], None],
                                                                                Union[Dict[int, set], None]]:
        """
        This method applies LCC and CC methods until the vertices are being removed
        :param g: background graph
        :param g0: template graph
        :param k_max: number of iterations for LCC
        :param edges_matching: function for matching edges instead of matching vertices
        :return: new T (subset of vertices of g)
        """
        bad_result = ({}, {})

        print('Getting cycles')
        k0 = g0.get_cycles()
        print('Cycles got')

        g0_labels = g0.get_labels()
        bg_label_to_vertex = g.get_rev_labels_extended()
        fr = {}
        for q in g0.vertices():
            r = bg_label_to_vertex.get(g0_labels[q], set()).copy()
            fr[q] = set()

            for v in r:
                # if g.degree(v) >= g0.degree(p):
                if DWGraph.vertex_greater(g, v, g0, q):
                    fr[q].add(v)

            if not fr[q]:
                return bad_result

        while True:
            print('LCC started')
            fr, exit_flag, is_lcc_effective = DWGraph.lcc_r(g, g0, fr, k_max, edges_matching)
            # print('LCC done', t, f)
            print('LCC done')
            if exit_flag:
                return bad_result
            print('CC started')
            fr, exit_flag, is_cc_effective = DWGraph.cc_r(g, g0, fr, k0, edges_matching)
            # print('CC done', t)
            print('CC done')
            if exit_flag:
                return bad_result

            # f, d, t, exit_flag, is_refine_effective = DWGraph.refine_candidate_function(f, t, d)
            if exit_flag:
                return bad_result
            if not is_lcc_effective and not is_cc_effective:  # and not is_refine_effective:
                break

        # reorganize output format
        # We'll return dictionary T -> 2^V0
        # We'll also change new vertices to new ones
        g_old = g.get_old_vertices()
        g0_old = g0.get_old_vertices()
        result = {}
        result_new = {}

        for q, v_s in fr.items():
            for v in v_s:
                result_new.setdefault(v, set()).add(q)
                result.setdefault(g_old[v], set()).add(g0_old[q])

        return result, result_new

    def subgraph(self, vertices: List[int]) -> 'DWGraph':
        filter_option = self.new_vertex_property('bool')
        for v in vertices:
            filter_option[v] = True
        sub = gt.GraphView(self, filter_option)
        new_g = DWGraph.from_graph(sub, self.is_directed_prop, self.is_weighted_prop)

        return new_g

    @staticmethod
    def get_vertices_list_recursive(g: 'DWGraph', g0: 'DWGraph', t: Dict[int, set],
                                    output_file: str = None, edges_matching: EdgesMatchingType = None,
                                    comp_approx: int = 10) -> \
            Tuple[List[List[Tuple[int, int]]], List[List[Tuple[int, int]]]]:
        """
        Returns the result of vertex elimination as a list of lists of vertices of subgraphs of background graph
        :param g: Background graph
        :param g0: Template graph
        :param t: Subset of vertices of background multi-mapped into V0
        :param output_file: File for vertices output
        :param edges_matching: Edges matching function for this configuration
        :param comp_approx: Approximate number of vertices in each component
        :return: List of list of tuples. Each tuple is a pair (q, v) where q is an index in G0 and v is an index in G.
                 List is sorted by q (in ascending order)
        """
        from recursive_unique_tuples import recursive_unique_tuples

        with open(output_file, 'w'):
            pass

        # Reverse the mapping. It's not bijective, that's why we map each element of V0 to a set of V
        rev_map = {}
        g_candidates = []

        for k, v_set in t.items():
            if v_set:
                g_candidates.append(k)
            for v in v_set:
                rev_map.setdefault(v, set()).add(k)
        g0_candidates = list(rev_map.keys())

        # Initialize the result with empty list
        res = []
        res_new = []

        # Candidate subgraph
        g_subgraph = g.subgraph(g_candidates)
        # g_subgraph.get_labels = g.get_labels

        # Make new rev_map for subgraph
        rev_map_new = {}
        g_old_subgraph = g_subgraph.get_old_vertices()
        g_old_rev_subgraph = {}
        g_old = g.get_old_vertices()
        for i, o in enumerate(g_old_subgraph):
            g_old_rev_subgraph[o] = i
        for v in rev_map:
            rev_map_new[v] = set([g_old_rev_subgraph[g_old[p]] for p in rev_map[v] if g_old[p] in g_old_rev_subgraph])

        # for tpl in unique_tuples(g_subgraph, g0_candidates, g0, edges_matching, u_tuples_sets):

        g0_old = g0.get_old_vertices()
        # g_old = g.get_old_vertices()

        for tpl in \
                recursive_unique_tuples(g_subgraph, g0, rev_map_new, g0_candidates, edges_matching, comp_approx):
            # tpl : New V0 -> Old V
            res_list_old_to_old = [(g0_old[g0_candidates[i]], g_old_subgraph[v]) for i, v in enumerate(tpl)]
            res_list_new_to_old = [(g0_candidates[i], g_old_subgraph[v]) for i, v in enumerate(tpl)]
            if output_file is not None:
                with open(output_file, 'a') as f:
                    f.write(str(res_list_old_to_old) + '\n')
                    res.append(res_list_old_to_old)
                    res_new.append(res_list_new_to_old)

        return res, res_new

    @staticmethod
    def relabel_nodes(g: 'DWGraph', mapping: Dict[int, int]) -> 'DWGraph':
        """
        Changes old vertex property according to new labeling
        :param g:
        :param mapping:
        :return:
        """
        gc = g.copy()
        g_old = gc.get_old_vertices()
        for v_old, v_new in mapping.items():
            g_old[v_old] = v_new

        return gc

    @staticmethod
    def get_graphs_by_vertices_list(g0: 'DWGraph', vertices: List[List[Tuple[int, int]]]) \
            -> List['DWGraph']:
        """
        Transforms vertices lists to graphs
        :param g0: Template graph
        :param vertices: List which structure is described in get_vertices_list method description
                         New V0 -> Old V mapping expected
        :return: List of subgraphs of G
        """

        res = []
        for v_list in vertices:
            mapping = dict(v_list)
            res.append(DWGraph.relabel_nodes(g0, mapping))
        return res

    def get_max_weight(self, u, v):
        wt = self.get_weights()
        edges = self.edge(u, v, all_edges=True)
        return wt[max(edges, key=lambda e: wt[e])]

    def get_all_weights(self, u, v):
        wt = self.get_weights()
        edges = self.edge(u, v, all_edges=True)
        return [wt[e] for e in edges]

    @staticmethod
    def vertex_greater_by_dict(x_info: Dict[int, int], y_info: Dict[int, int]) -> bool:
        for k, n in y_info.items():
            if k not in x_info:
                return False
            if x_info[k] < y_info[k]:
                return False

        return True

    @staticmethod
    def vertex_greater(g1: 'DWGraph', x: gt.Vertex, g2: 'DWGraph', y: gt.Vertex) -> bool:
        x_info = g1.get_vertex_label_info(x)
        y_info = g2.get_vertex_label_info(y)
        return DWGraph.vertex_greater_by_dict(x_info, y_info)

    def get_vertex_label_info(self, v: gt.Vertex, f: Dict[int, Set[int]] = None, memo: bool=True) -> Dict[int, int]:
        memo = memo and f is None
        if memo and self.info_memoization.get(v, False):
            return self.info_memoization[v]

        v_info = {}
        labels = self.get_labels()
        out_neighbors = self.get_out_neighbors(v).tolist()
        for u in out_neighbors:
            if f is not None:
                u_candidates = f.get(u, False)
                if not u_candidates:
                    continue

            prev_count = v_info.get(labels[u], 0)

            d = self.edge(v, u, all_edges=True)
            v_info[labels[u]] = prev_count + len(d)

        if memo:
            self.info_memoization[v] = v_info

        return v_info

    def write_to_file(self, file_path):
        """
        Write to file the list of graph's [un]weighted edges.
        Each line of output file will have the format
        vertex1 vertex2 [edge_weight]
        """
        edges = self.edges()

        weights = self.get_weights()
        old_vertex = self.get_old_vertices()

        with open(file_path, 'w') as f:
            for e in edges:
                xv, yv = tuple(e)
                if self.is_weighted():
                    w = weights[e]
                    f.write("{} {} {}\n".format(old_vertex[xv], old_vertex[yv], w))
                else:
                    f.write("{} {}\n".format(old_vertex[xv], old_vertex[yv]))

    def get_labels(self) -> gt.PropertyMap:
        """
        Returns new V -> Label mapping
        """
        # return self._labels
        return self.vp['label']

    def get_weights(self) -> Union[gt.PropertyMap, None]:
        """
        Returns graph weights for each edge
        :return:
        """
        if self.is_weighted():
            return self.ep['w']

    def get_old_vertices(self) -> gt.PropertyMap:
        return self.vp['old']

    def get_rev_labels_extended(self):
        """
        Returns Label -> 2^V dictionary
        For each label, result[l] is a set of vertices with label l.
        """
        rev = {}

        for v, label in enumerate(self.get_labels().get_array()):
            rev.setdefault(label, set()).add(v)
        return rev

    def set_labels_as_properties(self) -> None:
        """
        Adds 'label' property to vertices.
        """
        prop = self.new_vp('int')
        self.vp['label'] = prop
        for old, new in self.nodes_orig_new_mapping.items():
            prop[new] = self._labels[old]

    def add_default_labels(self) -> None:
        """
        Adds default label value V to each vertex V.
        """
        self.add_labels_by_identifiers(lambda v: v)

    def add_labels_by_identifiers(self, func: Callable[[int], int]):
        """
        Adds label value func(V) to each vertex V.
        """
        for v in self.orig_nodes:
            self._labels[v] = func(v)
        self.set_labels_as_properties()

    def add_labels_from_file(self, file_path):
        """
        Reads the file each line of which has the format:
        Vertex Label
        Assigns Labels to corresponding vertices
        """
        df = pd.read_csv(filepath_or_buffer=file_path, delim_whitespace=True,
                         names=['v', 'label', 'w'], dtype={'v': np.int32, 'label': np.int32}, header=None)
        vl, lab_i = df['v'], iter(df['label'])
        for v in vl:
            self._labels[v] = next(lab_i)
        self.set_labels_as_properties()

    def has_cycle(self, cycle: List[int]) -> bool:
        """
        Does this graph has such cycle?
        :param cycle: Cycle as a list of vertices
        :return: True if this graph has such cycle
        """
        for i in range(len(cycle)):
            fr = cycle[i]
            to = cycle[(i+1) % len(cycle)]
            if not self.edge(fr, to):
                return False

        return True

    def get_cycles(self) -> Iterable[List[int]]:
        """
        Returns the generator of cycles of this graph used by CC algorithm.
        :return: Generator of lists-cycles.
        """
        # There is no way to get only simple cycles for graph or cycle basis
        # So we use networkx (but only for getting cycles)
        import networkx as nx

        v_ind = self.vertex_index

        simple_edges = [(v_ind[e.source()], v_ind[e.target()]) for e in self.edges()]
        cycles = nx.cycle_basis(nx.Graph(simple_edges))
        '''
        if self.is_directed():
            g = self
        else:
            g = self.copy()
            g.is_directed = lambda: True

        cycles = all_circuits(g, unique=True)
        cl = list(cycles)
        '''
        if self.is_directed():
            for c in cycles:
                if self.has_cycle(c):
                    # print('Cycle!')
                    yield c
        else:
            return cycles

        # return cl

    @staticmethod
    def get_ratios(g: 'DWGraph', g0: 'DWGraph',
                   edge_type_g, e_type_g_labels: bool,
                   edge_type_g0, e_type_g0_labels: bool, w_eps: float) -> \
            Union[Dict[Tuple[int, int], Dict[float, float]], Dict[Tuple[int, int], float]]:
        """
        Returns the matching ratios of edges in g0 to edges of g.
        :param g: Background graph
        :param g0: Template graph
        :param edge_type_g: (u, v) -> type mapping for G
        :param e_type_g_labels: edge_type_g takes pair of labels, not vertices
        :param edge_type_g0: (u, v) -> type mapping for G0
        :param e_type_g0_labels: edge_type_g0 takes pair of labels, not vertices
        :param w_eps: epsilon for matching weights
        :return: Dictionary (u, v) |-> {w_1 -> r_1, w_2 -> r_2, ..., w_s -> r_s} OR
                            (u, v) |-> r [in case of unweighted graph]
        """
        g_labels = g.get_labels()
        g0_labels = g0.get_labels()

        '''
        When we match edges, we base on the following:
        |w - w0| < eps  <=>
        w0 - eps < w < w0 + eps  
        '''

        # e2 = 1/(w_eps*2)
        edge_type_check = True

        if edge_type_g is None or edge_type_g0 is None:
            edge_type_check = False

            # This function is not used, just for conformity
            def e_type_g(_u, _v):
                return 0 * (_u + _v)
            e_type_g0 = e_type_g
        else:
            e_type_g = (lambda _u, _v: edge_type_g(g_labels[_u], g_labels[_v])) \
                if e_type_g_labels else edge_type_g
            e_type_g0 = (lambda _u, _v: edge_type_g0(g0_labels[_u], g0_labels[_v])) \
                if e_type_g0_labels else edge_type_g0

        print("Making stats by graph G...")
        g_stats = {}
        total = 0

        g_weights = g.get_weights()

        for e in g.edges():
            w = g_weights[e]
            u, v = e

            k = (g_labels[u], g_labels[v])

            if edge_type_check:
                c = e_type_g(u, v)
                k += (c, )

            weights_dict = g_stats.setdefault(k, {})

            weights_dict[w] = weights_dict.get(w, 0) + 1
            total += 1

        print("Converting stats to searchable form...")
        g_list_stats = {}
        for k, weights_dict in g_stats.items():
            items_list = list(weights_dict.items())
            if items_list[0][0] is not None:
                items_list.append((-1e10, 0))
                items_list.sort(key=lambda t: t[0])
                w_list = [w for w, _ in items_list]
                prefix_sums = [count for _, count in items_list]
                for i in range(1, len(prefix_sums)):
                    prefix_sums[i] += prefix_sums[i - 1]
                g_list_stats[k] = (w_list, prefix_sums)
            else:
                g_list_stats[k] = items_list[0][1]

        del g_stats

        print("Making resulting dictionary...")
        res = {}

        g0_weights = g0.get_weights()
        g0_old = g0.get_old_vertices()

        for e in g0.edges():
            w = g0_weights[e]
            u, v = e

            k = (g0_labels[u], g0_labels[v])

            if edge_type_check:
                c = e_type_g0(u, v)
                k += (c, )

            w_search = g_list_stats.get(k, None)
            if w_search is None:
                count = 0
            elif isinstance(w_search, int):
                count = w_search
            else:
                w_list, prefix_sums = w_search
                i_from = bisect_right(w_list, w - w_eps)
                i_to = bisect_left(w_list, w + w_eps)
                count = prefix_sums[i_to - 1] - prefix_sums[i_from - 1]

            r = count / total

            old_edge_g0 = (g0_old[u], g0_old[v])

            if w is not None:
                res.setdefault(old_edge_g0, {}).__setitem__(w, r)
            else:
                res[old_edge_g0] = r

        return res


class NonUniqueLabelsError(Exception):
    """
    This exception is thrown if you try to invert labels
    of non-unique-labeled graph.
    """

    def __init__(self):
        pass
