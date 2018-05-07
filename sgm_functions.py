import numpy as np
import time
import collections as coll
import graph_tool as gt
import graph_tool.topology as gt_top
import graph_tool.generation as gt_gen
import graph_tool.search as gt_s
import graph_tool.draw.gtk_draw as gt_draw
from copy import deepcopy
from typing import Iterable, Union


def construct_gt_graph(vertices, edges, g_from: gt.Graph):
    g = create_q_graph(g_from, vertices, edges, False)
    return g


def calc_random_spanning_tree(q: gt.Graph):
    curr_node = np.random.choice(list(q.vertices()))
    # root_node = curr_node
    nodes_visited = {curr_node}
    edges_used = set()
    n_nodes = q.num_vertices()

    while len(nodes_visited) < n_nodes:
        # Choose a random neighbour
        next_node = np.random.choice(q.get_out_neighbors(curr_node))
        if next_node not in nodes_visited:
            edges_used.add((curr_node, next_node))
            nodes_visited.add(next_node)
        curr_node = next_node

    t = construct_gt_graph(nodes_visited, edges_used, q)
    return t


def calculate_mdst_v2(g: gt.Graph, n_idx, e_idx, used_stuff=set()):
    # Step 1: Figure out the weights.
    n_att_name = list(n_idx.keys())[0]
    e_att_name = list(e_idx.keys())[0]

    # Create an MDSTWeight vector on the nodes and edges.
    v_weight = vp_map(g, 'MDST_v_weight', 'float')
    e_weight = ep_map(g, 'MDST_e_weight', 'float')

    v_attribute_list = list(n_idx[n_att_name].keys())
    e_attribute_list = list(e_idx[e_att_name].keys())

    v_a_map = vp_map(g, n_att_name)
    e_a_map = ep_map(g, e_att_name)

    for n in g.vertices():
        if v_a_map[n] in v_attribute_list:
            v_weight[n] = len(n_idx[n_att_name][v_a_map[n]]) / n_idx[n_att_name]['size']
        else:
            v_weight[n] = 0

    for e in g.edges():
        if e in used_stuff:
            e_weight[e] = 1
        else:
            if e_a_map[e] in e_attribute_list:
                e_weight[e] = len(e_idx[e_att_name][e_a_map[e]]) / e_idx[e_att_name]['size']
            else:
                e_weight[e] = 0

    #    for e1,e2 in G.edges():
    #        G.adj[e1][e2]['Nonsense'] = 5

    # Step 2: Calculate the MST.
    # gt.draw.graph_draw(g, vertex_text=g.vp['old'], vertex_font_size=18, output_size=(300, 300), output='G.png')
    t_map = gt_top.min_spanning_tree(g, e_weight, g.vertex(0))

    # T = nx.algorithms.minimum_spanning_tree(G,weight='Nonsense')
    # Step 3: Figure out which root results in us doing the least work.

    t = gt.GraphView(g, efilt=t_map, directed=False)
    # gt.draw.graph_draw(t, vertex_text=t.vp['old'], vertex_font_size=18, output_size=(300, 300), output='T.png')

    best_t = None
    best_score = np.inf
    for root in t.vertices():
        # Generate a new tree

        it = gt_s.bfs_iterator(t, root)
        nodes = []
        edges = []
        for e in it:
            edges.append(e)
            nodes.extend([e.source(), e.target()])
        nodes = np.unique(nodes)
        new_t = create_q_graph(t, q_nodes=nodes, q_edges=edges, directed=True)

        new_t_score = mdst_score_v2(t, root)
        if new_t_score < best_score:
            # print(best_score)
            best_t = new_t
            best_score = new_t_score

    return best_t, best_score


def copy_node_attributes(g_to: gt.Graph, node_to: gt.Vertex, g_from: gt.Graph, node_from: gt.Vertex):
    for p_type, vp_name in g_from.vp.properties:
        if p_type != 'v':
            continue
        old_vp = g_from.vp[vp_name]
        if vp_name not in g_to.vp:
            g_to.vp[vp_name] = g_to.new_vp(old_vp.value_type())
        new_vp = g_to.vp[vp_name]
        new_vp[node_to] = deepcopy(old_vp[node_from])


def copy_edge_attributes(g_to: gt.Graph, edge_to: gt.Edge, g_from: gt.Graph, edge_from: gt.Edge):
    for p_type, ep_name in g_from.ep.properties:
        if p_type != 'e':
            continue
        old_ep = g_from.ep[ep_name]
        if ep_name not in g_to.ep:
            g_to.ep[ep_name] = g_to.new_ep(old_ep.value_type())
        new_ep = g_to.ep[ep_name]
        new_ep[edge_to] = deepcopy(old_ep[edge_from])


def mdst_score_v2(t: gt.Graph, root, weight='MDST_v_weight', weight_e='MDST_e_weight'):  # Use Old formula for now
    percent_left = vp_map(t, 'PercentLeft', 'float')
    weight_p = vp_map(t, weight)
    weight_ep = ep_map(t, weight_e)

    percent_left[root] = weight_p[root]
    total_score = percent_left[root]
    for e in gt_s.bfs_iterator(t, root):
        start_v, end_v = e
        sc = percent_left[end_v] = weight_p[end_v] * weight_ep[e] * percent_left[start_v]
        total_score += sc

    return total_score


def mdst_score(t: gt.Graph, n_idx, e_idx, used_stuff):
    n_att_name = next(n_idx.keys())
    e_att_name = next(e_idx.keys())

    percent_left = vp_map(t, 'PercentLeft', 'float')
    v_attribute_p = vp_map(t, n_att_name, 'int')
    e_attribute_p = ep_map(t, e_att_name, 'int')

    # Choose a vertex with input degree = 0
    root = np.where(t.get_in_degrees(t.get_vertices()) == 0)[0]

    if root in used_stuff:
        percent_left[root] = 1
    else:
        percent_left[root] = len(n_idx[n_att_name][v_attribute_p[root]]) / n_idx[n_att_name]['size']

    total_score = percent_left[root]
    for e in gt_s.bfs_iterator(t, root):
        start_node, end_node = e
        if e in used_stuff:
            edge_score = 1
            end_node_score = 1
        else:
            edge_score = len(e_idx[e_att_name][e_attribute_p[e]]) / e_idx[e_att_name]['size']
            if end_node in used_stuff:
                end_node_score = 1
            else:
                end_node_score = len(n_idx[n_att_name][v_attribute_p[end_node]]) / n_idx[n_att_name]['size']
        percent_left[end_node] = end_node_score * percent_left[start_node] * edge_score
        total_score += percent_left[end_node]

    return total_score


def insert_clique_subgraph(a: gt.Graph, n_nodes, att_dist, att_name, target_solutions):
    q_nodes = np.random.choice(a.get_vertices(), n_nodes, replace=False)
    a_to_q = {}
    q_to_a = {}
    nodes_list = []
    edges_list = []

    for i, n in enumerate(q_nodes):
        a_to_q[n] = i  # Maps from archive node to T.
        q_to_a[i] = n  # Maps from T to archive node
        nodes_list.append(n)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            n1 = q_to_a[i]
            n2 = q_to_a[j]
            # e = a.edge(n1, n2)
            # if e:
            #     edges_list.append(e)
            #     continue
            e = a.add_edge(n1, n2)
            edges_list.append(e)
            add_single_edge_attribute(a, e, att_dist, att_name)

    new_target = {'nodes': nodes_list, 'edges': edges_list, 'isClutter': False}
    target_solutions.append(new_target)
    return create_q_graph(a, q_nodes)


# def getRandomSubgraphOld(A, nNodes):
#    for i in range(1000):
#        qNodes = np.random.choice(A.nodes(), nNodes, replace=False)
#        AtoQ = dict()
#        QtoA = dict()
#        for i, n in enumerate(qNodes):
#            AtoQ[n] = i  # Maps from archive node to T.
#            QtoA[i] = n  # Maps from T to archive node
#
#        Q = nx.Graph()
#        Q.add_nodes_from(QtoA.keys());
#
#        for eStart, eEnd in A.edges():
#            if eStart in qNodes and eEnd in qNodes:
#                Q.add_edge(AtoQ[eStart], AtoQ[eEnd])
#
#        if len([g for g in nx.connected_components(Q.to_undirected())]) > 1:
#            continue
#
#        for n in Q.nodes():
#            Q.node[n] = A.node[QtoA[n]]
#            Q.node[n]['realNode'] = QtoA[n]
#        for n1, n2 in Q.edges():
#            Q.adj[n1][n2] = A.adj[QtoA[n1]][QtoA[n2]]
#            Q.adj[n1][n2]['realEdge'] = (QtoA[n1], QtoA[n2])
#        return Q
#    return None


def get_random_subgraph(a: gt.Graph, n_nodes):
    for i in range(1000):
        q_nodes = np.random.choice(a.num_vertices(), n_nodes, replace=False)
        q = create_q_graph(a, q_nodes)
        if not accept_q_graph(q):
            continue
        return q
    return None


def accept_q_graph(q: gt.Graph):
    _, r = gt_top.label_components(q, directed=False)
    return r.shape[0] <= 1


def create_q_graph(a_graph: gt.Graph,
                   q_nodes: Union[None, Iterable[gt.Vertex]]=None,
                   q_edges: Union[None, Iterable[gt.Edge]]=None,
                   add_back_reference=True,
                   directed: Union[bool, None]=None) -> gt.Graph:
    _directed = directed if directed is not None else a_graph.is_directed()
    q = gt.Graph(directed=_directed)
    a_q_v = {}
    a_q_e = {}
    q_a_v = {}
    q_a_e = {}

    if q_nodes is None:
        q_nodes = set(a_graph.vertices())

    for v in q_nodes:
        nv = a_q_v[v] = q.add_vertex()
        q_a_v[nv] = v

    for p_type, vp_name in a_graph.vp.properties:
        if p_type != 'v':
            continue
        old_vp = a_graph.vp[vp_name]
        q.vp[vp_name] = q.new_vp(old_vp.value_type())
        new_vp = q.vp[vp_name]
        for v in a_graph.vertices():
            if a_graph.vertex_index[v] in q_nodes:
                new_vp[a_q_v[v]] = deepcopy(old_vp[v])

    if q_edges is None:
        q_edges = set(a_graph.edges())

    for e in q_edges:
        e_start, e_end = e
        if e_start in q_nodes and e_end in q_nodes:
            ne = q.add_edge(a_q_v[e_start], a_q_v[e_end])
            a_q_e[e] = ne
            q_a_e[ne] = e

    for p_type, ep_name in a_graph.ep.properties:
        if p_type != 'e':
            continue
        old_ep = a_graph.ep[ep_name]
        q.ep[ep_name] = q.new_ep(old_ep.value_type())
        new_ep = q.ep[ep_name]
        for e, ne in a_q_e.items():
            new_ep[ne] = deepcopy(old_ep[e])

    if add_back_reference:
        from_a_node = q.new_vp('int')
        q.vp['fromANode'] = from_a_node
        from_a_edge = q.new_ep('object')
        q.ep['fromAEdge'] = from_a_edge

        for v in q.vertices():
            from_a_node[v] = a_graph.vertex_index[q_a_v[v]]

        for e in q.edges():
            e_a = q_a_e[e]
            vs, ve = e_a
            from_a_edge[e] = (a_graph.vertex_index[vs], a_graph.vertex_index[ve])

    return q


def insert_targets(a: gt.Graph, q: gt.Graph, n_targets, target_solutions, is_clutter):
    print("Inserting targets, n_targets = {}.".format(n_targets))

    mp_q = vp_map(q, 'nValue', 'int')
    mp_a = vp_map(a, 'nValue', 'int')

    for i in range(n_targets):
        print("Target {}...".format(i))
        # Map Q to a corresponding set of random nodes in A
        map_to_a = {}
        new_target = {'nodes': [], 'edges': [], 'isClutter': is_clutter}
        for node in q.vertices():
            while True:
                ind = np.random.choice(list(range(a.num_vertices())))
                if not is_node_in_existing_target(ind, target_solutions):
                    break
            map_to_a[node] = ind
            new_target['nodes'].append(ind)
            # copy node attributes exactly
            mp_a[ind] = mp_q[node]

        # Add the same edges in Q to A
        for edge in q.edges():
            source_a = map_to_a[edge.source()]
            destination_a = map_to_a[edge.target()]

            # edgeData = Q.get_edge_data(edge[0], edge[1]).copy()

            e_a = a.add_edge(source_a, destination_a)
            # print('bf: {}, {}'.format(a.ep['eValue'][e_a], q.ep['eValue'][edge]))
            copy_edge_attributes(a, e_a, q, edge)
            # print('af: {}'.format(a.ep['eValue'][e_a]))
            # for edge_attr, attr_val in edgeData.items():
            #     nx.set_edge_attributes(Q, {(source_a, destination_a): attr_val}, edge_attr)

            new_target['edges'].append(e_a)
        target_solutions.append(new_target)
    print(a)


def is_node_in_existing_target(ind, target_solutions):
    for target in target_solutions:
        if ind in target['nodes']:
            return True
    return False


def gen_random_graph(n, p):
    def f():
        return np.random.poisson(p * (n - 1))
    g = gt_gen.random_graph(n, f, directed=False, model='erdos')
    return g


def add_original_enumeration(g: gt.Graph) -> gt.Graph:
    g.vp['original'] = g.vertex_index.copy()
    return g


def vp_map(g: gt.Graph, v_property: str, p_type: str = 'int') -> gt.PropertyMap:
    if v_property not in g.vp:
        g.vp[v_property] = g.new_vp(p_type)
    return g.vp[v_property]


def ep_map(g: gt.Graph, e_property: str, p_type: str = 'int') -> gt.PropertyMap:
    if e_property not in g.ep:
        g.ep[e_property] = g.new_ep(p_type)
    return g.ep[e_property]


def add_node_attributes(g: gt.Graph, att_dist, att_name, p_type: str = 'int'):
    prop = vp_map(g, att_name, p_type)

    for n in g.vertices():
        prop[n] = np.random.choice(list(range(len(att_dist))), p=att_dist)
    return g


def add_edge_attributes(g: gt.Graph, att_dist, att_name, p_type: str = 'int'):
    for e in g.edges():
        add_single_edge_attribute(g, e, att_dist, att_name, p_type)
    return g


def add_single_edge_attribute(g: gt.Graph, e, att_dist, att_name, p_type: str = 'int'):
    val = np.random.choice(list(range(len(att_dist))), p=att_dist)
    p = ep_map(g, att_name, p_type)
    p[e] = val


def create_dict(g: gt.Graph, n_att_name, e_att_name):
    node_dict = {n_att_name: {'size': g.num_vertices()}}

    # find all the total values
    unique_keys = np.unique([g.vp[n_att_name][n] for n in g.vertices()])
    for k in unique_keys:
        node_dict[n_att_name][k] = set([n for n in g.vertices() if (g.vp[n_att_name][n] == k)])

    edge_dict = {e_att_name: {'size': int(g.num_edges())}}
    # find all the total values
    unique_keys = np.unique([g.ep[e_att_name][e] for e in g.edges()])
    for k in unique_keys:
        edge_dict[e_att_name][k] = set([(e.source(), e.target()) for e in g.edges() if g.ep[e_att_name][e] == k])
        edge_dict[e_att_name][k].update(set([(e.target(), e.source()) for e in g.edges() if g.ep[e_att_name][e] == k]))
        # mirror_edges = []
        # for e in edge_dict[e_att_name][k]:
        #     mirror_edges.append((e[1], e[0]))
        # edge_dict[e_att_name][k].update(mirror_edges)

    return node_dict, edge_dict


def debug_match_graph(_match_graph):
    _zero_id = vp_map(_match_graph, 'zero_id')
    _one_id = vp_map(_match_graph, 'one_id')
    v_names = _match_graph.new_vp('object')
    for v in _match_graph.vertices():
        v_names[v] = str(_zero_id[v]) + ", " + str(_one_id[v])
    gt_draw.graph_draw(_match_graph, vprops={'text': v_names}, output_size=(1000, 1000))


def sgm_match(t_graph: gt.Graph, g_graph: gt.Graph, delta, tau, n_idx, e_idx) -> gt.Graph:
    # T is a query tree
    # G is a query graph
    # Delta is the score delta that we can accept from perfect match
    # tau is how far off this tree is from the graph, at most.
    # nIdx is an index containing node attributes
    # eIdx is an index containing edge attributes
    # root_match = [n for n, d in list(T.in_degree().items()) if d == 0]

    root_match = [v for v in t_graph.vertices() if v.in_degree() == 0]
    root = root_match[0]
    n_keys = list(n_idx.keys())[0]
    e_keys = list(e_idx.keys())[0]

    #    print 'Building matching graph'

    print('Printing MDST Graph')
    print(root)
    print_graph(t_graph)

    # Step 1: Get all the matches for the nodes
    node_matches = dict()
    for v in t_graph.vertices():
        if t_graph.vp[n_keys][v] in list(n_idx[n_keys].keys()):
            node_matches[v] = n_idx[n_keys][t_graph.vp[n_keys][v]]
        else:
            node_matches[v] = set()

    # Step 2: Get all the edge matches for the node
    edge_matches = dict()
    for e in t_graph.edges():
        if t_graph.ep[e_keys][e] in list(e_idx[e_keys].keys()):
            edge_matches[e] = e_idx[e_keys][t_graph.ep[e_keys][e]]
        else:
            edge_matches[e] = set()
        # Make sure you count just the ones that have matching nodes too.
        edge_matches[e] = set(
            [em for em in edge_matches[e] if
             em[0] in node_matches[e.source()] and
             em[1] in node_matches[e.target()]])

    # Scoring, initially, is going to be super-simple:
    # You get a 1 if you match, and a 0 if you don't.  Everything's created equal.

    # Score everything and put it in a graph.

    for k in list(edge_matches.keys()):
        if len(edge_matches[k]) == 0:
            pass
            # stop_here = 1

    match_graph = gt.Graph(directed=True)
    #    for nT in T.nodes():
    #        for nG in node_matches[nT]:
    #            MatchGraph.add_node(tuple([nT,nG]),score=1,solo_score=1)
    mg_edges = set()
    mg_vertices = set()
    mg_vertices_to_index = {}
    for eT in t_graph.edges():
        for eG in edge_matches[eT]:
            v1 = (eT.source(), eG[0])
            v2 = (eT.target(), eG[1])
            mg_vertices.add(v1)
            mg_vertices.add(v2)
            mg_edges.add((v1, v2))

    # match_graph.add_edge([(eT.source(), eG.source()), (eT.target(), eG.target())])
    zero_id = vp_map(match_graph, 'zero_id')
    one_id = vp_map(match_graph, 'one_id')

    for tup in mg_vertices:
        v = match_graph.add_vertex()
        zero_id[v], one_id[v] = tup
        mg_vertices_to_index[tup] = v

    # it = iter(mg_vertices)
    # for v in match_graph.vertices():
    #     tup = next(it)
    #     zero_id[v], one_id[v] = tup
    #     mg_vertices_to_index[tup] = v

    for t1, t2 in mg_edges:
        match_graph.add_edge(mg_vertices_to_index[t1], mg_vertices_to_index[t2])

    # debug_match_graph(match_graph)

    solo_score_vp = vp_map(match_graph, 'solo_score', 'int')
    score_vp = vp_map(match_graph, 'score_v', 'int')
    score_ep = ep_map(match_graph, 'score_e', 'int')
    path_vp = vp_map(match_graph, 'path', 'object')

    g_graph_original = original_vp(g_graph)
    t_graph_original = original_vp(t_graph)

    for v in match_graph.vertices():
        solo_score_vp[v] = 1
        score_vp[v] = 1

        # Here we insert original nodes
        d = coll.deque()
        d.append((t_graph_original[zero_id[v]], g_graph_original[one_id[v]]))
        path_vp[v] = d

    for e in match_graph.edges():
        score_ep[e] = 1

    # gt_draw.graph_draw(match_graph, vprops={'text': zero_id})

    # Get rid of anybody flying solo
    match_graph = clear_unconnected(match_graph, root)  # this is clearly not working.

    # Now acquire/organize all hypotheses with scores above Max_Score - tau - delta

    # Figure out how much score you could possibly get at every node in the query.
    max_score_v = vp_map(t_graph, 'max_score_v', 'int')
    max_score_e = ep_map(t_graph, 'max_score_e', 'int')
    score_vp = vp_map(match_graph, 'score_v', 'int')
    score_ep = ep_map(match_graph, 'score_e', 'int')
    path_vp = vp_map(match_graph, 'path', 'object')
    zero_id = vp_map(match_graph, 'zero_id')

    # gt_draw.graph_draw(match_graph, vprops={'text': zero_id})

    for n in t_graph.vertices():
        max_score_v[n] = 1
    for e in t_graph.edges():
        max_score_e[e] = 1

    bfs_edges = list(gt_s.bfs_iterator(t_graph, source=root))
    reversed_bfs_edges = list(reversed(bfs_edges))

    t_index = t_graph.vertex_index

    # debug_match_graph(match_graph)

    for e in reversed_bfs_edges:  # Reverse BFS search - should do leaf nodes first.
        # What's the best score we could get at this node?
        v1, v2 = e

        max_score_v[v1] += max_score_v[v2] + max_score_e[e]

        # Find all the edges equivalent to this one in the match graph
        edge_matches = [(eG1, eG2) for eG1, eG2 in match_graph.edges() if
                        zero_id[eG1] == t_index[v1] and zero_id[eG2] == t_index[v2]]

        parent_nodes = set([eM1 for eM1, eM2 in edge_matches])

        for p in parent_nodes:
            child_nodes = [eM2 for eM1, eM2 in edge_matches if eM1 == p]
            # First, check if the bottom node has a score
            best_score = 0
            # best_node = None
            c_path = None
            for c in child_nodes:
                c_edge = match_graph.edge(p, c)
                c_score = score_vp[c] + score_ep[c_edge]
                c_path = path_vp[c]

                if c_score > best_score:
                    best_score = c_score
                    # best_child_path = c_path
            score_vp[p] += best_score
            for pathNode in c_path:
                path_vp[p].appendleft(pathNode)

    leave_prop = match_graph.new_vertex_property('bool')

    # CLEAN IT UP.
    for n in match_graph.vertices():
        leave_prop[n] = score_vp[n] >= max_score_v[t_graph.vertex(zero_id[n])] - delta

    sub = gt.GraphView(match_graph, leave_prop)
    new_match_graph = create_q_graph(sub, add_back_reference=False)

    # Get rid of anybody flying solo
    match_graph = save_root_children(new_match_graph, root)
    zero_id = vp_map(match_graph, 'zero_id')
    one_id = vp_map(match_graph, 'one_id')
    path_list_vp = vp_map(match_graph, 'path_list', 'object')
    for n in match_graph.vertices():
        d = coll.deque()
        d.append((t_graph_original[zero_id[n]], g_graph_original[one_id[n]]))
        path_list_vp[n] = [d]

    # Get a list of solutions alive in the graph
    for e in reversed_bfs_edges:
        v1, v2 = e
        edge_matches = [(eG1, eG2) for eG1, eG2 in match_graph.edges() if
                        zero_id[eG1] == t_index[v1] and zero_id[eG2] == t_index[v2]]

        parent_nodes = set([eM1 for eM1, eM2 in edge_matches])

        for p in parent_nodes:
            child_nodes = [eM2 for eM1, eM2 in edge_matches if eM1 == p]
            # First, check if the bottom node has a score
            tmpList = []
            for c in child_nodes:
                for _p in path_list_vp[p]:
                    for _c in path_list_vp[c]:
                        tmpList.append(_p + _c)
            path_list_vp[p] = tmpList

    # debug_match_graph(match_graph)

    # Score the root solutions
    return match_graph


def original_vp(g: gt.Graph):
    return vp_map(g, 'original', 'int')


def dict_from_property_map(g: gt.Graph, p: gt.PropertyMap) -> dict:
    s = {}
    for v in g.vertices():
        s[p[v]] = v

    return s


def score_solution(q: gt.Graph, a: gt.Graph, solution):
    # Archive graph A
    # query graph Q
    # Solution = list of tuples, length |G|, e.g. (1,3),(2,8),(3,12)
    solution_score = 0
    sol_dict = {}
    sol_dict_new = {}

    q_original = original_vp(q)
    a_original = original_vp(a)
    q_o = dict_from_property_map(q, q_original)
    a_o = dict_from_property_map(a, a_original)

    q_v_map = vp_map(q, 'nValue')
    a_v_map = vp_map(a, 'nValue')

    q_e_map = ep_map(q, 'eValue')
    a_e_map = ep_map(a, 'eValue')

    # Current impl simply adds 1 for each matching node or edge
    for s in solution:  # s should be a tuple e.g. (1,3)
        q_node, a_node = s
        if q_node in q_o and a_node in a_o:
            solution_score += q_v_map[q_o[q_node]] == a_v_map[a_o[a_node]]
            sol_dict[q_node] = a_node
            sol_dict_new[q_o[q_node]] = a_o[a_node]

    for e_q in q.edges():
        u_q, v_q = e_q
        if u_q in sol_dict_new and v_q in sol_dict_new:
            edges_a = a.edge(sol_dict_new[u_q], sol_dict_new[v_q], all_edges=True)
            for e_a in edges_a:
                if q_e_map[e_q] == a_e_map[e_a]:
                    u_a, v_a = e_a
                    solution_score += 1
                    u1 = q_original[u_q]
                    v1 = q_original[v_q]
                    u2 = a_original[u_a]
                    v2 = a_original[v_a]

                    sol_dict[(u1, v1)] = (u2, v2)
                    break

    return sol_dict, solution_score


def clear_unconnected(g: gt.Graph, key_node) -> gt.Graph:
    # Get rid of any component that doesn't contain the key node.
    label_map, hist = gt_top.label_components(g, directed=False)

    zero_id = vp_map(g, 'zero_id', 'int')
    key_comps = set()
    for v in g.vertices():
        if zero_id[v] == key_node:
            key_comps.add(label_map[v])

    leave_prop = g.new_vertex_property('bool')
    for v in g.vertices():
        leave_prop[v] = label_map[v] in key_comps

    sub = gt.GraphView(g, leave_prop)
    new_g = create_q_graph(sub, add_back_reference=False)

    return new_g


def save_root_children(g: gt.Graph, root_id) -> gt.Graph:
    zero_id = vp_map(g, 'zero_id', 'int')
    roots = set()
    for v in g.vertices():
        if zero_id[v] == root_id:
            roots.add(v)

    leave_prop = g.new_vertex_property('bool')
    for v in roots:
        gt_top.label_out_component(g, v, leave_prop)

    sub = gt.GraphView(g, leave_prop)
    new_g = create_q_graph(sub, add_back_reference=False)
    return new_g


def path_exists(g: gt.Graph, source, target):
    vp, _ = gt_top.shortest_path(g, source, target)
    return len(vp) != 0


def subsample_archive_from_matching(a: gt.Graph, mg: gt.Graph, t_graph: gt.Graph, e_idx):
    leave_prop = a.new_vertex_property('bool')
    one_prop = vp_map(mg, 'one_id', 'int')

    to_save = set([one_prop[n] for n in mg.vertices()])
    for v in a.vertices():
        leave_prop[v] = a.vertex_index[v] in to_save

    # get rid of all the nodes in A that aren't there
    sub = gt.GraphView(a, leave_prop)
    new_g = create_q_graph(sub, add_back_reference=True)

    return new_g


def reduce_space(q: gt.Graph, a_graph: gt.Graph, min_score, delta, method='MDST'):
    num_edges_a = []
    num_nodes_a = []
    used_edges = set()
    # delta = 0  # set to non-zero for corruption experiments - njp

    # Hash
    print('Hashing...')
    start = time.time()
    n_idx, e_idx = create_dict(a_graph, 'nValue', 'eValue')  # Create an attribute index
    # print_weights(n_idx,e_idx)
    print('Done at ' + str(time.time() - start))

    alg_start = time.time()

    num_edges_a.append(a_graph.num_edges())
    num_nodes_a.append(a_graph.num_vertices())

    print('Calculating MDST')
    start = time.time()

    t_score = None
    t_graph = None
    if method == 'MDST' and len(used_edges) < 2 * q.num_edges():
        t_graph, t_score = calculate_mdst_v2(q, n_idx, e_idx, used_stuff=used_edges)
    if method == 'Normal' or len(used_edges) >= 2 * q.num_edges():
        if method == 'MDST':
            # stop_here = 1
            pass
        t_graph = calc_random_spanning_tree(q)
    print('Printing t_graph')
    print_graph(t_graph)
    print('t_score', t_score)
    print('Done at ' + str(time.time() - start))
    # print('Further the code for "graph_tool" is not translated. Further calculations are not considered valid.')
    # Add used stuff to used.
    # edges_used = t_graph.edges()
    # used_edges.update(edges_used)
    # used_edges.update([tuple([e2, e1]) for e1, e2 in edges_used])  #???WHAT FOR???

    # also figure out what is unused.
    tau = q.num_edges() - t_graph.num_edges()  # Dumb way of calculating tau

    print('Matching')
    start = time.time()
    mg = sgm_match(t_graph, a_graph, delta, tau, n_idx, e_idx)

    print('Matching done at ' + str(time.time() - start))
    print('Printing mg:')

    root = None
    for v in mg.vertices():
        if v.in_degree() == 0:
            root = v
            break
    print(root)
    print_graph(mg)
    a_prime = subsample_archive_from_matching(a_graph, mg, t_graph, e_idx)
    # print 'Printing a_prime'
    # PrintGraph(a_prime)
    # gt.draw.graph_draw(a_prime, vertex_text=a_prime.vp['old'], vertex_font_size=18, output_size=(300, 300),
    #                    output='a_prime.png')

    # Score the solutions
    print('Scoring solutions:')
    start = time.time()

    scores = []
    threshold_matches = []
    roots = [n for n in mg.vertices() if n.in_degree() == 0]
    path_list_vp = vp_map(mg, 'path_list', 'object')

    for root in roots:
        sol = path_list_vp[root]
        for _sol in sol:
            origin_nodes = []
            for _d in _sol:
                d = []
                for i in _d:
                    d.append(a_graph.vp['old'][a_graph.vertex(i)])
                origin_nodes.append(tuple(d))

            print('trying solutions', origin_nodes)
            match, score = score_solution(q, a_prime, _sol)
            origin_match = match_to_original_nodes(a_graph, match)
            print('Got match with score: ', score)
            print(origin_match)
            if score >= min_score:
                threshold_matches.append(origin_match)
                scores.append(score)
                for key, val in origin_match.items():
                    print('key {} val {}'.format(key, val))

    num_edges_a.append(a_prime.num_edges())
    num_nodes_a.append(a_prime.num_vertices())

    print('Algorithm Post-hash Stages Done at ' + str(time.time() - start))
    print('Algorithm Post-hash Stages Done at ' + str(time.time() - alg_start))
    # print('Threshold matches: {}'.format(len(threshold_matches)))

    return num_nodes_a, num_edges_a, threshold_matches, scores


def match_to_original_nodes(g, match):
    origin_match = {}
    for key in match.keys():
        if isinstance(key, tuple):
            _key = []
            _value = []
            for k in key:
                _key.append(g.vp['old'][g.vertex(k)])
            for v in match[key]:
                _value.append(g.vp['old'][g.vertex(v)])
            origin_match[tuple(_key)] = tuple(_value)
        else:
            k = g.vp['old'][g.vertex(key)]
            v = g.vp['old'][g.vertex(match[key])]
            origin_match[k] = v
    return origin_match


# I/O Utilities
def print_graph(g: gt.Graph):
    nodes = g.vertices()

    print('Directed' if g.is_directed() else 'Undirected')

    for node in nodes:
        prop = {}
        for key in g.vp.keys():
            prop[key] = g.vp[key][node]
        print(node, ' : ', prop)
    for e in g.edges():
        prop = {}
        for key in g.ep.keys():
            prop[key] = g.ep[key][e]
        print(e, ' : ', prop)


def print_weights(n_idx, e_idx):
    n_att_name = list(n_idx.keys())[0]
    e_att_name = list(e_idx.keys())[0]

    for k in n_idx[n_att_name]:
        if k != 'size':
            print('Node attribute ' + str(k) + ':' + str(len(n_idx[n_att_name][k]) / n_idx[n_att_name]['size']))

    for k in e_idx[e_att_name]:
        if k != 'size':
            print('Edge attribute ' + str(k) + ':' + str(len(e_idx[e_att_name][k]) / e_idx[e_att_name]['size']))


if __name__ == '__main__':
    from DirectedWeightedGraph import DWGraph

    g = DWGraph.from_file_edge_list("art_ex1.graph",
                                    is_weighted=False,
                                    make_undirected=not False,
                                    skip_first_line=False)
    g.add_labels_from_file("art_ex1.labels")

    g0 = DWGraph.from_file_edge_list("art_template1.graph",
                                     is_weighted=False,
                                     make_undirected=not False,
                                     skip_first_line=False)
    g0.add_labels_from_file("art_template1.labels")
    g = add_original_enumeration(g)
    g0 = add_original_enumeration(g0)
    prop_label_g = g.vp['label']
    prop_label_g0 = g0.vp['label']
    prop_g_v = vp_map(g, 'nValue', 'int')
    for i, v in enumerate(g.vertices()):
        prop_g_v[v] = prop_label_g[v]

    prop_g0_v = vp_map(g0, 'nValue', 'int')
    for i, v in enumerate(g0.vertices()):
        prop_g0_v[v] = prop_label_g0[v]

    prop_g = ep_map(g, 'eValue', 'int')
    for e in g.edges():
        prop_g[e] = 1

    prop_g0 = ep_map(g0, 'eValue', 'int')
    for e in g0.edges():
        prop_g0[e] = 1

    reduce_space(g0, g, 0, 0, method='MDST')
