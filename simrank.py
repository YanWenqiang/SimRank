import itertools
 
import numpy as np
import networkx as nx
from typedecorator import params, returns
from pygraph.classes.digraph import digraph


@params(G=nx.Graph, r=float, max_iter=int, eps=float)
def simrank(G, r=0.8, max_iter=100, eps=1e-4):
    if isinstance(G, nx.MultiGraph):
        assert("The SimRank of MultiGraph is not supported.")
    if isinstance(G, nx.MultiDiGraph):
        assert("The SimRank of MultiDiGraph is not supported.")
    directed = False
    if isinstance(G, nx.DiGraph):
        directed = True
    nodes = G.nodes()
    nodes_i = {}
    for (k, v) in [(nodes[i], i) for i in range(0, len(nodes))]:
        nodes_i[k] = v
    sim_prev = np.zeros(len(nodes))
    sim = np.identity(len(nodes))
    for i in range(max_iter):
        if np.allclose(sim, sim_prev, atol=eps):
            break
        sim_prev = np.copy(sim)
        for u, v in itertools.product(nodes, nodes):
            if u is v: continue
            if directed:
                u_ns, v_ns = G.predecessors(u), G.predecessors(v)
            else:
                u_ns, v_ns = G.neighbors(u), G.neighbors(v)
            # Evaluating the similarity of current nodes pair
            if len(u_ns) == 0 or len(v_ns) == 0:
                sim[nodes_i[u]][nodes_i[v]] = 0
            else:
                s_uv = sum([sim_prev[nodes_i[u_n]][nodes_i[v_n]] for u_n, v_n in itertools.product(u_ns, v_ns)])
                sim[nodes_i[u]][nodes_i[v]] = (r * s_uv) / (len(u_ns) * len(v_ns))
    print("Converge after %d iterations (eps=%f)." % (i, eps))
    return sim

@params(G = nx.DiGraph, r = float, max_iter = int, eps = float)
def simrank_bipartite(G, r = 0.8, max_iter = 100, eps = 1e-4):
    if not nx.is_bipartite(G):
        assert("A bipartite graph is required")

    nodes = G.nodes()
    nodes_i = {node: i for i, node in enumerate(nodes)}
    
    sim_prev = np.zeros(len(nodes))
    sim = np.identity(len(nodes))

    lns = {}
    rns = {}
    for n in nodes:
        preds = G.predecessors(n)
        succs = G.successors(n)
        if len(preds) == 0:
            lns[n] = succs
        else:
            rns[n] = preds
        
    def _update_partite(ns):
        for u,v in itertools.product(ns.keys(), ns.keys()):
            if u == v: continue
            u_ns, v_ns = ns[u], ns[v]
            if len(u_ns) == 0 or len(v_ns) == 0:
                sim[nodes_i[u]][nodes_i[v]] = 0
            else:
                s_uv = sum([sim_prev[nodes_i[u_n]][nodes_i[v_n]] for u_n, v_n in itertools.product(u_ns, v_ns)])
                sim[nodes_i[u]][nodes_i[v]] = (r * s_uv) / (len(u_ns) * len(v_ns))
    for i in range(max_iter):
        if np.allclose(sim, sim_prev, atol = eps):
            break
        sim_prev = np.copy(sim)
        _update_partite(lns)
        _update_partite(rns)

    print("Converge after %d iterations (eps=%f)." % (i, eps))
 
    return sim


class PRIterator:
    __doc__ = '''计算一张图中的PR值'''
    
    def __init__(self, dg):
        self.damping_factor = 0.85 # 阻尼系数， 即a
        self.max_iterations = 100
        self.min_delta = 0.0001
        self.graph = dg
        
    def page_rank(self):
        # 先将图中没有出链的节点改为对所有节点都有出链
        for node in self.graph.nodes():
            if len(self.graph.neighbors(node)) == 0:
                for node2 in self.graph.nodes():
                    digraph.add_edge(self.graph, (node, node2))
        
        nodes = self.graph.nodes()
        graph_size = len(nodes)
        
        if graph_size == 0:
            return {}
        ### 初始化 pr
        page_rank = dict.fromkeys(nodes, 1.0 / graph_size)
        damping_value = (1.0 - self.damping_factor) / graph_size
        
        flag = False
        for i in range(self.max_iterations):
            change = 0
            for node in nodes:
                rank = 0
                for incident_page in self.graph.incidents(node):
                    rank += self.damping_factor * (page_rank[incident_page] / len(self.graph.neighbors(incident_page)))
                rank += damping_value
                change += abs(page_rank[node] - rank)
                page_rank[node] = rank
            
            print("This is NO.%s iterations" % (i+1))
            print(page_rank)
            
            if change < self.min_delta:
                flag = True
                break
                
        if flag:
            print("finished in %s iterations!" % node)
        else:
            print("finished out of 100 iterations!")
        
        return page_rank

if __name__ == '__main__':
    G = nx.DiGraph()
    G.add_edges_from([(1,2), (1,3), (2,4), (4,1), (3,5), (5,3)])
    print(simrank(G))
 

    G = nx.DiGraph()
    G.add_edges_from([(1,3), (1,4), (1,5), (2,4), (2,5), (2,6)])
    print(simrank_bipartite(G))

    dg = digraph()
    dg.add_nodes(["A", "B", "C", "D", "E"])

    dg.add_edge(("A", "B"))
    dg.add_edge(("A", "C"))
    dg.add_edge(("A", "D"))
    dg.add_edge(("B", "D"))
    dg.add_edge(("C", "E"))
    dg.add_edge(("D", "E"))
    dg.add_edge(("B", "E"))
    dg.add_edge(("E", "A"))
    
    pr = PRIterator(dg)
    page_ranks = pr.page_rank()
    
    print("The final page rank is\n", page_ranks)