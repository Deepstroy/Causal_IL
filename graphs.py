import networkx as nx

class nx_graph(nx.DiGraph):
    def __init__(self, nodes, edges, bidirected_edges=None, ordered_topology=None):
        super(nx_graph, self).__init__(edges)
        self.nodes = nodes
        self.bidirected_edges = bidirected_edges
        self.ordered_topology = ordered_topology