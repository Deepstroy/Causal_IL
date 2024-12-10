import networkx as nx
import itertools
from src.utils.utils import pairs_to_dict_bi, pairs_to_dict, union, visited_deque
from src.utils.graphs import nx_graph
from typing import AbstractSet as ASet, Tuple, Union, FrozenSet, Collection
from agent_utils import StructuralCausalModel as SCM
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing as mp


import pandas as pd
import numpy as np


Vars = Union[str, ASet[str]]

def _wrap(v_or_vs: Vars) -> FrozenSet[str]: # make frozenset
    return frozenset({v_or_vs}) if isinstance(v_or_vs, str) else frozenset(v_or_vs)

class Agent:
    '''
    test_g10 = nx_graph({'A', 'C', 'X', 'Y'},
                        {('C', 'X'), ('X', 'Y')},
                        {('A','X'),('A','Y'),('C','Y')})
    '''
    def __init__(self, dag: nx): #, is_expert=False):
        self.G = dag
        self.edges = set(self.G.edges)
        self.Vs = set(self.G.nodes)
        self.bidirected_edges = self.G.bidirected_edges
        self.order = self.G.order
        self._ch = pairs_to_dict(self.edges)
        self._pa = pairs_to_dict(self.edges, reverse=True)
        self._sib = pairs_to_dict_bi(dag.bidirected_edges) # ('Y':['W1','W3'], ...)

    def bi_dir(self, V: Vars) -> FrozenSet[str]:
        """bi-directed edges of a vertex"""
        #print(self._sib[V])
        return self._sib[V]

    def pa(self, V_or_Vs: Vars) -> FrozenSet[str]:
        """ parents of a vertex or vertices """
        Vs = _wrap(V_or_Vs)
        return union(self._pa[V] for V in Vs)

    def an(self, V_or_Vs: Vars) -> FrozenSet[str]:
        """ ancestors of a vertex or vertices """
        return self.__an(_wrap(V_or_Vs))

    def __an(self, V_or_Vs: FrozenSet[str]) -> FrozenSet[str]:
        Vs = _wrap(V_or_Vs)
        if not self.pa(Vs):
            return frozenset()
        return self.__an(self.pa(Vs)) | self.pa(Vs)

    def ch(self, V_or_Vs: Vars) -> FrozenSet[str]:
        """ children of a vertex or vertices """
        Vs = _wrap(V_or_Vs)
        return union(self._ch[V] for V in Vs)

    def de(self, V_or_Vs: Vars) -> FrozenSet[str]:
        return self.__de(_wrap(V_or_Vs))

    def __de(self, V_or_Vs: Vars) -> FrozenSet[str]:
        """ descendants of a vertex or vertices """
        Vs = _wrap(V_or_Vs)
        if not self.ch(Vs):
            return frozenset()
        return self.__de(self.ch(Vs)) | self.ch(Vs)

    def before(self, X, order):
        return set(order[:order.index(X)])

    def topological_order(self, backward = False):
        gg = nx.DiGraph(self.edges)
        gg.add_nodes_from(self.Vs)
        top_to_bottom = list(nx.topological_sort(gg))
        if backward:
            return tuple(reversed(top_to_bottom))
        else:
            return tuple(top_to_bottom)

    def d_separation(self, Xs: Vars, Ys: Vars, Zs: Vars = frozenset()) -> bool:
        Xs, Ys, Zs = _wrap(Xs), _wrap(Ys), _wrap(Zs)
        Xs = Xs - Zs
        Ys = Ys - Zs
        if Xs & Ys: # not disjoint node
            return False
        if not Xs or not Ys: # no node in the sets
            return True
        return all(self.__inner_d_separation(X, Y, Zs) for X, Y in itertools.product(Xs, Ys))

    def __inner_d_separation(self, X: str, Y: str, Zs: FrozenSet[str]) -> bool:
        # For *semi-markovian graph
        # create deque instance
        q = visited_deque({(W, True) for W in self.ch(X)} |
                          {(W, False) for W in self.pa(X)} |
                          {(W, True) for W in self.bi_dir(X)})
        while q:
            W, is_child = q.pop() # W is current node
            if W == Y:  # reached to Y
                return False
            if is_child:  # if child (..->W)
                if (Zs & self.de(W)) or W in Zs:
                    q.extend({(parent, False) for parent in self.pa(W)}) #the next node is parent(False)
                    q.extend(({(sib, True) for sib in self.bi_dir(W)})) #the next node can be treated as child of W
                if W not in Zs:
                    q.extend({(child, True) for child in self.ch(W)})
            else:  # if parent(..<-W)
                if W not in Zs:
                    q.extend({(child, True) for child in self.ch(W)})
                    q.extend({(parent, False) for parent in self.pa(W)})
                    q.extend(({(sib, True) for sib in self.bi_dir(W)}))  # the next node can be treated as child of W
        return True

    def get_c_factors(self, V_or_Vs):
        """return c-components nodes of this graph(class)"""
        if isinstance(V_or_Vs, str):
            V_or_Vs = set({V_or_Vs})
        return union(self.__get_c_factors(V, set()) for V in V_or_Vs)

    def __get_c_factors(self, V, tmp):
        """inner loop function for get_c_factors which returns c-components nodes"""
        bi_di_not_visit = self.bi_dir(V) - tmp # self.bi_dir(V)
        if not bi_di_not_visit:
            return {V}
        tmp |= {V}
        for node in bi_di_not_visit:
            tmp |= self.__get_c_factors(node, tmp)
        return tmp

    def do_graph(self, Xs):
        '''
        :param Xs: intervened node
        :return: Manipulated do-graph
        '''
        Xs = _wrap(Xs)
        # Manipulated graph for do operation
        new_di_g = [(k, v) for k, v in self.edges if v not in Xs]
        new_bi_di_g = []
        for k, v in self.bidirected_edges:
            if v not in Xs and k not in Xs:
                new_bi_di_g.append((k, v))

        mod_G = Agent(nx_graph({}, new_di_g, new_bi_di_g))
        return mod_G

    def backdoor_admissible(self, Xs, Ys, Zs) -> bool:
        Xs, Ys = _wrap(Xs), _wrap(Ys)
        # Manipulated graph e.g., ( {X} -x-> {...} )
        new_g = [(k, v) for k, v in self.edges if k not in Xs]
        mod_G = Agent(nx_graph({}, new_g))
        # ID test
        return True if mod_G.d_separation(Xs, Ys, Zs) else False

    def subgraph(self, nodes):
        """Return subgraph corresponding with input nodes """
        new_edges = set()
        new_bi_di = set()
        if nodes == None:
            return self.edges, self.bidirected_edges
        for node in nodes: # [a,b,c] -> a, pa[a] => {'a' : ['b','c'], 'b':['a'], ....}
            for pa in self.pa(node):
                if pa in nodes:
                    new_edges.add((pa, node))
            for ch in self.ch(node):
                if ch in nodes:
                    new_edges.add((node, ch))
            for sib in self.bi_dir(node):
                if sib in nodes:
                    new_bi_di.add((node, sib))
                    new_bi_di.add((sib, node))
        return new_edges, new_bi_di


    def identify(self, C: set, T=None) -> bool:
        """An identify function determining if Q[C] is computable from Q[T] """
        # Summation of An(C)
        new_edges, new_bi_di = self.subgraph(T)
        mod_G = Agent(nx_graph({}, new_edges, new_bi_di))
        A = mod_G.an(C) | C
        if A == C:
            return True
        if A == T:
            return False

        # Q-decomposition
        new_edges_A, new_bi_di_A = self.subgraph(A)
        mod_G_A = Agent(nx_graph({}, new_edges_A, new_bi_di_A))
        c_components_C = mod_G_A.get_c_factors(C)
        print(c_components_C)
        return self.identify(C, c_components_C)


    # Imitation Learning with Unobserved Confounders (Junzhe Zhang, 2020)
    def imitate(self, Xs: set, Ys: set):
        """Return the policy space that can imitate expert policy"""
        # Imitate the functions of listIDspace
        list_pi = []
        for policy in self.listIDspace(Xs, Ys):
            for surrogate_S in self.listMINsep(Xs, Ys, policy):
                if self.identify(surrogate_S, policy):
                    #(surrogate_S, policy) : imitation instrument
                    list_pi.append([surrogate_S, policy])
        return list_pi

    def listMINsep(self, Xs, Ys, Zs): # !!! 논문 기반 재구현
        # find the surrogate space S (i.e., X _||_ Y | S)
        setcandiS = []  # Zs is conditioned from the "listIDspace"
        #Vs = self.Vs - Xs - Ys - Zs
        Vs = Zs - Xs - Ys
        for num_ele in range(len(Vs)+1):
            for candiS in itertools.combinations(Vs, num_ele):
                if self.d_separation(Xs, Ys, set(candiS) | Zs): #
                    setcandiS.append(set(candiS))
        return setcandiS

    def listIDspace(self, Xs: set, Ys: set):
        """Find all identifiable subspaces w.r.t. (G, PI, Y)"""
        Zs = self.Vs - Xs - Ys - self.de(Xs) # (Z = V \X \Y \de(X))
        candiZ = list()
        self.__listIDspaceHelper(Xs, Ys, set(), Zs, candiZ)
        return candiZ

    def __listIDspaceHelper(self, Xs, Ys, PIL: set, PIR: set, candiZ):
        """inner loop function for listIDspace"""
        if self.identify(Ys, PIL | Ys):
            candiZ.append(PIL)
        if PIL == PIR or len(PIR) == 0: # end of search
            return
        else:
            V = next(iter(PIR - PIL))
            self.__listIDspaceHelper(Xs, Ys, PIL | {V}, PIR | {V}, candiZ)
            self.__listIDspaceHelper(Xs, Ys, PIL - {V}, PIR - {V}, candiZ)

    # Sequential Imitation Learning with Unobserved Confounders (Daniel Kumor, 2021)
    def order_based_topology(self, ordered_by, origin_order) -> Tuple:
        """ Reorder nodes based on origin order """
        _ordered = ()
        for e in origin_order:
            if e in ordered_by:
                _ordered = _ordered + (e,)
        return _ordered

    def hasvalidAdjustment(self, Ox, Oi, Xi, Y, order=None):
        GY = self.an(Y)|Y
        G_edges, G_bi_di = self.subgraph(GY)
        G_Y_new = Agent(nx_graph({}, G_edges, G_bi_di))

        C_comps = G_Y_new.get_c_factors(Oi)

        GC_ele = G_Y_new.do_graph(G_Y_new.pa(C_comps)).an(C_comps) | C_comps
        GC_edges, GC_bi_di = G_Y_new.subgraph(GC_ele)
        GC = Agent(nx_graph({}, GC_edges, GC_bi_di))

        OC = C_comps - (Ox | {Oi})
        return GC.d_separation(Oi, OC, OC & self.before(Xi, order))


    def findOx(self, Os, X, Y, order=None):
        Ox = dict() # elements of O to X
        GY = self.an(Y)|Y
        G_edges, G_bi_di = self.subgraph(GY) # get edges from ancestor nodes of Y
        G_Y_new = Agent(nx_graph({}, G_edges, G_bi_di)) # manipulative ancestral Y graph

        while True:
            pastOx = dict(Ox)
            for Oi in G_Y_new.topological_order(True):
                # Oi의 children이 존재 & Oi의 childern이 Ox의 key에 존재
                if G_Y_new.ch(Oi) and G_Y_new.ch(Oi) <= Ox.keys():
                    Xi = self.order_based_topology({Ox[_] for _ in G_Y_new.ch(Oi)}, G_Y_new.topological_order())[0]
                    if self.hasvalidAdjustment(set(Ox.keys()), Oi, Xi, {'Y'}, order=order):
                        Ox[Oi] = Xi
                elif Oi in X and self.hasvalidAdjustment(set(Ox.keys()), Oi, Oi, {'Y'}, order=order):
                    Ox[Oi] = Oi
            if pastOx == Ox:
                break
        return set(Ox.keys())


    def get_seq_piBD(self, Xs, Y, order=None):
        Ox = self.findOx(self.Vs, Xs, Y, order)
        Xs_prime = Ox & Xs

        # markov_boundary
        n_egdes, n_bidi = self.subgraph(self.an(Y)|Y)
        dag_anY_doXs = Agent(nx_graph({}, n_egdes, n_bidi)).do_graph(Xs_prime)
        c_comp_ox = dag_anY_doXs.get_c_factors(dag_anY_doXs.ch(Ox)|Ox)
        Zs = dag_anY_doXs.pa(c_comp_ox)|c_comp_ox - Ox # markov boundary

        # boundary action
        XBs = {Xi for Xi in Xs & self.Vs if not (self.ch(Xi) <= Ox)}

        # get seq-pi-BD
        policy = {Xi: (Zs | XBs) & self.before(Xi, order) for Xi in Xs_prime}
        return policy

def __inner(graph):
    '''
    Get an error of imitator based on "A SCM" sampled from given graph G
    return Squared Error (SE) between expert and imitator
    :param graph: Graph shared by both of two agent (expert, imitator)
    :return: Squared Error (SE)
    '''
    Imitator = Agent(graph)
    policy = Imitator.get_seq_piBD({'X1', 'X2'}, {'Y'}, Imitator.order)

    # Original (expert) data
    M = SCM(Imitator)
    data_df = M.sample_binary_data(sample_size=1000,
                                   intervened=False)
    E_EY = data_df['Y'].mean()

    # Soft intervened (imitator) data when data have noise (beta dist.)
    data_intervened_df = M.sample_binary_data(sample_size=1000,
                                              policy=policy,
                                              intervened=True)
    I_EY = data_intervened_df['Y'].mean()
    return pow(E_EY - I_EY, 2)

def main():
    test_imi_2 = nx_graph({'Z', 'X1', 'X2', 'Y'},
                          {('X1', 'X2'), ('X2', 'Y'), ('Z', 'Y')},
                          {('X1', 'Z')},
                          ['Z', 'X1', 'X2', 'Y'])

    test_imi_3 = nx_graph({'X1', 'Z', 'X2', 'Y'},
                          {('X1', 'X2'), ('X2', 'Y'), ('Z', 'Y')},
                          {('X1', 'Z')},
                          ['X1', 'Z', 'X2', 'Y'])

    dict = []
    for _ in tqdm(range(3)): # 1000
        dict.append([__inner(test_imi_2), __inner(test_imi_3)])

    print(np.array(dict).mean(axis=0))

    # diff_array2 = Parallel(n_jobs=2)(delayed(__inner)(test_imi_2) for _ in range(1000))
    #diff_array3 = Parallel(n_jobs=1)(delayed(__inner)(test_imi_3) for _ in range(100))
    # print('graph2:', np.array(diff_array2).mean())
    #print('graph3:', np.array(diff_array3).mean())

    # print('Expert E[Y]:', E_EY)
    # print('Imitator E[Y_hat]:', I_EY)
    # print('|E[Y] - E[Y_hat]|:', abs(E_EY - I_EY))



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Confounding graph
    test_g = nx_graph({'X', 'Y', 'Z'},
                      {('Z', 'X'), ('Z', 'Y')})
    # Primer's graph
    test_g2 = nx_graph({'X', 'Y', 'Z', 'W', 'U'},
                       {('X', 'Y'), ('X', 'W'), ('Z', 'W'), ('W', 'U')})
    # Collider without X->Y
    test_g3 = nx_graph({'X', 'Y', 'Z'},
                       {('X', 'Z'), ('Y', 'Z')})
    # Backdoor graph with two Z1, Z2
    test_g4 = nx_graph({'X', 'Y', 'Z1', 'Z2'},
                       {('Z1', 'X'), ('Z2', 'X'), ('Z1', 'Y'), ('Z2', 'Y'), ('X', 'Y')})
    # Backdoor graph with two Z1, Z2 and surrogate S
    test_g4 = nx_graph({'X', 'Y', 'Z1', 'Z2', 'S'},
                       {('Z1', 'X'), ('Z2', 'X'), ('Z1', 'S'), ('Z2', 'S'),
                        ('X', 'S'), ('S', 'Y')})
    # Backdoor graph with two Z1, Z2 and surrogate S
    test_g5 = nx_graph({'X', 'Y', 'Z1', 'Z2', 'Z3', 'S'},
                       {('X', 'Z1'), ('Z2', 'X'), ('X', 'S'), ('Z3', 'Z1'),
                        ('Z2', 'Z3'), ('Z3', 'S'), ('Z2', 'S'), ('S', 'Y')})
    # Front door graph1
    test_g6 = nx_graph({'X', 'Z', 'Y'},
                       {('X', 'Z'), ('Z', 'Y')},
                       {('X', 'Y')})
    # Napkin graph
    test_g7 = nx_graph({'X', 'W1', 'W2', 'Y'},
                       {('X', 'Y'), ('W1', 'W2'), ('W2', 'X')},
                       {('X', 'W1'), ('W1', 'Y')})
    # X-> W1 -> W2 -> W3 -> Y with bi-directed edges (X,W2,Y), (W1, W3)
    test_g8 = nx_graph({'X', 'W1', 'W2', 'W3', 'Y'},
                       {('X', 'W1'), ('W1', 'W2'), ('W2', 'W3'), ('W3', 'Y')},
                       {('X', 'W2'), ('W2', 'Y'), ('W1', 'W3')})
    # X  W2  Y with bi-directed edges (X,W2) (W2,Y)
    test_g9 = nx_graph({'X', 'W2', 'Y'},
                       {},
                       {('X', 'W2'), ('W2', 'Y')})
    # midterm graph-(1)
    test_g10 = nx_graph({'A', 'C', 'X', 'Y'},
                        {('C', 'X'), ('X', 'Y')},
                        {('A','X'),('A','Y'),('C','Y')})
    # bow graph
    test_g11 = nx_graph({'X', 'Y'},
                        {('X', 'Y')},
                        {('X', 'Y')})

    # Imitation learning graphs with temporal order
    test_imi_1 = nx_graph({'Z', 'X1', 'X2', 'Y'},
                          {('X1', 'X2'), ('X2', 'Y')},
                          {('X1', 'Z'), ('X2', 'Z'), ('Z', 'Y')},
                          ['Z', 'X1', 'X2', 'Y'])

    test_imi_2 = nx_graph({'Z', 'X1', 'X2', 'Y'},
                          {('X1', 'X2'),('X2', 'Y'), ('Z', 'Y')},
                          {('X1', 'Z')},
                          ['Z', 'X1', 'X2', 'Y'])

    test_imi_3 = nx_graph({'X1', 'Z', 'X2', 'Y'},
                          {('X1', 'X2'), ('X2', 'Y'), ('Z', 'Y')},
                          {('X1', 'Z')},
                          ['X1', 'Z', 'X2', 'Y'])

    test_imi_4 = nx_graph({'X1', 'Z', 'X2', 'Y'},
                          {('Z', 'X2'), ('X2', 'Y'), ('X1', 'Y')},
                          {('X1', 'Z'), ('Z', 'Y')},
                          ['X1', 'Z', 'X2', 'Y'])


    # Fully mSBD-1
    test_full_imi_mSBD = nx_graph({'X1', 'Z1', 'X2', 'Z2', 'Y1', 'Y2'},
                          {('Z1', 'X1'), ('Z1', 'Y1'), ('X1', 'Y1'),
                           ('Z2', 'X2'), ('Z2', 'Y2'), ('X2', 'Y2'),
                           ('Z1', 'X2'), ('Z1', 'Y2'), ('Z1', 'Y1'),
                           ('X1', 'X2'), ('X1', 'Y2'), ('X1', 'Z2'),
                           ('Y1', 'Z2'), ('Y1', 'X2'), ('Y1', 'Y2')},
                          {},
                          ['Z1','X1', 'Y1', 'Z2', 'X2', 'Y2'])


    # Fully mSBD-2?
    test_full_imi_mSBD2 = nx_graph({'X1', 'X2', 'X3', 'Z1', 'Z2', 'Z3', 'Y1'},
                          {('X1', 'X2'), ('X2', 'X3'),
                           ('Z1', 'Z2'), ('Z2', 'Z3'),
                           ('Z3', 'Y'), ('X3', 'Y')},
                          {('Z1', 'X1')},
                          ['X1', 'Z1', 'X2', 'Z2', 'X3', 'Z3', 'Y'])


    #### self-ai 2024 experiments
    # From Daniel Kumor's algorithm
    # test_msbd1 = nx_graph({'Z1', 'X1', 'X2', 'Y1', 'Z2', 'X3', 'X4', 'Y2'},
    #                       {('X1', 'X2'), ('X2', 'Y1'), ('Z1', 'Y1'),
    #                        ('X3', 'X4'), ('X4', 'Y2'), ('Z2', 'Y2'),
    #                        ('Y1', 'X3')},
    #                       {('X1', 'Z1'), ('X3', 'Z2')},
    #                       ['Z1', 'X1', 'X2', 'Y1', 'Z2', 'X3', 'X4', 'Y2'])
    #
    #
    # Imitator = Agent(test_msbd1)
    # policy = Imitator.get_seq_piBD({'X1', 'X2', 'X3', 'X4'}, {'Y1', 'Y2'}, Imitator.order)
    # print(policy)

    # Imitator = Agent(test_imi_mSBD)
    # policy = Imitator.get_seq_piBD({'X1', 'X2'}, {'Y2'}, Imitator.order)
    # print(policy)
    # policy = Imitator.get_seq_piBD({'X1', 'X2'}, {'Y1'}, Imitator.order)
    # print(policy)

    # Main source for imitation learning
    main()




    # Imitator3 = Agent(test_imi_3)
    # policy = Imitator3.get_seq_piBD({'X1', 'X2'}, {'Y'}, test_imi_3.order)
    # print(policy)

    #print(Imitator2.topological_order(True))
    #Imitator = Agent(test_g10)
    #print(Imitator.d_separation({'X'}, {'Y'}, {'W2'}))
    # print(Imitator.identify({'X'}, {'Y'}, {'Z'}))
    # print(Imitator.listIDspace({'X'}, {'Y'}))
    # Zs = Imitator.listIDspace({'X'}, {'Y'})
    # print(Imitator.listMINsep({'X'}, {'Y'}, Zs[0]))
    #print(Imitator.subgraph({'C', 'X', 'Y'}))
    # print(Imitator.get_c_factors('Y'))
    #print(Imitator.identify({'Y'}))

