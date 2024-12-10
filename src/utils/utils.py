from collections import defaultdict, deque
from typing import AbstractSet as ASet, TypeVar, Dict, FrozenSet, Iterable, Tuple

T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')

def pairs_to_dict_bi(bi_edges_pairs):
    tmp = defaultdict(set)
    for X, Y in bi_edges_pairs:
        tmp[X].add(Y)
        tmp[Y].add(X)
    # out = defaultdict(frozenset)
    # for k, vs in tmp.items():
    #     out[k] = frozenset(vs)
    return tmp

def pairs_to_dict(edge_pairs, reverse = False):
    '''
    convert the edge into dictionary
    :param edge_pairs: e.g.,[('X', 'Y'), ('Z', 'X'), ('Z', 'Y')]
    :param reverse: default False (True (X's children) or False (Y's parents))
    :return: dictionary
    '''
    tmp = defaultdict(list) #default setting as list
    if reverse: # _pa
        for X, Y in edge_pairs:
            tmp[Y].append(X) # save Y's parents
    else: # _ch
        for X, Y in edge_pairs:
            tmp[X].append(Y) # save X's parents
    out = defaultdict(frozenset)
    for k, vs in tmp.items():
        out[k] = frozenset(vs)
    return out

class visited_deque(deque):
    def __init__(self, iterable=frozenset()):
        super().__init__(added := set(iterable))
        self.check_visit = added

    # Override methods in deque
    def insert(self, i: int, v) -> None:
        if v not in self.check_visit:
            self.check_visit.add(v)
            super().insert(i, v)

    def append(self, v) -> None:
        if v not in self.check_visit:
            self.check_visit.add(v)
            super().append(v)

    def appendleft(self, x) -> None:
        if x not in self.check_visit:
            self.check_visit.add(x)
            super().appendleft(x)

    def extend(self, iterable: Iterable) -> None:
        difference = set(iterable) - self.check_visit
        super().extend(difference)
        self.check_visit.update(difference)

    def extendleft(self, iterable: Iterable) -> None:
        difference = set(iterable) - self.check_visit
        super().extendleft(difference)
        self.check_visit.update(difference)


def union(sets: Iterable[ASet[T]]) -> FrozenSet[T]:
    ret = set()
    for _ in sets:
        ret |= _
    return frozenset(ret)

