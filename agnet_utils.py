import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import tqdm
from collections import defaultdict
from itertools import product



class StructuralCausalModel:
    def __init__(self, G):
        self.G = G
        self.__data = None
        self.__intervened_data = None
        self.__data_exo = None
        self.data_tmp = None
        self.ordered_topology = self.G.ordered_topology

    def sample_binary_data(self, sample_size: int, policy=None, intervened=False, Ys=None) -> pd.DataFrame:
        """
        return sample binary data based on random SCM according to the intervened param
        :param policy: policy for imitator
        :param sample_size: sample size
        :param intervened: sampling from intervened policy or original SCM function
        :param Ys: time series reward variables Ys
        :return: pd.DataFrame
        """

        # No noise
        F_noise_no = [0 for _ in range(len(self.G.Vs))]

        # Beta distribution for injecting noise on each node
        F_noise_beta = [np.random.beta(2, 5) for _ in range(len(self.G.Vs))]

        if intervened:
            bin_data = self.__sample_binary_data_intervened(F_noise=F_noise_no,
                                                            Denoising=True,
                                                            policy=policy,
                                                            sample_size=sample_size,
                                                            Ys=Ys
                                                            )
            bin_data = pd.DataFrame(bin_data)
            self.__intervened_data = bin_data
        else:
            bin_data = self.__sample_binary_data(F_noise=F_noise_no,
                                                 sample_size=sample_size)
            bin_data = pd.DataFrame(bin_data)
            self.__data = bin_data
            self.data_tmp = bin_data
        return bin_data

    def __sample_binary_data(self,
                             F_noise: list[float],
                             sample_size: int) -> dict:
        """ Sample binary data from the random SCM. Return np array """
        data = {ele: list() for ele in self.ordered_topology}  # data = {'node1':[], 'node2':[], ...}
        data_exo = {
            ele: np.random.uniform(0, 1) if np.random.rand() > 0.5 else 1
            for ele in self.ordered_topology
        }

        self.__data_exo = data_exo
        for _ in range(sample_size):
            for _, node in enumerate(self.ordered_topology):
                if not self.G.pa(node):
                    p = self.__data_exo[node]
                    data[node].append(np.random.choice([0, 1], p=[1-p, p])) # from exogenous
                else:
                    # get xor operated between each parent values
                    xor = 0
                    for i, pa in enumerate(self.G.pa(node)):
                        if not i:
                            xor = data[pa][-1]
                        if i:
                            xor = xor ^ data[pa][-1]

                    # current node's exogenous
                    p = self.__data_exo[node]
                    xor = xor ^ np.random.choice([0, 1], p=[1-p, p])
                    data[node].append(xor)
        return data

    def __sample_binary_data_intervened(self,
                                        F_noise: list[float],
                                        policy: dict[str, list[str]],
                                        sample_size: int,
                                        Denoising=False,
                                        Ys = None
                                        ) -> dict:
        """ Sample binary data from the SOFT intervened SCM using policy cpt. Return np array """

        # get cpt for soft intervened data
        cpt = dict()
        cpt_check = dict() # 공집합 policy
        for intervened_node, given_Z in policy.items():
            if given_Z:
                cpt[intervened_node] = self.__conditional_probability_table(X=intervened_node, givenZ=given_Z, Ys=Ys)
            else: # empty intervention
                cpt_check[intervened_node] = True

        data = {ele: list() for ele in self.ordered_topology}
        cnt = 0
        for _ in range(sample_size):
            # ordered_topology = self.G.topological_order()
            for _, node in enumerate(self.ordered_topology):
                # applying Seq.pi.BD policy P(X|Z), sampling data from the policy
                if node in cpt.keys():
                    Z_keys = list(policy[node])  # 다중 Z의 리스트 (['Z1', 'Z2', 'Z3'])

                    ########## will be deprecated.. ##########
                    for Z_key in Z_keys: # 비어있으면 안됨
                        if not data[Z_key]:  # 이번 policy node(X)의 (Z)가 비었으면 값 배정
                            p = self.__data_exo[Z_key]  # Z의 SCM의 exogenous parameter
                            data[Z_key].append(np.random.choice([0, 1], p=[1 - p, p]))
                    ######################################

                    try:
                        Z_keys_values = tuple(data[z][-1] for z in Z_keys)
                        if np.random.rand(1) <= cpt[node][Z_keys_values][1]:
                            # P(X=1 | Z1=z1, Z2=z2, Z3=z3)에 따라 샘플링
                            data[node].append(1)  # P(X=1|Z1=z1, Z2=z2, Z3=z3)
                        else:
                            data[node].append(0)  # P(X=0|Z1=z1, Z2=z2, Z3=z3)
                    except KeyError as e:
                        print("cpt[node]:", cpt[node])
                        raise KeyError(f"Invalid key encountered: {e}")
                    continue

                if node in cpt_check.keys(): # empty intervention 수행. P(Y|{}) == behavioral cloning
                    # 데이터에서 1의 비율 계산
                    data_node = self.data_tmp[node]
                    p = sum(value == 1 for value in data_node) / len(data_node)  # 1의 개수 / 전체 개수
                    # p = self.__data_exo[node]
                    data[node].append(np.random.choice([0, 1], p=[1 - p, p]))  # from exogenous
                    continue


                if not self.G.pa(node): # cpt에도 없고, paranet도 없고
                    p = self.__data_exo[node]
                    data[node].append(np.random.choice([0, 1], p=[1 - p, p]))  # from exogenous
                else: # parent node 존재시
                    # xor operation for each parent values
                    xor = 0
                    for i, pa in enumerate(self.G.pa(node)):
                        if not i:
                            xor = data[pa][-1]
                        if i:
                            xor = xor ^ data[pa][-1]

                    p = self.__data_exo[node]
                    xor = xor ^ np.random.choice([0, 1], p=[1-p, p])
                    data[node].append(xor)
        return data

    def __conditional_probability_table(self, X, givenZ, Ys):
        """get P(X|Z1, Z2, ..., Zn) conditional probability table from the sampling data"""
        X_Zs = {X} | givenZ
        data_X_Zs = self.data_tmp[list(X_Zs)].to_numpy()  # X와 Z 데이터를 NumPy 배열로 변환
        #data_Zs = self.data_tmp[list(givenZ)].to_numpy()

        # 모든 Z 조합 생성
        Z_cols = list(givenZ)
        unique_z_combinations = list(product([0, 1], repeat=len(Z_cols)))

        unique_x = [0, 1]  # X의 도메인
        all_x_z_combinations = [(*z, x) for z in unique_z_combinations for x in unique_x]

        # 분자: P(X, Z) 초기화
        dom_comb1 = {z_x: 0 for z_x in all_x_z_combinations}

        # 분모: P(Z) 초기화
        dom_comb2 = {tuple(z): 0 for z in unique_z_combinations}

        # 데이터 집계
        for data in data_X_Zs:
            z_comb = tuple(data[:-1])  # Z 값들
            x_val = data[-1]  # X 값
            dom_comb1[(*z_comb, x_val)] += 1
            dom_comb2[z_comb] += 1

        cpt = defaultdict(lambda: {0: 0.5, 1: 0.5})  # 기본값으로 P(X=0|Z)=0.5, P(X=1|Z)=0.5 설정
        for (x_z_key, count) in dom_comb1.items():
            z_key = x_z_key[:-1]  # Z의 조합만 추출 (맨뒤에있는 X 제외, e.g., (0,0,1) )
            cpt_in = {0:0.5, 1:0.5}
            prob = (count+0.0001) / (dom_comb2[z_key] + 0.0001)
            if x_z_key[-1] == 1: # x가 1이면,
                cpt_in[1] = prob
                cpt_in[0] = 1 - prob
            else:
                cpt_in[0] = prob
                cpt_in[1] = 1 - prob
            cpt[z_key] = cpt_in
        return cpt
