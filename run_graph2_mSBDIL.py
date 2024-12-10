from src.agent.agent_mSBD import __inner
from src.agent.agent_mSBD import *

def main():
    test_msbd2 = nx_graph({'Z1', 'W1', 'W2', 'W3', 'W4', 'X1', 'X2', 'Y1',
                           'Z2', 'X3', 'X4', 'Y2',
                           'UX1Z1', 'UX3Z2'},
                          {('X1', 'X2'), ('X2', 'Y1'), ('Z1', 'Y1'),
                           ('X3', 'X4'), ('X4', 'Y2'), ('Z2', 'Y2'),
                           ('Y1', 'X3'), ('Z1', 'Y2'),
                           ('W1', 'X2'), ('W1', 'Y2'), # Ws attack
                           ('W2', 'Z1'), ('W2', 'Y2'),
                           ('W3', 'W1'), ('W3', 'X4'),
                           ('W4', 'W3'), ('W4', 'Y2'),
                           ('UX1Z1', 'X1'), ('UX1Z1', 'Z1'),  # UC
                           ('UX3Z2', 'X3'), ('UX3Z2', 'U2')},
                          {('X1', 'Z1'), ('X3', 'Z2')},
                          ordered_topology=('UX1Z1', 'UX3Z2','W4', 'W3',
                                            'W2', 'W1', 'Z1', 'X1',
                                            'X2', 'Y1', 'Z2', 'X3', 'X4', 'Y2'))

    def compute_naive_proj():
        naive_proj_result = __inner(test_msbd2, Ys={'Y1', 'Y2'})
        return naive_proj_result

    results_MSE = []
    for i in range(50):
        num_iterations = 100
        results = Parallel(n_jobs=-1)(
            delayed(compute_naive_proj)() for _ in tqdm(range(num_iterations))
        )

        dict_naive_Y1 = [res[0] for res in results]
        dict_naive_Y2 = [res[1] for res in results]
        dict_proj_Y1 = [res[2] for res in results]
        dict_proj_Y2 = [res[3] for res in results]

        MSE_naive_Y1 = np.array(dict_naive_Y1).mean(axis=0)
        MSE_naive_Y2 = np.array(dict_naive_Y2).mean(axis=0)
        MSE_naive = (MSE_naive_Y1 + MSE_naive_Y2)/2

        MSE_proj_Y1 = np.array(dict_proj_Y1).mean(axis=0)
        MSE_proj_Y2 = np.array(dict_proj_Y2).mean(axis=0)
        MSE_proj = (MSE_proj_Y1 + MSE_proj_Y2) / 2

        reduction_ratio = (MSE_naive - MSE_proj) / MSE_naive * 100
        print(f"MSE_naive : {MSE_naive}, MSE_proj : {MSE_proj}, diff of two : {MSE_naive - MSE_proj}")
        print(f"Reduction ratio: {reduction_ratio:.5f}%")
        results_MSE.append(reduction_ratio)

    print("================")
    print(np.array(results_MSE))
    print(np.array(results_MSE).max())

if __name__ == '__main__':
    main()