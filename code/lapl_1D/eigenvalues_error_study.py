import numpy as np
import matplotlib.pyplot as plt
import time
plt.rcParams.update({'font.family':'serif'})
plt.style.use("bmh")

def segment_eigenvalues_error_study(js_eps, nb_point_N, range_test):
    """
    Deeper analysis of error evolution of eigenvalues
    """
    # setup simulation data
    js_eps = [(1,1)] + js_eps # for plot

    # setup plot
    num_plots = len(js_eps)
    num_cols = min(3, num_plots)
    num_rows = (num_plots - 1) // num_cols + 1

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 3*num_rows), facecolor='white')
    fig.suptitle(r'Relative error $e_{r,j}$ in function of $N$', fontsize=18)

    #Start computation
    for index, ((j, epsilon), ax) in enumerate(zip(js_eps, axs.flat)):
        assert j>=0 and epsilon >= 0 , f"Error, j or epsilon is negative at index {index}"
        if index == 0:
            ax.axis('off')
            continue

        estimated_N = int(np.pi*j/np.sqrt(12*epsilon))
        print(f'Start j = {j} & eps = {epsilon}, estimated N : {estimated_N}.')
        start_time = time.time()

        range_of_N = np.linspace(int(estimated_N*range_test[0]), int(estimated_N*range_test[1]), nb_point_N)

        upper_bound_relative_error = j**2 * np.pi**2 / (12 * range_of_N**2)
        error_num_jr = []

        for ind, N in enumerate(range_of_N):
            N = int(N)
            print(f"\r\tIteration n°{ind+1}/{nb_point_N}: N={N}.", end = '\r')

            j_range = np.linspace(1, N-1, N-1)
            M = (2 * np.eye(N-1) - np.eye(N-1, k=1) - np.eye(N-1, k=-1)) * (N**2)

            numerical_M_eigenval = np.linalg.eigvalsh(M)
            exact_eigenval = j_range**2 * np.pi**2

            error_num_jr.append((exact_eigenval[j-1] - numerical_M_eigenval[j-1])/exact_eigenval[j-1])

        ax.set_title(rf'$j={j},~\epsilon={epsilon}.~\tilde N_j(\epsilon)={estimated_N}$')
        ax.plot(range_of_N, error_num_jr, linewidth = 2.5, label = r"$e_{r,j}(N)$")
        ax.plot(range_of_N, upper_bound_relative_error, linewidth = 2.5, linestyle = '--', label = r"$N\mapsto j^2\pi^2/12N^2$")
        ax.axvline(estimated_N, color='r', linestyle='--', linewidth = 1,label=rf'$\tilde N_j(\epsilon)$')
        ax.axhline(epsilon, color='g', linestyle='--', linewidth = 1, label=r'$\epsilon$')
        ax.set_xlabel(r'$N$')
        if index == 1:
            fig.legend(loc='center', bbox_to_anchor=(0.18, 0.7), fontsize=18)

        computation_time = time.time() - start_time
        print("\r\033[K\r", end='') # clean print output
        print(f"\tComputation time: {computation_time:.7f}s.")

    for ax in axs.flat[num_plots:]:
        ax.axis('off')

    print("Computation completed successfully")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    assert False, "Please run the code from notebook main.ipynb"