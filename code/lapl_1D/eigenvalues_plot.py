import numpy as np
import matplotlib.pyplot as plt
import time
plt.rcParams.update({'font.family':'serif'})
plt.style.use("bmh")

def segment_eigenvalues_error(Ns):
    """
    Plot error between numerical and exact eigenvalue
    """
    time_start = time.time()

    #set plot data
    fig = plt.figure(figsize=(20, 4))
    gs = fig.add_gridspec(1, 3)
    ax2 = fig.add_subplot(gs[0, 1:3])


    #Make computation
    for ind, N in enumerate(Ns):
        print(f"\rIteration n°{ind+1}/{len(Ns)}: curently compute for N={N}.", end = '\r')
        j_range = np.linspace(1, N-1, N-1)
        M = (2 * np.eye(N-1) - np.eye(N-1, k=1) - np.eye(N-1, k=-1)) * (N**2)

        numerical_M_eigenval = np.linalg.eigvalsh(M)
        exact_eigenval = j_range**2 * np.pi**2
        error = exact_eigenval - numerical_M_eigenval

        ax2.plot(j_range, error, linewidth = 2.5, label = rf"$N={N}$")

    # end plot
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')  # Désactiver les axes pour rendre le sous-plot totalement blanc
    ax2.set_xlabel("j")
    ax2.set_xlim((0,100))
    ax2.set_ylim((0,100))
    fig.suptitle(rf"Error of numerical w.r.t exact eigenvalue for various $N$", fontsize=18)
    fig.legend(loc='center', bbox_to_anchor=(0.2, 0.5), fontsize=18)

    print("\r\033[K", end='') 
    print(f'Computation completed successfully in {time.time() - time_start}s')

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    assert False, "Please run the code from notebook main.ipynb"