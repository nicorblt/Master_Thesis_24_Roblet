import numpy as np
import matplotlib.pyplot as plt
import time
plt.rcParams.update({'font.family':'serif'})
plt.style.use("bmh")

def segment_eigenvectors(N, js):
    """
    Plot numerical and exact eigenvectors
    """
    #Initialisation of data
    time_start = time.time()

    x = np.linspace(0,1,N-1)
    M = (2*np.eye(N-1) - np.eye(N-1, k=1) - np.eye(N-1, k=-1) ) * (N**2)

    eigenvalues, eigenvectors = np.linalg.eigh(M)

    sorted_indices = np.argsort(eigenvalues)
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    #Plotting
    num_plots = len(js)
    num_cols = min(3, num_plots)
    num_rows = (num_plots - 1) // num_cols + 1
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 3*num_rows), facecolor='white')
    fig.suptitle(r'Some eigenfunctions, $\tilde{u}_n$ & $u_n$' +rf' for $N={N}$', fontsize=16)

    for ind, (j, ax) in enumerate(zip(js, axs.flat)):
        print(f"\r\tIteration n°{ind+1}/{num_plots}: curently compute for j={j}.", end = '\r')
        exact_sol = np.sin(np.pi* j * x) / np.linalg.norm(np.sin(np.pi* j * x))
        ax.plot(x, exact_sol, linewidth = 2.5, color = 'black', label=r"$u_j(x)$")
        ax.plot(x, np.sign(sorted_eigenvectors[1,j-1])*sorted_eigenvectors[:,j-1], color = 'y', linewidth = 2.5, linestyle='--', label=r"$\tilde u_j(x_i)$")
        ax.set_title(rf'$j = {j}$ \ ' + r'$e_{max}=$' + '{:.3e}'.format(np.max(np.abs(np.sign(sorted_eigenvectors[1,j-1])*sorted_eigenvectors[:,j-1] - exact_sol))))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        if not ind:
            ax.legend(fontsize='xx-large')

    # Close simulation
    print("\r\033[K", end='') # clean output line
    print(f'Computation completed successfully in {time.time() - time_start}s')

    for ax in axs.flat[num_plots:]:
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    assert False, "Please run the code from notebook main.ipynb"