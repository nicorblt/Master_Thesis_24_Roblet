import numpy as np
import matplotlib.pyplot as plt
import time
plt.rcParams.update({'font.family':'serif'})
plt.style.use("bmh")

def crea_dico_corres(N):
    """
    Create dictionnary to make correspond eigenvalues
    """
    classement_to_lm = {}
    lm_to_classement = {}
    set_couple = set()
    exact_eigval = []
    
    for l in range(1,N):
        for m in range(1,N):
            set_couple.add((l,m))

    for (l,m) in set_couple:
        eigval = (4*N**2)*((np.sin(l*np.pi/(2*N)))**2 + (np.sin(m*np.pi/(2*N)))**2)
        exact_eigval.append([(l,m), eigval])
    exact_eigval.sort(key=lambda x: x[1])

    for ind, ((l,m), val) in enumerate(exact_eigval):
        classement_to_lm[f"{ind+1}"] = ((l,m), val)
        lm_to_classement[f"({l},{m})"] = (ind+1, val)

    return lm_to_classement, classement_to_lm

def error_eigval_2D_square(Ns):
    """
    Plot evolution of error for numerical eigenvalue for the square 
    """
    time_start = time.time()

    num_plots = len(Ns)
    num_cols = min(3, num_plots)
    num_rows = (num_plots - 1) // num_cols + 1

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 3*num_rows), facecolor='white')
    fig.suptitle(r'Evolution of error $\lambda - \widetilde \lambda$ in function of $(l,m)$. $\widetilde \lambda$ computed numerically.', fontsize=16)

    for ind, (N, ax) in enumerate(zip(Ns, axs.flat)):
        print(f"\rIteration n°{ind+1}/{num_plots}: curently compute for N={N}.", end = '\r')
        # Construct M
        M = (4*np.eye((N-1)**2) - np.eye((N-1)**2, k=1) - np.eye((N-1)**2, k=-1) - np.eye((N-1)**2, k=N-1) - np.eye((N-1)**2, k=-(N-1))) * (N**2)
        for j in range(1,N-1):
            M[j*(N-1)-1, j*(N-1)] = 0 # 0 on the upper diagonal
            M[j*(N-1),j*(N-1)-1] = 0  # 0 on the lower diagonal

        # Numerical eigenvalues
        numerical_M_eigenval = np.linalg.eigvalsh(M)

        numerical_M_eigenval_matrix = np.zeros((N-1,N-1))
        _, classement_to_lm = crea_dico_corres(N)
        for index, eigval in enumerate(numerical_M_eigenval):
            (l,m), _ = classement_to_lm[str(index+1)]
            numerical_M_eigenval_matrix[l-1,m-1] = eigval

        #exact eigenvalue
        exact_eigenval = np.zeros((N-1,N-1))
        for l in range(1,N):
            for m in range(1,N):
                exact_eigenval[l-1,m-1] = (l**2 + m**2)*np.pi**2
        exact_eigenval.sort()

        # To plot elt z_(i,j) in graph at (i*h, j*h) origin = 'upper', then nothing due, to matrix construction
        im = ax.imshow(exact_eigenval - numerical_M_eigenval_matrix, cmap='inferno')
        plt.colorbar(im, ax=ax)
        ax.set_title(rf"$N = {N}$")
        ax.invert_yaxis()
        ax.grid(False)
        ax.set_xlabel('l')
        ax.set_ylabel('m')

    for ax in axs.flat[num_plots:]:
        ax.axis('off')
        
    print("\r\033[K", end='') 
    print(f'Computation completed successfully in {time.time() - time_start}s')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    assert False, "Please run the code from notebook main.ipynb"