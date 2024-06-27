import numpy as np
import matplotlib.pyplot as plt
import time
plt.rcParams.update({'font.family':'serif'})
plt.style.use("bmh")

# some fct
def crea_dico_corres(N):
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

def convert_eigenvector(eigenfunction, N):
    eigenmatrix = eigenfunction.reshape(N-1, N-1, order="F")
    return eigenmatrix # conv 2


def numerical_eigenfunctions_square(N, lms):
    """
    Compute numerical solutions for square
    """
    #Initialisation of data
    time_start = time.time()
    lm_to_classement, _ = crea_dico_corres(N)

    # construct M
    M = (4*np.eye((N-1)**2) - np.eye((N-1)**2, k=1) - np.eye((N-1)**2, k=-1) - np.eye((N-1)**2, k=N-1) - np.eye((N-1)**2, k=-(N-1))) * (N**2)
    for j in range(1,N-1):
        M[j*(N-1)-1, j*(N-1)] = 0 #0 on the upper diagonal
        M[j*(N-1),j*(N-1)-1] = 0  #0 on the lower diagonal

    #consctruct eigenvlaues and vectors
    eigenvalues, eigenvectors = np.linalg.eigh(M)

    sorted_indices = np.argsort(eigenvalues)
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    #Plotting
    num_plots = len(lms)
    num_cols = min(4, num_plots)
    num_rows = (num_plots - 1) // num_cols + 1
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(14, 3*num_rows), facecolor='white')
    fig.suptitle(rf'Numerical square eigenfunctions for $N={N}$', fontsize=16)
    
    for ind, ((l,m), ax) in enumerate(zip(lms, axs.flat)):
        pos, _ = lm_to_classement[f"({l},{m})"]
        eigenmatrix = convert_eigenvector(sorted_eigenvectors[:,pos-1], N) # conv 2
        # To plot elt z_(i,j) in graph at (i*h, j*h) origin = lower + transpose due to matrix construction
        im = ax.imshow(eigenmatrix.T, extent=[0, 1, 0, 1], cmap='inferno', origin = 'lower', aspect='auto')
        ax.contour(eigenmatrix.T, extent=[0, 1, 0, 1], levels=[0], colors='red', origin = 'lower', linewidths = 1.2)
        plt.colorbar(im, ax=ax)
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(rf"$(l,m) = ({l},{m}),~\tilde \lambda = $" + "{:.2f}".format(sorted_eigenvalues[pos-1]), fontsize=16)
        ax.grid(False)

    # Close simulation
    print(f'Computation completed successfully in {time.time() - time_start}s')

    for ax in axs.flat[num_plots:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    assert False, "Please run the code from notebook main.ipynb"