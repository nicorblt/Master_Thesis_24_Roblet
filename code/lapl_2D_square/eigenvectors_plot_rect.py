import numpy as np
import matplotlib.pyplot as plt
import time
plt.rcParams.update({'font.family':'serif'})
plt.style.use("bmh")

def crea_dico_corres(N, dx, dy):
    """
    Create dictionnary to sort
    """
    classement_to_lm = {}
    lm_to_classement = {}
    set_couple = set()
    exact_eigval = []
    
    for l in range(1,N):
        for m in range(1,N):
            set_couple.add((l,m))

    for (l,m) in set_couple:
        eigval = 2*((dx**-2)*(1-np.cos(l*np.pi/N)) + (dy**-2)*(1-np.cos(m*np.pi/N)))
        exact_eigval.append([(l,m), eigval])
    exact_eigval.sort(key=lambda x: x[1])

    for ind, ((l,m), val) in enumerate(exact_eigval):
        classement_to_lm[f"{ind+1}"] = ((l,m), val)
        lm_to_classement[f"({l},{m})"] = (ind+1, val)

    return lm_to_classement, classement_to_lm

def numerical_eigenfunctions_rectangle(N, lms, a, b):
    """
    Numerical eigenfunction for the rectangle
    """
    #Initialisation of data
    time_start = time.time()
    dx, dy = a/N, b/N
    lm_to_classement, _ = crea_dico_corres(N, dx, dy)

    # construct M
    M = 2*(dx**-2 + dy**-2)*np.eye((N-1)**2) - (dx**-2)*(np.eye((N-1)**2, k=1) + np.eye((N-1)**2, k=-1)) - (dy**-2)*(np.eye((N-1)**2, k=N-1) + np.eye((N-1)**2, k=-(N-1)))
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
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 4), facecolor='white')
    fig.suptitle(rf'Numerical rectangle eigenfunctions for $N={N},~ a={a}$ and $b={b}$', fontsize=16)
    
    for ind, ((l,m), ax) in enumerate(zip(lms, axs.flat)):
        pos, _ = lm_to_classement[f"({l},{m})"]
        eigenmatrix = sorted_eigenvectors[:,pos-1].reshape(N-1, N-1, order="F") # conv 2
        im = ax.imshow(eigenmatrix.T, extent=[0, a, 0, b], origin = 'lower', cmap='inferno')
        ax.contour(eigenmatrix.T, extent=[0, a, 0, b], levels=[0], colors='red', origin = 'lower', linewidths = 1.2)
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
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