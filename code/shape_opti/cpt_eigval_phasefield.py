import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.sparse import diags as sp_diags
from scipy.sparse.linalg import eigsh as sp_eigsh

plt.rcParams.update({'font.family':'serif'})
plt.style.use("bmh")

def diag_B(b, phase_field, eps):
    """
    Compute diag elements of B
    """
    def b_func(x):
        return b * (1 - x) / (2 * eps ** (4/3) )
    vect_phase = phase_field.flatten(order='F')
    return b_func(vect_phase)


def compute_eigenval_phasefield(phase_field, eps = 0.002, k = 10, b = 550, plot = True, js = [1]):
    """
    Compute and display the eigenvalues and eigenvectors of the Laplacian with Dirichlet boundary conditions within the provided domain.

    Args:
    domain (numpy.ndarray): Matrix composed of coeff between -1 and 1s defining the working domain.
    """
    time_start = time.time()
    phase_field = np.flip(phase_field.T, axis=1) # convert domain conv 1 to conv 2
    Nx, Ny = phase_field.shape # Initial domain is of size Ny x Nx.
    dx, dy = 1/(Nx-1), 1/(Ny-1)
    size = (Nx) * (Ny)

    #construct Left matrix
    main_diag = np.full(size, 2 / dx**2 + 2 / dy**2) + diag_B(b, phase_field, eps)
    off_diag_x = np.full(size - 1, -1 / dx**2)
    off_diag_y = np.full(size - (Nx), -1 / dy**2)

    # Create the diagonals
    diags = [main_diag, off_diag_x, off_diag_x, off_diag_y, off_diag_y]
    offsets = [0, 1, -1, Nx, -(Nx)]

    # Create the sparse matrix
    full_matrix = sp_diags(diags, offsets, shape=(size, size), format='csr')

    # print(f'Compute numerical eigenvalues', end= '\r')
    eigenvalues, eigenvectors = sp_eigsh(full_matrix, k=k, sigma=0, which = 'LM')

    eigenvectors = eigenvectors.T

    if plot:
        print(f'Creating plots displaying eigenvectors', end= '\r')
        num_plots = len(js)
        num_cols = min(5, num_plots)
        num_rows = (num_plots - 1) // num_cols + 1
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(14, 3*num_rows), facecolor='white')

        # To plot elt z_(i,j) in graph at (i*h, j*h) origin = 'upper', then nothing, due to matrix construction
        for j, ax in zip(js, axs.flat):
            eigen_matrix = eigenvectors[j-1].reshape((Nx, Ny), order='F').T
            im = ax.imshow(eigen_matrix, extent=[0, 1, 0, 1], cmap='inferno', origin = 'lower', aspect='auto')
            # ax.contour(eigen_matrix, extent=[0, 1, 0, 1], levels=[0], colors='black', origin = 'lower', linewidths = 5)
            ax.set_facecolor('white')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(rf"$j = {j},~\tilde \lambda = $" + "{:.2f}".format(eigenvalues[j-1]), fontsize=16)
            ax.grid(False)

        for ax in axs.flat[num_plots:]:
            ax.axis('off')

        print(f'\rComputation completed successfully in {time.time() - time_start}s.')
        plt.tight_layout()
        plt.show()
    
    return eigenvectors, eigenvalues

if __name__ == "__main__":
    assert False, "Please run the code from notebook main.ipynb"