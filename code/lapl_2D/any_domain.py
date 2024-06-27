import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.sparse import diags as sp_diags
from scipy.sparse.linalg import eigsh as sp_eigsh

plt.rcParams.update({'font.family':'serif'})
plt.style.use("bmh")

def test_null_bounday(matrix):
    """
    Verify null boundary
    """
    if np.any(matrix[0]) or np.any(matrix[-1]):
        return False

    # Vérifier les bords gauche et droit
    if np.any(matrix[1:-1, 0]) or np.any(matrix[1:-1, -1]):
        return False

    return True

def convert_vect_to_matrix(vector, domain):
    """
    Construct matrix from vector according to domain, i.e. pput values if domain[i,j] !=0
    """
    matrix = np.zeros(domain.shape)
    counter_elt_vect = 0
    for j in range(domain.shape[1]):
        for i in range(domain.shape[0]):
            if domain[i,j] != 0:
                matrix[i,j] = vector[counter_elt_vect]
                counter_elt_vect += 1
    return matrix # conserv conv

def compute_eigenval(domain, a, b, k, plot = True, js = [1]):
    """
    Compute and display the eigenvalues and eigenvectors of the Laplacian with Dirichlet boundary conditions within the provided domain.

    Args:
    domain (numpy.ndarray): Matrix composed of 1s and 0s defining the working domain. The considered function will be zero where the domain is 0. To ensure Dirichlet boundary conditions, the matrix must have 0s at least at all its extremities.
    a (float): Width
    b (float): Height
    k (int): Number of eigenvalue to compute
    plot (bool): True to make plot
    js (numpy.ndarray): Eigenvalue to plot
    """
    assert test_null_bounday(domain), "Domain need to have at least all boundary coefficient to 0 for Dirichlet condition."
    time_start = time.time()

    print('\rConstruct matrix M', end = '\r')
    domain_adapted = np.flip(domain.T, axis=1) # convert domain conv 1 to conv 2
    Nx, Ny = domain_adapted.shape # Initial domain is of size Ny x Nx.
    dx, dy = a/(Nx-1), b/(Ny-1)
    size = (Nx - 1) * (Ny - 1)
    
    #construct M
    main_diag = np.full(size, 2 / dx**2 + 2 / dy**2)
    off_diag_x = np.full(size - 1, -1 / dx**2)
    off_diag_y = np.full(size - (Nx - 1), -1 / dy**2)

    # Create the diagonals
    diags = [main_diag, off_diag_x, off_diag_x, off_diag_y, off_diag_y]
    offsets = [0, 1, -1, Nx-1, -(Nx-1)]

    # Create the sparse matrix
    M = sp_diags(diags, offsets, shape=(size, size), format='csr')

    # Zero out elements at the boundary conditions
    for j in range(1, Ny-1):
        M[j*(Nx-1)-1, j*(Nx-1)] = 0
        M[j*(Nx-1), j*(Nx-1)-1] = 0

    # Identify indices to delete
    index_to_del = [ic + (Nx - 1) * (jc - 1) - 1 for jc in range(1, Ny) for ic in range(1, Nx) if domain_adapted[ic, jc] == 0]

    # Create a mask for the rows and columns to keep
    keep_mask = np.ones(size, dtype=bool)
    keep_mask[index_to_del] = False
    keep_indices = np.nonzero(keep_mask)[0]

    # Extract the submatrix for the kept indices
    M = M[keep_indices, :][:, keep_indices]

    print(f'Compute numerical eigenvalues', end= '\r')
    eigenvalues, eigenvectors = sp_eigsh(M, k=k, sigma=0, which = 'LM')

    eigenvectors = eigenvectors.T

    #plotting
    if plot:
        print(f'Creating plots displaying eigenvectors', end= '\r')
        num_plots = len(js) +1
        num_cols = min(5, num_plots)
        num_rows = (num_plots - 1) // num_cols + 1
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(14, 3*num_rows), facecolor='white')

        # To plot elt z_(i,j) in graph at (i*h, j*h) origin = 'upper', then nothing, due to matrix construction
        ax_domain = axs.flat[num_plots-1]
        im = ax_domain.imshow(domain, extent=[0, a, 0, b], cmap='viridis', aspect= 'auto')
        plt.colorbar(im, ax=ax_domain)
        ax_domain.set_xlabel('x')
        ax_domain.set_ylabel('y')
        ax_domain.grid(False)
        ax_domain.set_title(rf"$\Omega,~N_x =${Nx}$,~N_y =${Ny}", fontsize=16)

        for j, ax in zip(js, axs.flat[:-1]):
            eigen_matrix = convert_vect_to_matrix(eigenvectors[j-1], domain_adapted)  #conv 2
            masked_eigen = np.ma.masked_where(domain_adapted == False, eigen_matrix)
            im = ax.imshow(masked_eigen.T, extent=[0, a, 0, b], cmap='inferno', origin = 'lower', aspect='auto')
            ax.contour(masked_eigen.T, extent=[0, a, 0, b], levels=[0], colors='black', origin = 'lower', linewidths = 5)
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