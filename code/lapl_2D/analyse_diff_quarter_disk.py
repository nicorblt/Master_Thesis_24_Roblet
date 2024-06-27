from lapl_2D.diff_quarter_disk import test_diff_quarter_disk, compute_erreur
from lapl_2D.exact_quarter_disk import create_domain_quarter_disk


import numpy as np
import matplotlib.pyplot as plt
import time
plt.rcParams.update({'font.family':'serif'})
plt.style.use("bmh")

def compute_rel_err_eigenfunc(error_matrix, exact_eigen_func, domain):
    plt.imshow(domain)
    plt.plot()
    plt.imshow(error_matrix)
    plt.plot()
    plt.imshow(exact_eigen_func)
    plt.plot()
    mask_domain = domain < 1
    error_matrix_adapter = np.where(mask_domain, 0, error_matrix)
    exact_eigen_func_adapted = np.where(mask_domain, 0, exact_eigen_func)
    print(np.linalg.norm(error_matrix_adapter))
    return np.linalg.norm(error_matrix_adapter) / np.linalg.norm(exact_eigen_func_adapted)


def analyse_diff_quarter_disk(Nxys, k, k_to_plot = (0,0)):
    """ 
    Plot the error made by the numerical estimation of the eigenvalues and eigenvectors of the quarter circle.
    The main code for this function is in test_diff_quarter_disk.
    """
    #set variable
    if k_to_plot == (0,0) : k_to_plot = (0,k)
    fig, axs = plt.subplots(1, 2, figsize=(20, 5), facecolor='white')

    for Nx, Ny in Nxys:
        print(f"Start Nx, Ny = {Nx}, {Ny}")

        #create domain
        domain = create_domain_quarter_disk(Nx, Ny)
        domain_adapted = np.flip(domain.T, axis=1) # conv 2

        #get data of the problem
        num_eigen_func, num_eigenval, exact_eigen_func, exact_eigenval = test_diff_quarter_disk(Nx, Ny, k, plot = False)

        #compute error data to plot
        eigenval_error = []
        eigenvect_error = []
        for index in range(k):
            eigenval_error.append(np.abs(exact_eigenval[index][1]- num_eigenval[index])/exact_eigenval[index][1])
            _, err_rel = compute_erreur(exact_eigen_func[index], num_eigen_func[index], domain_adapted)
            eigenvect_error.append(err_rel)
        
        axs[0].plot(eigenvect_error[k_to_plot[0]:k_to_plot[1]], label = f"Nx, Ny = {Nx},{Ny}")
        axs[1].plot(eigenval_error[k_to_plot[0]:k_to_plot[1]], label = f"Nx, Ny = {Nx},{Ny}")

    axs[0].set_xlabel(r'$j$')
    axs[0].set_ylabel(r'error')
    axs[0].set_title(r'Relative eigenvectors error, $\Delta_r\widetilde{u}_{j}$', fontsize = 16)
    axs[0].legend(fontsize=14)
    axs[0].set_yscale('log')
    axs[1].set_xlabel(r'$j$')
    axs[1].set_ylim((0,0.03))
    axs[1].set_ylabel(r'error')
    axs[1].set_title(r'Relative eigenvalues error, $\Delta_r\widetilde\lambda_{j}$', fontsize = 16)
    fig.suptitle(r'Evolution of relative error in function of $j$.', fontsize=16)
    plt.show()


if __name__ == "__main__":
    assert False, "Please run the code from notebook main.ipynb"