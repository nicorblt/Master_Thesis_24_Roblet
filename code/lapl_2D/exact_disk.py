# from lapl_2D.any_domain import compute_eigenval
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import time
plt.rcParams.update({'font.family':'serif'})
plt.style.use("bmh")

def create_domain_disk(Nx, Ny, a, b, radius):
    domain = np.zeros((Ny,Nx))
    centerx, centery = a / 2, b / 2
    for j in range(Ny):
        for i in range(Nx):
            norme = np.sqrt(((i + 1/2)*a/Nx - centerx)**2 + (b - (j + 1/2)*b/Ny - centery)**2)
            if norme <= radius:
                domain[j, i] = 1
    return domain

def exact_disk(Nx, Ny, k, js = [1], plot = True):
    """
    Compute exact data for the disk 
    """
    assert max(js)<=k , 'k too smal for js asked.'

    #set variable
    time_start = time.time()
    a, b = 2, 2
    radius = 1 - max(a/Nx, b/Ny) * 3/2

    #create useful domains
    print("Create domain", end = '\r')
    x_values = np.linspace(-a/2,a/2,Nx)
    y_values = np.linspace(-b/2,b/2,Ny)
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    r_grid = np.sqrt(x_grid**2 + y_grid**2)
    theta_grid = np.arctan2(y_grid, x_grid)

    #compute exact solutions
    print("Compute exact eigenfunctions", end = '\r')
    #eigval
    exact_eigenvalues = [] # elt_i = ((n,k), lambda, An, Bn)
    order = max((k,10))
    for n in range(order):
        roots_n = sc.special.jn_zeros(n,int(np.sqrt(order)))
        for k1, bessel_zero in enumerate(roots_n):
            exact_eigenvalues.append(((n,k1), (bessel_zero/(a/2))**2, 1, 0))
            if n != 0:
                exact_eigenvalues.append(((n,k1), (bessel_zero/(a/2))**2, 0, 1))
    exact_eigenvalues = sorted(exact_eigenvalues, key=lambda x: x[1])[:k]

    #eigvect   
    eigen_func = np.array([sc.special.jv(n, np.sqrt(lamb_nk) * r_grid) * (an * np.cos(n * theta_grid) + bn * np.sin(n * theta_grid)) for ((n, k1), lamb_nk, an, bn) in exact_eigenvalues])

    #plot
    if plot:
        print(f'Creating plots displaying eigenvectors', end= '\r')
        domain = create_domain_disk(Nx, Ny, a, b, radius) # domain in conv 1
        num_plots = len(js)
        num_cols = min(4, num_plots)
        num_rows = (num_plots - 1) // num_cols + 1
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(14, 3*num_rows), facecolor='white')

        for j, ax in zip(js, axs.flat):
            masked_eigen = np.ma.masked_where(np.flip(domain, axis = 0) == False, eigen_func[j-1]) # both in conv 1' so in 1'
            im = ax.imshow(masked_eigen, extent=[0, a, 0, b], cmap='inferno', origin = 'lower', aspect = 'auto')
            ax.contour(masked_eigen, extent=[0, a, 0, b], levels=[0], colors='red', origin = 'lower', linewidths = 1.2)
            ax.set_facecolor('white')
            plt.colorbar(im, ax=ax)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(rf"$j = {j},~\tilde \lambda = $" + "{:.2f}".format(exact_eigenvalues[j-1][1]) + rf",$~n={exact_eigenvalues[j-1][0][0]},~k={exact_eigenvalues[j-1][0][1]}, ~A_n ={exact_eigenvalues[j-1][2]}, ~B_n ={exact_eigenvalues[j-1][3]}$", fontsize=10)
            ax.grid(False)

        for ax in axs.flat[num_plots:]:
            ax.axis('off')

        # fig.suptitle(rf'Exact solution for the disk', fontsize=16)
        print(f'\rComputation completed successfully in {time.time() - time_start}s.')
        plt.tight_layout()
        plt.show()

    return eigen_func, exact_eigenvalues


if __name__ == "__main__":
    assert False, "Please run the code from notebook main.ipynb"