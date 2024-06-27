# from lapl_2D.any_domain import compute_eigenval
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import time
plt.rcParams.update({'font.family':'serif'})
plt.style.use("bmh")

def create_domain_quarter_disk(Nx, Ny):
    domain = np.zeros((Ny,Nx))
    sub_domain = np.zeros((Ny-2,Nx-2))
    for j in range(Ny-2):
        for i in range(Nx-2):
            if (i/(Nx-2)) ** 2 + (j/(Ny-2)) ** 2 <= 1:
                sub_domain[j,i] = 1
    domain[1:-1,1:-1] = sub_domain[::-1, :]
    return domain

def exact_quarter_disk(Nx, Ny, k, js = [1], plot = True):
    assert max(js)<=k , 'k too smal for js asked.'
    #set variable
    time_start = time.time()

    print("Create domain", end = '\r')
    x_values = np.linspace(0,1,Nx)
    y_values = np.linspace(0,1,Ny)
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    r_grid = np.sqrt(x_grid**2 + y_grid**2)
    theta_grid = np.arctan2(y_grid, x_grid)

    domain = create_domain_quarter_disk(Nx, Ny) # domain in conv 1

    print("Compute exact eigenfunctions", end = '\r')
    exact_eigenvalues = [] # elt_i = ((n,k), lambda)
    order = max((k,10))
    for n in range(1,order):
        roots_n = sc.special.jn_zeros(2*n, order)
        for k1, bessel_zero in enumerate(roots_n):
            exact_eigenvalues.append(((n,k1), (bessel_zero)**2))
    exact_eigenvalues = sorted(exact_eigenvalues, key=lambda x: x[1])[:k]

    eigen_func = np.array([sc.special.jv(2*n, np.sqrt(lamb_nk) * r_grid) * np.sin(2*n * theta_grid) for ((n, k1), lamb_nk) in exact_eigenvalues])

    #plotting
    if plot:
        print(f'Creating plots displaying eigenvectors', end= '\r')
        num_plots = len(js)
        num_cols = min(4, num_plots)
        num_rows = (num_plots - 1) // num_cols + 1
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(14, 3*num_rows), facecolor='white')

        for j, ax in zip(js, axs.flat):
            masked_eigen = np.ma.masked_where(np.flip(domain, axis = 0) == False, eigen_func[j-1]) # both in conv 1' so in 1'
            im = ax.imshow(masked_eigen, extent=[0, 1, 0, 1], cmap='inferno', origin = 'lower', aspect = 'auto')
            ax.contour(masked_eigen, extent=[0, 1, 0, 1], levels=[0], colors='red', origin = 'lower', linewidths = 1.2)
            ax.set_facecolor('white')
            plt.colorbar(im, ax=ax)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title(rf"$j = {j},~\tilde \lambda = $" + "{:.2f}".format(exact_eigenvalues[j-1][1]) + rf",$~n={exact_eigenvalues[j-1][0][0]},~k={exact_eigenvalues[j-1][0][1]}$", fontsize=10)
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