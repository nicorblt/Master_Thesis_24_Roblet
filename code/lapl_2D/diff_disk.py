from lapl_2D.any_domain import compute_eigenval, convert_vect_to_matrix
from lapl_2D.exact_disk import exact_disk, create_domain_disk

import numpy as np
import matplotlib.pyplot as plt
import time
plt.rcParams.update({'font.family':'serif'})
plt.style.use("bmh")

def compute_erreur(exact_ev, num_ev, domain_adapted):
    """
    
    """
    mask_domain = domain_adapted < 1

    #convert and normalize num ev
    num_ev = convert_vect_to_matrix(num_ev, domain_adapted) # conv 2
    num_ev_masked = np.where(mask_domain, 0, num_ev)
    normal_num_ev_masked = num_ev_masked / np.linalg.norm(num_ev_masked)

    #convert and normalize exact ev
    exact_ev = exact_ev.T # conv 2
    exact_ev_masked = np.where(mask_domain, 0, exact_ev)
    normal_exact_ev_masked = exact_ev_masked / np.linalg.norm(exact_ev_masked)

    #same oscillation sign
    index_max = np.unravel_index(np.argmax(normal_exact_ev_masked), normal_exact_ev_masked.shape)
    sgn_change = np.sign(normal_exact_ev_masked[index_max]*normal_num_ev_masked[index_max])

    matrice_error = np.abs(normal_exact_ev_masked - normal_num_ev_masked*sgn_change) # conv 2

    err_rel = np.max(matrice_error)
    return matrice_error, err_rel

def adapt_order_list(domain_adapted, exact_eigen_func, num_eigen_func, num_eigenval):
    """ 
    Exchanges neighbouring values if this provides a better match with the exact data.
    It is assumed that values close to An, Bn = 1.0 and An, Bn = 0.1 are always quoted with a maximum rel error of 0.01.
    """
    k = exact_eigen_func.shape[0]

    new_num_eigen_func = np.zeros_like(num_eigen_func) #deep copy
    flag_previous_tested = False
    for index_curr in range(k-1):

        if flag_previous_tested:
            flag_previous_tested = False
            continue

        if (np.abs(num_eigenval[index_curr] - num_eigenval[index_curr+1]))/num_eigenval[index_curr] < 0.01: #eigenvalue assez proche pour justifier le calcul
            _, error_current = compute_erreur(exact_eigen_func[index_curr], num_eigen_func[index_curr], domain_adapted)
            _, error_next = compute_erreur(exact_eigen_func[index_curr], num_eigen_func[index_curr+1], domain_adapted)
            if error_next < error_current: # next better
                new_num_eigen_func[index_curr] = num_eigen_func[index_curr + 1]
                new_num_eigen_func[index_curr+1] = num_eigen_func[index_curr ]
                num_eigenval[index_curr], num_eigenval[index_curr + 1] = num_eigenval[index_curr+1], num_eigenval[index_curr]
            else: #current better
                new_num_eigen_func[index_curr] = num_eigen_func[index_curr]
                new_num_eigen_func[index_curr + 1] = num_eigen_func[index_curr + 1]
            flag_previous_tested = True
        else:
            new_num_eigen_func[index_curr] = num_eigen_func[index_curr]
            
    if np.all(new_num_eigen_func[-1] ==0):
        new_num_eigen_func[-1] = num_eigen_func[-1]
    return new_num_eigen_func, num_eigenval


def test_diff_disk(Nx, Ny, k, js=[1], plot = True):
    """
    Compute exact and numeric eigendata.
    Return, result of simulation and can plot particular error analysis.
    """
    #set variable
    time_start = time.time()
    a, b = 2, 2
    radius = 1 - max(a/Nx, b/Ny) * 3/2

    # create domain
    print("Create domain", end = '\r')
    domain = create_domain_disk(Nx, Ny, a, b, radius) # conv 1
    domain_adapted = np.flip(domain.T, axis=1) # conv 2

    # compute numerical solution
    time_start_cpt_num_ev = time.time()
    num_eigen_func, num_eigenval = compute_eigenval(domain, a, b, k, plot = False) # vector
    print(f'\r{num_eigen_func.shape[0]} numerical eigenvector calculated in {time.time() - time_start_cpt_num_ev}s.')

    # compute exact solution
    time_start_cpt_exact_ev = time.time()
    exact_eigen_func, exact_eigenval = exact_disk(Nx, Ny, k = k, plot = False) # conv 1', 2n**2 -n
    print(f'\r{exact_eigen_func.shape[0]} exact eigenvector calculated in {time.time() - time_start_cpt_exact_ev}s.')

    # sort to compare corresponding eigenvectors
    num_eigen_func, num_eigenval = adapt_order_list(domain_adapted, exact_eigen_func, num_eigen_func, num_eigenval)

    # plot    
    if plot:
        print(f'Creating plots displaying eigenvectors', end= '\r')
        fig = plt.figure(constrained_layout=True, figsize=(10, 3*len(js)))
        fig.suptitle('Comparison of results on the disk', fontsize=16)
        subfigs = fig.subfigures(nrows=len(js), ncols=1, facecolor='white')
        for j, subfig in zip(js, subfigs):
            axs = subfig.subplots(nrows=1, ncols=3)

            #compute eigenvalue data
            (n,k1), lambda_exact, An, Bn = exact_eigenval[j-1]
            lambda_num = num_eigenval[j-1]
            delta_lambda_r = (lambda_num - lambda_exact)/lambda_exact

            #compute eigenvector data
            exact_ev = exact_eigen_func[j-1].T # conv 2
            normal_exact_ev = exact_ev / np.linalg.norm(exact_ev)

            num_ev = convert_vect_to_matrix(num_eigen_func[j-1], domain_adapted) # conv 2
            normal_num_ev = num_ev / np.linalg.norm(num_ev)

            error_matrix, err_rel = compute_erreur(exact_eigen_func[j-1], num_eigen_func[j-1], domain_adapted)
            

            masked_exact = np.ma.masked_where(domain_adapted == False, normal_exact_ev)
            im = axs[0].imshow(masked_exact.T, extent=[0, a, 0, b], cmap='inferno', origin = 'lower', aspect = 'auto')
            axs[0].contour(masked_exact.T, extent=[0, a, 0, b], levels=[0], colors='red', origin = 'lower', linewidths = 1.2)
            axs[0].set_facecolor('white')
            plt.colorbar(im, ax=axs[0])
            axs[0].set_xlabel('x')
            axs[0].set_ylabel('y')
            axs[0].set_title(r'$u_{n,k},$' + rf'$~\lambda = {lambda_exact}$', fontsize = 12)
            axs[0].grid(False)

            masked_num = np.ma.masked_where(domain_adapted == False, normal_num_ev)
            im = axs[1].imshow(masked_num.T, extent=[0, a, 0, b], cmap='inferno', origin = 'lower', aspect = 'auto')
            axs[1].contour(masked_num.T, extent=[0, a, 0, b], levels=[0], colors='red', origin = 'lower', linewidths = 1.2)
            axs[1].set_facecolor('white')
            plt.colorbar(im, ax=axs[1])
            axs[1].set_xlabel('x')
            axs[1].set_ylabel('y')
            axs[1].set_title(r'$\widetilde u_{n,k},$' + rf'$~\widetilde \lambda = {lambda_num}$', fontsize = 12)
            axs[1].grid(False)

            masked_diff = np.ma.masked_where(domain_adapted == False, error_matrix)
            im = axs[2].imshow(masked_diff.T, extent=[0, a, 0, b], cmap='OrRd', origin = 'lower', aspect = 'auto')
            axs[2].contour(masked_diff.T, extent=[0, a, 0, b], levels=[0], colors='red', origin = 'lower', linewidths = 1.2)
            axs[2].set_facecolor('white')
            plt.colorbar(im, ax=axs[2])
            axs[2].set_xlabel('x')
            axs[2].set_ylabel('y')
            axs[2].set_title(r'Relative error $\Delta_r\widetilde{u}_{n,k}$', fontsize = 12)
            axs[2].grid(False)

            subfig.suptitle(rf'$N_x={Nx},~N_y={Ny},~j={j},~n={n},~k={k1},~A_n={An},~B_n={Bn}$' + r'$,~\Delta_r\widetilde\lambda_{n,k} =$' + "{:.2f}".format(delta_lambda_r)+ r'$,~\Delta_r\widetilde{u}_{n,k} =$' + "{:.2f}".format(err_rel), fontsize=14)
        print(f'\rComputation completed successfully in {time.time() - time_start}s.')
        plt.show()
    return num_eigen_func, num_eigenval, exact_eigen_func, exact_eigenval

if __name__ == "__main__":
    assert False, "Please run the code from notebook main.ipynb"