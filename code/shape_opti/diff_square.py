import numpy as np
import matplotlib.pyplot as plt
from shape_opti.cpt_eigval_phasefield import compute_eigenval_phasefield
from lapl_2D.any_domain import compute_eigenval

def test_error_square(N, liste_eps, k_max = 10, js = [j for j in range(1,11)], plot = False):
    #cpt eigval and eigvect with previous method
    domain = np.zeros((N,N+1)) #break symetry to get orthogonal eigenvectors
    domain[1:-1,1:-1] = 1
    sharp_eigenvectors, sharp_eigenvalues = compute_eigenval(domain, 1, 1, k_max, js = js, plot = plot)

    # compute eigval and eigvect for new method
    list_error_evect = []
    list_error_eval = []
    for eps in liste_eps:
        phase_field = - np.ones((N, N+1))
        phase_field[1:-1,1:-1] = 1
        eigenvectors, eigenvalues = compute_eigenval_phasefield(phase_field, eps = eps, k = k_max, js = js, plot = plot)
        error_ev = []
        for sharp_ev, ev in zip(sharp_eigenvectors, eigenvectors):
            mat_ev = (ev.reshape((N+1,N), order='F').T)[1:-1,1:-1]
            mat_sharp_ev = sharp_ev.reshape((N-1, N-2), order='F').T

            #same oscillation sign
            index_max = np.unravel_index(np.argmax(mat_ev), mat_ev.shape)
            sgn_change = np.sign(mat_ev[index_max]*mat_sharp_ev[index_max])
            matrice_error = np.abs(mat_ev - mat_sharp_ev*sgn_change) # conv 2

            error_ev.append(np.max(matrice_error))
        list_error_evect.append(error_ev)
        list_error_eval.append((np.array(sharp_eigenvalues) - np.array(eigenvalues))/np.array(eigenvalues))

    #plotting
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), facecolor='white')
    for eps, error_evect, error_eval in zip(liste_eps, list_error_evect, list_error_eval):
        axs[0].plot(error_evect, label = rf"$\epsilon =${eps}")
        axs[1].plot(error_eval, label = rf"$\epsilon =${eps}")

    axs[0].set_xlabel(r'$k$')
    axs[0].set_ylabel(r'error')
    axs[0].set_title(r'$\Delta \widetilde{u_k}$', fontsize = 12)
    axs[0].legend()
    axs[0].set_yscale('log')
    axs[1].set_xlabel(r'$k$')
    axs[1].set_title(r'$\Delta \widetilde{\lambda_k}$', fontsize = 12)
    axs[1].set_ylabel(r'error')
    # axs[1].legend()
    axs[1].set_yscale('log')
    fig.suptitle(fr'Evolution of relative error in function of $\epsilon$ for $N=${N}.', fontsize=16)
    plt.show()

if __name__ == "__main__":
    assert False, "Please run the code from notebook main.ipynb"