import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.colors import LinearSegmentedColormap
plt.rcParams.update({'font.family':'serif'})
plt.style.use("bmh")

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

def plot_rel_error_eigval(lms_eps, shift_plot):
    """
    Compute and plot relative error for square
    """
    #setup
    time_start = time.time()

    num_plots = len(lms_eps)
    num_cols = min(4, num_plots)
    num_rows = (num_plots - 1) // num_cols + 1

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(14, 3*num_rows), facecolor='white')
    fig.suptitle(r'Evolution of relative error in function of $(l,m)$.', fontsize=16)

    for ind, (((lo,mo), epsilon), extension_plot, ax) in enumerate(zip(lms_eps, shift_plot, axs.flat)):
        # Construct M
        tilde_N = int(np.pi*max([lo,mo])/np.sqrt(12*epsilon) + 1) #+1 to upper round
        print(f"\rIteration n°{ind+1}/{num_plots}. Estimated N = {tilde_N}", end = '\r')

        M = (4*np.eye((tilde_N-1)**2) - np.eye((tilde_N-1)**2, k=1) - np.eye((tilde_N-1)**2, k=-1) - np.eye((tilde_N-1)**2, k=tilde_N-1) - np.eye((tilde_N-1)**2, k=-(tilde_N-1))) * (tilde_N**2)
        for j in range(1,tilde_N-1):
            M[j*(tilde_N-1)-1, j*(tilde_N-1)] = 0 # 0 on the upper diagonal
            M[j*(tilde_N-1),j*(tilde_N-1)-1] = 0  # 0 on the lower diagonal

        # Numerical eigenvalues
        numerical_M_eigenval = np.linalg.eigvalsh(M)

        numerical_M_eigenval_matrix = np.zeros((tilde_N-1,tilde_N-1))
        _, classement_to_lm = crea_dico_corres(tilde_N)
        for index, eigval in enumerate(numerical_M_eigenval):
            (l,m), _ = classement_to_lm[str(index+1)]
            numerical_M_eigenval_matrix[l-1,m-1] = eigval

        # result_matrix
        result_matrix = np.zeros((tilde_N-1,tilde_N-1))
        for l in range(1,tilde_N):
            for m in range(1,tilde_N):
                lambda_lm = (l**2 + m**2)*np.pi**2
                result_matrix[l-1,m-1] = (lambda_lm - numerical_M_eigenval_matrix[l-1,m-1])/lambda_lm

        M_grid, L_grid = np.meshgrid(range(1, tilde_N), range(1, tilde_N))
        ind_x_min, ind_x_max = max(0, lo - extension_plot-1), min(tilde_N, lo + extension_plot)
        ind_y_min, ind_y_max = max(0, mo - extension_plot-1), min(tilde_N, mo + extension_plot)
        ind_min, ind_max = min(ind_x_min, ind_y_min), max(ind_x_max, ind_y_max)

        # To plot elt z_(i,j) in graph at (i*h, j*h) origin = 'upper', then nothing due, to matrix construction
        contour = ax.contourf(L_grid[ind_min:ind_max, ind_min:ind_max], M_grid[ind_min:ind_max, ind_min:ind_max], result_matrix[ind_min:ind_max, ind_min:ind_max], cmap='inferno')
        plt.colorbar(contour, ax=ax)
        contour_r = ax.contour(L_grid[ind_min:ind_max, ind_min:ind_max], M_grid[ind_min:ind_max, ind_min:ind_max], result_matrix[ind_min:ind_max, ind_min:ind_max], levels = [epsilon], colors='black')
        ax.clabel(contour_r, inline=True, fontsize=10)
        ax.scatter(lo, mo, color='green', s=100, marker='x', linewidth=2)

        ax.set_title(rf"$(l^*,m^*)=({lo},{mo}), \epsilon = {epsilon}.$" + r"$\widetilde N(\epsilon)$" + rf"$= {tilde_N}$")
        ax.invert_yaxis()
        ax.grid(False)
        ax.set_xlabel('l')
        ax.set_ylabel('m')
        ax.invert_yaxis()

    for ax in axs.flat[num_plots:]:
        ax.axis('off')

    print("\r\033[K", end='')
    print(f'Computation completed successfully in {time.time() - time_start}s')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    assert False, "Please run the code from notebook main.ipynb"