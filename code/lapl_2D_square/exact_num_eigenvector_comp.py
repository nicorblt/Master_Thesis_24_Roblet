import numpy as np
import matplotlib.pyplot as plt
import time
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

def convert_eigenvector(eigenfunction, N):
    eigenmatrix = eigenfunction.reshape(N-1, N-1, order="F")
    return eigenmatrix # conv 2

def exact_num_eigenvector_comp(N, l, m):
    """
    Comparison between exact and numerical data
    """
    #Initialisation of data
    time_start = time.time()

    # Construct M
    M = (4*np.eye((N-1)**2) - np.eye((N-1)**2, k=1) - np.eye((N-1)**2, k=-1) - np.eye((N-1)**2, k=N-1) - np.eye((N-1)**2, k=-(N-1))) * (N**2)
    for j in range(1,N-1):
        M[j*(N-1)-1, j*(N-1)] = 0 # 0 on the upper diagonal
        M[j*(N-1),j*(N-1)-1] = 0  # 0 on the lower diagonal

    exact_lamb = (4*N**2)*((np.sin(l*np.pi/(2*N)))**2 + (np.sin(m*np.pi/(2*N)))**2)

    # Exact eigen vector
    my_eigenvector = np.zeros((N-1)**2)
    for i in range(1,N):
        for j in range(1,N):
            my_eigenvector[i+(N-1)*(j-1) -1] = np.sin(l*np.pi*i/N) * np.sin(m*np.pi*j/N)

    # Test exact eigenvalue
    prod_exact = M @ my_eigenvector

    fig, axs = plt.subplots(2, 3, figsize=(11, 6), facecolor='white')

    # To plot elt z_(i,j) in graph at (i*h, j*h) origin = lower + transpose due to matrix construction
    im = axs[0,0].imshow(convert_eigenvector(my_eigenvector, N).T, extent=[0, 1, 0, 1], origin = 'lower', cmap='inferno')
    plt.colorbar(im, ax=axs[0,0])
    axs[0,0].set_title(r"Theorical $\widetilde u = \widetilde u_{\text{th}}$")
    axs[0,0].grid(False)
    axs[0,0].set_xticks([])
    axs[0,0].set_yticks([])

    im = axs[0,1].imshow(convert_eigenvector(prod_exact, N).T, extent=[0, 1, 0, 1], origin = 'lower', cmap='inferno')
    plt.colorbar(im, ax=axs[0,1])
    axs[0,1].set_title(r"$M \widetilde u_{\text{th}}$")
    axs[0,1].grid(False)
    axs[0,1].set_xticks([])
    axs[0,1].set_yticks([])
    
    error = convert_eigenvector(my_eigenvector, N) * exact_lamb - convert_eigenvector(prod_exact, N)
    im = axs[0,2].imshow(error.T, extent=[0, 1, 0, 1], origin = 'lower', cmap='inferno')
    plt.colorbar(im, ax=axs[0,2])
    axs[0,2].set_title(r"$\widetilde \lambda_{\text{th}} \widetilde u_{\text{th}} - M\widetilde u_{\text{th}}$")
    axs[0,2].grid(False)
    axs[0,2].set_xticks([])
    axs[0,2].set_yticks([])

    # Test numerical eigenvalue 
    eigenvalues, eigenvectors = np.linalg.eigh(M)

    sorted_indices = np.argsort(eigenvalues)
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    lm_to_classement, _ = crea_dico_corres(N)
    classement, _ = lm_to_classement[f"({l},{m})"]
    computed_eigenvector = sorted_eigenvectors[:,classement-1]
    computed_eigenvalue = sorted_eigenvalues[classement-1]

    prod_num = M @ computed_eigenvector

    im = axs[1,0].imshow(convert_eigenvector(computed_eigenvector, N).T, extent=[0, 1, 0, 1], origin = 'lower', cmap='inferno')
    plt.colorbar(im, ax=axs[1,0])
    axs[1,0].set_title(r"Numerical $\widetilde u = \widetilde u_{\text{num}}$")
    axs[1,0].grid(False)
    axs[1,0].set_xticks([])
    axs[1,0].set_yticks([])


    im = axs[1,1].imshow(convert_eigenvector(prod_num, N).T, extent=[0, 1, 0, 1], origin = 'lower', cmap='inferno')
    plt.colorbar(im, ax=axs[1,1])
    axs[1,1].set_title(r"$M\widetilde u_{\text{num}}$")
    axs[1,1].grid(False)
    axs[1,1].set_xticks([])
    axs[1,1].set_yticks([])

    error = convert_eigenvector(computed_eigenvector, N) * computed_eigenvalue - convert_eigenvector(prod_num, N)
    im = axs[1,2].imshow(error.T, extent=[0, 1, 0, 1], origin = 'lower', cmap='inferno')
    plt.colorbar(im, ax=axs[1,2])
    axs[1,2].set_title(r"$\widetilde \lambda_{\text{num}} \widetilde u_{\text{num}} - M \widetilde u_{\text{num}}$")
    axs[1,2].grid(False)
    axs[1,2].set_xticks([])
    axs[1,2].set_yticks([])

    fig.suptitle(rf'Test, for (l,m)=({l},{m})' + r', if the numerically and exactly computed $\widetilde u_{l,m}$ are eigenvectors of $M$.', fontsize=16)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    assert False, "Please run the code from notebook main.ipynb"