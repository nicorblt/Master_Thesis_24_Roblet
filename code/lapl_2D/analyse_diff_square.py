import numpy as np
import matplotlib.pyplot as plt
import time
plt.rcParams.update({'font.family':'serif'})
plt.style.use("bmh")

def average_list(list, ampl):
    len_list = len(list)
    averaged_list = []
    for i in range(len_list):
        start = max(0, i - ampl)
        end = min(len_list, i + ampl + 1)
        average = sum(list[start:end]) / (end - start)
        averaged_list.append(average)
    return averaged_list

def test_M_same_square(N, a, b):
    """ 
    Plot the error made by the numerical estimation of the eigenvalues and eigenvectors of the circle.
    The main code for this function is in test_diff_disk.
    """
    domain = np.zeros((N,N)) 
    domain[1:-1,1:-1] = 1

    # Initial contruction for square
    N_old = N-1
    dx, dy = a/N_old, b/N_old
    M_old = 2*(dx**-2 + dy**-2)*np.eye((N_old-1)**2) - (dx**-2)*(np.eye((N_old-1)**2, k=1) + np.eye((N_old-1)**2, k=-1)) - (dy**-2)*(np.eye((N_old-1)**2, k=N_old-1) + np.eye((N_old-1)**2, k=-(N_old-1)))
    for j in range(1,N_old-1):
        M_old[j*(N_old-1)-1, j*(N_old-1)] = 0 #0 on the upper diagonal
        M_old[j*(N_old-1),j*(N_old-1)-1] = 0  #0 on the lower diagonal
    
    # New version
    dx, dy = a/(N-1), b/(N-1)
    M = np.eye((N-1)*(N-1)) * (2/dx**2 + 2/dy**2)
    M += (- np.eye((N-1)*(N-1), k=1) - np.eye((N-1)*(N-1), k=-1)) * 1/dx**2
    M += (- np.eye((N-1)*(N-1), k=N-1) - np.eye((N-1)*(N-1), k=-(N-1))) * 1/dy**2
    for j in range(1,N-1):
        M[j*(N-1)-1, j*(N-1)] = 0 #0 on the upper diagonal
        M[j*(N-1), j*(N-1)-1] = 0  #0 on the lower diagonal

    # delete element as described
    index_to_del = [ic + (N - 1) * (jc - 1) - 1 for jc in range(1, N) for ic in range(1, N) if domain[ic, jc] == 0]
    M = np.delete(M, index_to_del, axis=0)
    M = np.delete(M, index_to_del, axis=1)
    
    print(M_old.shape, M.shape)
    print(np.all(M_old == M))
    fig, axs = plt.subplots(1, 3, figsize=(14, 5), facecolor='white')
    im = axs[0].imshow(M_old)
    axs[0].set_title("Old version")
    plt.colorbar(im, ax=axs[0])
    im=axs[1].imshow(M)
    axs[1].set_title("New version")
    plt.colorbar(im, ax=axs[1])
    im=axs[2].imshow(M - M_old)
    plt.colorbar(im, ax=axs[2])
    axs[2].set_title(f"Difference, max={np.max(np.abs(M - M_old))}")
    plt.show()


if __name__ == "__main__":
    assert False, "Please run the code from notebook main.ipynb"