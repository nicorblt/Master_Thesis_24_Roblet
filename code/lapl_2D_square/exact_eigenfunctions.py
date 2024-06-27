import numpy as np
import matplotlib.pyplot as plt
import time
plt.rcParams.update({'font.family':'serif'})
plt.style.use("bmh")

def exact_eigenfunction_square(ind_to_plot, Nb_pt):
    time_start = time.time()

    # prepare plotting
    num_plots = len(ind_to_plot)
    num_cols = min(4, num_plots)
    num_rows = (num_plots - 1) // num_cols + 1
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(14, 3*num_rows+0.2), facecolor='white')
    fig.suptitle(r'Exact square eigenfunctions $(x,y)\mapsto \sin(lx\pi)\sin(my\pi)$', fontsize=16)

    # create mesh
    x = np.linspace(0, 1, Nb_pt)
    y = np.linspace(0, 1, Nb_pt)
    x, y = np.meshgrid(x, y)

    for (l, m), ax in zip(ind_to_plot, axs.flat):
        z = np.sin(l *np.pi * x) * np.sin(m*np.pi * y) # conv 1'
        # To plot elt z_(i,j) in graph at (i*h, j*h) origin = lower due to mesh construction
        im = ax.imshow(z, extent=[0, 1, 0, 1], cmap='inferno', origin = 'lower')
        ax.contour(z, extent=[0, 1, 0, 1], levels=[0], colors='red', origin = 'lower', linewidths = 1.2)
        plt.colorbar(im, ax=ax)
        ax.set_title(rf"$(l,m) = ({l},{m}),~ \lambda = $" + "{:.2f}".format((l**2 + m**2) * np.pi**2))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

    for ax in axs.flat[num_plots:]:
        ax.axis('off')

    print(f'Computation completed successfully in {time.time() - time_start}s')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    assert False, "Please run the code from notebook main.ipynb"