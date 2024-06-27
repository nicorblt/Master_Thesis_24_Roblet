import numpy as np
import matplotlib.pyplot as plt
from shape_opti.cpt_eigval_phasefield import compute_eigenval_phasefield
from tqdm.auto import tqdm

def cpt_v(phase_field):
    """
    v(phi)
    """
    sum = np.sum(phase_field + 1)
    vol = sum / (2*phase_field.size)
    return vol
    
def cpt_int_grad(phase_field):
    """
    First term of E
    """
    (Nx, Ny) = phase_field.shape
    grad_phase_field = np.zeros_like(phase_field)
    #interior
    for i in range(1,Nx-1):
        for j in range(1,Ny-1):
            coeff =  ((Nx / 2) * (phase_field[i,j-1] - phase_field[i,j+1]))**2 # 1
            coeff += ((Ny / 2) * (phase_field[i-1,j] - phase_field[i+1,j]))**2 # 3
            grad_phase_field[i,j] = coeff
    
    #exterior
    for j in range(1,Ny-1): #upper : i = 0
        coeff =  ((Nx / 2) * (phase_field[0,j-1] - phase_field[0,j+1]))**2 # 1
        coeff += ((Ny) * (phase_field[0,j] - phase_field[0+1,j]))**2 # 3
        grad_phase_field[0,j] = coeff

    for j in range(1,Ny-1): #lower : i = Nx-1
        coeff =  ((Nx / 2) * (phase_field[Nx-1,j-1] - phase_field[Nx-1,j+1]))**2 # 1
        coeff += ((Ny) * (phase_field[Nx-1-1,j] - phase_field[Nx-1,j]))**2 # 3
        grad_phase_field[Nx-1,j] = coeff
    
    for i in range(1,Nx-1): #left : j = 0
        coeff =  ((Nx) * (phase_field[i,0] - phase_field[i,0+1]))**2 # 1
        coeff += ((Ny / 2) * (phase_field[i-1,0] - phase_field[i+1,0]))**2 # 3
        grad_phase_field[i,0] = coeff
    
    for i in range(2,Nx-2): #right  : j = Ny-1
        coeff =  ((Nx) * (phase_field[i,Ny-1-1] - phase_field[i,Ny-1]))**2 # 1
        coeff += ((Ny / 2) * (phase_field[i-1,Ny-1] - phase_field[i+1,Ny-1]))**2 # 3
        grad_phase_field[i,Ny-1] = coeff

    #First top left corner : i = 0, j = 0
    coeff =  ((Nx) * (phase_field[0,0] - phase_field[0,0+1]))**2 # 1
    coeff += ((Ny) * (phase_field[0,0] - phase_field[0+1,0]))**2 # 3
    grad_phase_field[0,0] = coeff


    #First top right corner : i = 0, j = Ny-1
    coeff =  ((Nx) * (phase_field[0,Ny-1-1] - phase_field[0,Ny-1]))**2 # 1
    coeff += ((Ny) * (phase_field[0,Ny-1] - phase_field[0+1,Ny-1]))**2 # 3
    grad_phase_field[0,Ny-1] = coeff


    #First bottom left corner : i = Nx-1, j = 0
    coeff =  ((Nx) * (phase_field[Nx-1,0] - phase_field[Nx-1,0+1]))**2 # 1
    coeff += ((Ny) * (phase_field[Nx-1-1,0] - phase_field[Nx-1,0]))**2 # 3
    grad_phase_field[Nx-1,0] = coeff

    #First bottom right corner : i = Nx-1, j = Ny-1
    coeff =  ((Nx) * (phase_field[Nx-1,Ny-1-1] - phase_field[Nx-1,Ny-1]))**2 # 1
    coeff += ((Ny) * (phase_field[Nx-1-1,Ny-1] - phase_field[Nx-1,Ny-1]))**2 # 3
    grad_phase_field[Nx-1,Ny-1] = coeff

    return np.sum(grad_phase_field)/phase_field.size
    
def cpt_grad_int_norme_grad(phase_field):
    """
    grad of the first ter of the energy
    """
    (Nx, Ny) = phase_field.shape
    grad_int_norme_grad = np.zeros_like(phase_field)
    #interior
    for i in range(2,Nx-2):
        for j in range(2,Ny-2):
            coeff =  ((Nx**2) /2) * (phase_field[i,j] - phase_field[i,j-2]) # 1
            coeff += ((Nx**2) /2) * (phase_field[i,j] - phase_field[i,j+2]) # 2
            coeff += ((Ny**2) /2) * (phase_field[i,j] - phase_field[i+2,j]) # 3
            coeff += ((Ny**2) /2) * (phase_field[i,j] - phase_field[i-2,j]) # 4
            grad_int_norme_grad[i,j] = coeff

    #undercoat
    for j in range(2,Ny-2): #upper : i = 1
        coeff =  ((Nx**2) /2) * (phase_field[1,j] - phase_field[1,j-2]) # 1
        coeff += ((Nx**2) /2) * (phase_field[1,j] - phase_field[1,j+2]) # 2
        coeff += ((Ny**2) /2) * (phase_field[1,j] - phase_field[1+2,j]) # 3
        coeff += ((Ny**2) *2) * (phase_field[1,j] - phase_field[0,j])   
        grad_int_norme_grad[1,j] = coeff
    
    for j in range(2,Ny-2): #lower : i = Nx-2
        coeff =  ((Nx**2) /2) * (phase_field[Nx-2,j] - phase_field[Nx-2,j-2]) # 1
        coeff += ((Nx**2) /2) * (phase_field[Nx-2,j] - phase_field[Nx-2,j+2]) # 2
        coeff += ((Ny**2) *2) * (phase_field[Nx-2,j] - phase_field[Nx-1,j]) 
        coeff += ((Ny**2) /2) * (phase_field[Nx-2,j] - phase_field[Nx-2-2,j]) # 4
        grad_int_norme_grad[Nx-2,j] = coeff
    
    for i in range(2,Nx-2): #left : j = 1
        coeff =  ((Nx**2) *2) * (phase_field[i,1] - phase_field[i,0])
        coeff += ((Nx**2) /2) * (phase_field[i,1] - phase_field[i,1+2]) # 2
        coeff += ((Ny**2) /2) * (phase_field[i,1] - phase_field[i+2,1]) # 3
        coeff += ((Ny**2) /2) * (phase_field[i,1] - phase_field[i-2,1]) # 4
        grad_int_norme_grad[i,1] = coeff

    for i in range(2,Nx-2): #right  : j = Ny-2
        coeff =  ((Nx**2) /2) * (phase_field[i,Ny-2] - phase_field[i,Ny-2-2]) #1
        coeff += ((Nx**2) *2) * (phase_field[i,Ny-2] - phase_field[i,Ny-1]) 
        coeff += ((Ny**2) /2) * (phase_field[i,Ny-2] - phase_field[i+2,Ny-2]) # 3
        coeff += ((Ny**2) /2) * (phase_field[i,Ny-2] - phase_field[i-2,Ny-2]) # 4
        grad_int_norme_grad[i,Ny-2] = coeff
    
    #top left corner : i = 1, j = 1
    coeff =  ((Nx**2) *2) * (phase_field[1,1] - phase_field[1,0]) 
    coeff += ((Nx**2) /2) * (phase_field[1,1] - phase_field[1,1+2]) # 2
    coeff += ((Ny**2) /2) * (phase_field[1,1] - phase_field[1+2,1]) # 3
    coeff += ((Ny**2) *2) * (phase_field[1,1] - phase_field[0,1]) 
    grad_int_norme_grad[1,1] = coeff

    #top right corner : i = 1, j = Ny-2
    coeff =  ((Nx**2) /2) * (phase_field[1,Ny-2] - phase_field[1,Ny-2-2]) # 1
    coeff += ((Nx**2) *2) * (phase_field[1,Ny-2] - phase_field[1,Ny-1]) 
    coeff += ((Ny**2) /2) * (phase_field[1,Ny-2] - phase_field[1+2,Ny-2]) # 3
    coeff += ((Ny**2) *2) * (phase_field[1,Ny-2] - phase_field[0,Ny-2]) 
    grad_int_norme_grad[1,Ny-2] = coeff

    #bottom left corner : i = Nx-2, j = 1
    coeff =  ((Nx**2) *2) * (phase_field[Nx-2,1] - phase_field[Nx-2,0]) 
    coeff += ((Nx**2) /2) * (phase_field[Nx-2,1] - phase_field[Nx-2,1+2]) # 2
    coeff += ((Ny**2) *2) * (phase_field[Nx-2,1] - phase_field[Nx-1,1]) 
    coeff += ((Ny**2) /2) * (phase_field[Nx-2,1] - phase_field[Nx-2-2,1]) # 4
    grad_int_norme_grad[Nx-2,1] = coeff

    #bottom left corner : i = Nx-2, j = Ny-2
    coeff =  ((Nx**2) /2) * (phase_field[Nx-2,Ny-2] - phase_field[Nx-2,Ny-2-2]) # 1
    coeff += ((Nx**2) *2) * (phase_field[Nx-2,Ny-2] - phase_field[Nx-2,Ny-1]) 
    coeff += ((Ny**2) *2) * (phase_field[Nx-2,Ny-2] - phase_field[Nx-1,Ny-2]) 
    coeff += ((Ny**2) /2) * (phase_field[Nx-2,Ny-2] - phase_field[Nx-2-2,Ny-2]) # 4
    grad_int_norme_grad[Nx-2,Ny-2] = coeff

    #exterior
    for j in range(2,Ny-2): #upper : i = 0
        coeff =  ((Nx**2) /2) * (phase_field[0,j] - phase_field[0,j-2]) # 1
        coeff += ((Nx**2) /2) * (phase_field[0,j] - phase_field[0,j+2]) # 2
        coeff += ((Ny**2) /2) * (phase_field[0,j] - phase_field[0+2,j]) # 3
        coeff += ((Ny**2) *2) * (phase_field[0,j] - phase_field[1,j])
        grad_int_norme_grad[0,j] = coeff

    for j in range(2,Ny-2): #lower : i = Nx-1
        coeff =  ((Nx**2) /2) * (phase_field[Nx-1,j] - phase_field[Nx-1,j-2]) # 1
        coeff += ((Nx**2) /2) * (phase_field[Nx-1,j] - phase_field[Nx-1,j+2]) # 2
        coeff += ((Ny**2) *2) * (phase_field[Nx-1,j] - phase_field[Nx-2,j])
        coeff += ((Ny**2) /2) * (phase_field[Nx-1,j] - phase_field[Nx-1-2,j]) # 4
        grad_int_norme_grad[Nx-1,j] = coeff
    
    for i in range(2,Nx-2): #left : j = 0
        coeff =  ((Nx**2) *2) * (phase_field[i,0] - phase_field[i,1])
        coeff += ((Nx**2) /2) * (phase_field[i,0] - phase_field[i,0+2]) # 2
        coeff += ((Ny**2) /2) * (phase_field[i,0] - phase_field[i+2,0]) # 3
        coeff += ((Ny**2) /2) * (phase_field[i,0] - phase_field[i-2,0]) # 4
        grad_int_norme_grad[i,0] = coeff
    
    for i in range(2,Nx-2): #right  : j = Ny-1
        coeff =  ((Nx**2) /2) * (phase_field[i,Ny-1] - phase_field[i,Ny-1-2]) # 1
        coeff += ((Nx**2) *2) * (phase_field[i,Ny-1] - phase_field[i,Ny-2]) 
        coeff += ((Ny**2) /2) * (phase_field[i,Ny-1] - phase_field[i+2,Ny-1]) # 3
        coeff += ((Ny**2) /2) * (phase_field[i,Ny-1] - phase_field[i-2,Ny-1]) # 4
        grad_int_norme_grad[i,Ny-1] = coeff

    #First top left corner : i = 1, j = 0
    coeff =  ((Nx**2) *2) * (phase_field[1,0] - phase_field[1,1]) 
    coeff += ((Nx**2) /2) * (phase_field[1,0] - phase_field[1,0+2]) # 2
    coeff += ((Ny**2) /2) * (phase_field[1,0] - phase_field[1+2,0]) # 3
    coeff += ((Ny**2) *2) * (phase_field[1,0] - phase_field[0,0])
    grad_int_norme_grad[1,0] = coeff

    #Second top left corner : i = 1, j = 0
    coeff =  ((Nx**2) *2) * (phase_field[0,0] - phase_field[0,1])
    coeff += ((Nx**2) /2) * (phase_field[0,0] - phase_field[0,0+2]) # 2
    coeff += ((Ny**2) /2) * (phase_field[0,0] - phase_field[0+2,0]) # 3
    coeff += ((Ny**2) *2) * (phase_field[0,0] - phase_field[1,0])
    grad_int_norme_grad[0,0] = coeff

    #Third top left corner : i = 0, j = 1
    coeff =  ((Nx**2) *2) * (phase_field[0,1] - phase_field[0,0])
    coeff += ((Nx**2) /2) * (phase_field[0,1] - phase_field[0,1+2]) # 2
    coeff += ((Ny**2) /2) * (phase_field[0,1] - phase_field[0+2,1]) # 3
    coeff += ((Ny**2) *2) * (phase_field[0,1] - phase_field[1,1])
    grad_int_norme_grad[0,1] = coeff

    #First top right corner : i = 0, j = Ny-2
    coeff =  ((Nx**2) /2) * (phase_field[0,Ny-2] - phase_field[0,Ny-2-2]) # 1
    coeff += ((Nx**2) *2) * (phase_field[0,Ny-2] - phase_field[0,Ny-1])
    coeff += ((Ny**2) /2) * (phase_field[0,Ny-2] - phase_field[0+2,Ny-2]) # 3
    coeff += ((Ny**2) *2) * (phase_field[0,Ny-2] - phase_field[1,Ny-2]) 
    grad_int_norme_grad[0,Ny-2] = coeff

    #Second top right corner : i = 0, j = Ny-1
    coeff =  ((Nx**2) /2) * (phase_field[0,Ny-1] - phase_field[0,Ny-1-2]) # 1
    coeff += ((Nx**2) *2) * (phase_field[0,Ny-1] - phase_field[0,Ny-2])
    coeff += ((Ny**2) /2) * (phase_field[0,Ny-1] - phase_field[0+2,Ny-1]) # 3
    coeff += ((Ny**2) *2) * (phase_field[0,Ny-1] - phase_field[1,Ny-1])
    grad_int_norme_grad[0,Ny-1] = coeff

    #Third top right corner : i = 1, j = Ny-1
    coeff =  ((Nx**2) /2) * (phase_field[1,Ny-1] - phase_field[1,Ny-1-2]) # 1
    coeff += ((Nx**2) *2) * (phase_field[1,Ny-1] - phase_field[1,Ny-2])
    coeff += ((Ny**2) /2) * (phase_field[1,Ny-1] - phase_field[1+2,Ny-1]) # 3
    coeff += ((Ny**2) *2) * (phase_field[1,Ny-1] - phase_field[0,Ny-1])
    grad_int_norme_grad[1,Ny-1] = coeff

    #First bottom left corner : i = Nx-2, j = 0
    coeff =  ((Nx**2) *2) * (phase_field[Nx-2,0] - phase_field[Nx-2,1])
    coeff += ((Nx**2) /2) * (phase_field[Nx-2,0] - phase_field[Nx-2,0+2]) # 2
    coeff += ((Ny**2) *2) * (phase_field[Nx-2,0] - phase_field[Nx-1,0])
    coeff += ((Ny**2) /2) * (phase_field[Nx-2,0] - phase_field[Nx-2-2,0]) # 4
    grad_int_norme_grad[Nx-2,0] = coeff

    #Second bottom left corner : i = Nx-1, j = 0
    coeff =  ((Nx**2) *2) * (phase_field[Nx-1,0] - phase_field[Nx-1,1])
    coeff += ((Nx**2) /2) * (phase_field[Nx-1,0] - phase_field[Nx-1,0+2]) # 2
    coeff += ((Ny**2) *2) * (phase_field[Nx-1,0] - phase_field[Nx-2,0])
    coeff += ((Ny**2) /2) * (phase_field[Nx-1,0] - phase_field[Nx-1-2,0]) # 4
    grad_int_norme_grad[Nx-1,0] = coeff

    #Third bottom left corner : i = Nx-1, j = 1
    coeff =  ((Nx**2) *2) * (phase_field[Nx-1,1] - phase_field[Nx-1,0])
    coeff += ((Nx**2) /2) * (phase_field[Nx-1,1] - phase_field[Nx-1,1+2]) # 2
    coeff += ((Ny**2) *2) * (phase_field[Nx-1,1] - phase_field[Nx-2,1])
    coeff += ((Ny**2) /2) * (phase_field[Nx-1,1] - phase_field[Nx-1-2,1]) # 4
    grad_int_norme_grad[Nx-1,1] = coeff

    #First bottom right corner : i = Nx-2, j = Ny-1
    coeff =  ((Nx**2) /2) * (phase_field[Nx-2,Ny-1] - phase_field[Nx-2,Ny-1-2]) # 1
    coeff += ((Nx**2) *2) * (phase_field[Nx-2,Ny-1] - phase_field[Nx-2,Ny-2])
    coeff += ((Ny**2) *2) * (phase_field[Nx-2,Ny-1] - phase_field[Nx-1,Ny-1])
    coeff += ((Ny**2) /2) * (phase_field[Nx-2,Ny-1] - phase_field[Nx-2-2,Ny-1]) # 4
    grad_int_norme_grad[Nx-2,Ny-1] = coeff

    #Second bottom right corner : i = Nx-1, j = Ny-1
    coeff =  ((Nx**2) /2) * (phase_field[Nx-1,Ny-1] - phase_field[Nx-1,Ny-1-2]) # 1
    coeff += ((Nx**2) *2) * (phase_field[Nx-1,Ny-1] - phase_field[Nx-1,Ny-2])
    coeff += ((Ny**2) *2) * (phase_field[Nx-1,Ny-1] - phase_field[Nx-2,Ny-1]) 
    coeff += ((Ny**2) /2) * (phase_field[Nx-1,Ny-1] - phase_field[Nx-1-2,Ny-1]) # 4
    grad_int_norme_grad[Nx-1,Ny-1] = coeff

    #Third bottom right corner : i = Nx-1, j = Ny-2
    coeff =  ((Nx**2) /2) * (phase_field[Nx-1,Ny-2] - phase_field[Nx-1,Ny-2-2]) # 1
    coeff += ((Nx**2) *2) * (phase_field[Nx-1,Ny-2] - phase_field[Nx-1,Ny-1])
    coeff += ((Ny**2) *2) * (phase_field[Nx-1,Ny-2] - phase_field[Nx-2,Ny-2])
    coeff += ((Ny**2) /2) * (phase_field[Nx-1,Ny-2] - phase_field[Nx-1-2,Ny-2]) # 4
    grad_int_norme_grad[Nx-1,Ny-2] = coeff

    return grad_int_norme_grad / phase_field.size

def cpt_grad_quad_potential(phase_field):
    """
    grad of the second term of the energy
    """
    return -phase_field * (1-phase_field*phase_field) / phase_field.size 

def cpt_grad_Ginzburg_Landau_energy(phase_field, eps):
    """
    grad E_GL^\eps
    """
    grad_int_norme_grad = cpt_grad_int_norme_grad(phase_field)
    grad_quad_potential = cpt_grad_quad_potential(phase_field)
    return (eps/2) * grad_int_norme_grad + (1/eps) * grad_quad_potential

def cpt_grad_lambda(ev_to_opt, size_eigenspace, phase_field, eps, b):
    """
    grad lambda
    """
    k =  ev_to_opt -1 + size_eigenspace + 2 * (size_eigenspace == 0) 
    eigenvects, eigvals = compute_eigenval_phasefield(phase_field, eps = eps, b = b, k = k, plot = False)

    ev_to_opt_val = eigvals[ev_to_opt-1]
    eigenspace =  np.argsort(np.abs(np.array(eigvals) - ev_to_opt_val))[:size_eigenspace]
    eigenvect = np.sum(eigenvects[np.array(eigenspace)], axis=0) / len(eigenspace)

    norm_eigval = np.linalg.norm(eigenvect)**2
    
    grad_lambda = (eigenvect**2) * (-b / (2 * eps**(4/3)) / norm_eigval)
    
    return grad_lambda, ev_to_opt_val

def cpt_grad_V(phase_field, vol_init):
    """
    grad V(phi)
    """
    grad_g = np.ones_like(phase_field)/(2*phase_field.size)
    vol_curr = cpt_v(phase_field)
    grad_G = 2*(vol_curr - vol_init)*grad_g
    return grad_G

def cpt_grad_J(ev_to_opt, size_eigenspace, phase_field, alpha, beta, eps, b, vol_init):
    """
    Grad widetilde J
    """
    grad_lambda, prev_eigval = cpt_grad_lambda(ev_to_opt, size_eigenspace, phase_field, eps, b)
    grad_lambda = np.flipud(grad_lambda.reshape(phase_field.shape, order = 'F').T)
    grad_G = cpt_grad_V(phase_field, vol_init)
    grad_Ginzburg_Landau_energy = cpt_grad_Ginzburg_Landau_energy(phase_field, eps)
    return grad_lambda + alpha * grad_G + beta * grad_Ginzburg_Landau_energy, prev_eigval, grad_lambda, grad_G, grad_Ginzburg_Landau_energy

def opti_shape_ev(ev_to_opt, size_eigenspace, phase_field_0, alpha, beta, gamma, eps, b, nb_iter, plot = True, min_ev = 0):
    """
    Gradient descent for first eigenvalue

    Args:
    ev_to_opt (int): Index of eigenvalue to optimize (>=1)
    size_eigenspace (int): Size of associated eigenspace (>=1)
    phase_field_0 (numpy.ndarray): Matrix describing initial domain as phase-field (-1<=coeff<=1)
    alpha (float): Lagrange coefficient associated with volume constraint (>=0)
    beta (float): Lagrange coefficient associated with Ginzburg–Landau energy (>=0)
    gamma (float): Gradient descent step (>=0)
    eps (float): Phase-field coefficient (>0)
    b (float): coefficient associated to approximate eigenvalue problem (>0)
    nb_iter (int): number of descent gradient iteration (>=1)
    plot (bool): plot evolution of importantn quanityt as J, G and error on lambda
    min_ev (float): exact solution to the problem, mandatory if plot is True
    """
    # set value
    phase_field = np.clip(phase_field_0, -1, 1)
    vol_init = cpt_v(phase_field)
    list_phase_field = [phase_field]
    list_grad_J = []
    list_grad_lamb = []
    list_grad_V = []
    list_grad_E = []
    list_eigval = []

    for nb_iter in tqdm(range(0, nb_iter), desc='Progress', unit='step'):
        # Gradient descent iteration
        grad_J, prev_eigval, grad_lambda, grad_G, grad_E = cpt_grad_J(ev_to_opt, size_eigenspace, phase_field, alpha, beta, eps, b, vol_init)
        phase_field = phase_field - gamma * grad_J
        phase_field = np.clip(phase_field, -1, 1)

        # Save data
        list_phase_field.append(phase_field)
        list_grad_J.append(grad_J)
        list_eigval.append(prev_eigval)
        list_grad_lamb.append(grad_lambda)
        list_grad_V.append(grad_G)
        list_grad_E.append(grad_E)

    # complete eigenvalue list
    _, last_eigvals = compute_eigenval_phasefield(phase_field, eps = eps, b = b, k = ev_to_opt, plot = False)
    list_eigval.append(last_eigvals[ev_to_opt-1])

    list_V = np.array([(cpt_v(phase_field) - vol_init)**2 for phase_field in list_phase_field])
    list_E = np.array([(eps/2) * cpt_int_grad(phase_field) + (1/eps) * (np.sum(1 - phase_field*phase_field) / (2 * phase_field.size)) for phase_field in list_phase_field ])
    list_J = np.array(list_eigval) + alpha * list_V + beta * list_E
    
    bool_decreas = all(list_J[i] >= list_J[i + 1] for i in range(len(list_J) - 1))
    print(f"J decrease: {bool_decreas}")
    #plot

    if plot:
        fig, ax = plt.subplots(1, 3, figsize=(15, 5), facecolor='white')
        ax[0].plot(np.abs((np.array(list_eigval) - min_ev)/min_ev))
        ax[0].set_title(r"$\widetilde{\lambda}_k - \lambda^{min}_k$, $\lambda^{min}_k = $" + "{:.3f}".format(min_ev))
        ax[0].set_yscale('log')
        ax[1].plot([cpt_v(phase_field) - vol_init for phase_field in list_phase_field])
        ax[1].set_title(rf"$G(\phi) - G(\phi_0)$.")
        ax[1].set_yscale('log')
        ax[2].plot(list_J)
        ax[2].set_title(f"Evolution of J. Decrease: {bool_decreas}")
        ax[2].set_yscale('log')
        plt.show()

    return list_phase_field, list_grad_J, list_grad_lamb, list_grad_V, list_grad_E

if __name__ == "__main__":
    assert False, "Please run the code from notebook main.ipynb"