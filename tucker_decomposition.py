import numpy as np
import time as tm
import tensorly as tl

from numpy.random import rand
from numpy.random import seed
from scipy.linalg import svd
from tensorly.random import random_tucker
from tensorly.tenalg import mode_dot
from tensorly.tucker_tensor import tucker_to_tensor

'''
Tucker Tensor Decomposition
This Python script implements Tucker decomposition (also known as Higher-Order SVD), 
a tensor decomposition method that factorizes a tensor into a core tensor multiplied 
by factor matrices along each mode.

Some features:
    Implementation of Tucker decomposition using SVD-based approach
    Automatic rank determination based on singular value cutoff
    Reconstruction error evaluation
    
Three unit tests demonstrating different use cases:
    Synthetic tensor (3rd-order)
    Synthetic tensor (4th-order)
    Random tensor (3rd-order)
'''

def timer(func):
    def wrapper(*args, **kwargs):
        start_t = tm.time()
        result = func(*args, **kwargs)
        end_t = tm.time()
        print(f"{func} took {end_t - start_t} seconds.")
        return result
    return wrapper

# Evaluate the reconstruction error (relative error) of Tucker decomposition
def recon_error_eval(core: np.ndarray, factor: list[np.array], tensor: np.array) -> float:
    recon_tensor = tucker_to_tensor((core, factor))
    rel_error = tl.norm(recon_tensor - tensor) / tl.norm(tensor)
    return rel_error

@timer
def tucker_reconstruction(core: np.ndarray, factor: list[np.array]):
    return tucker_to_tensor((core, factor))

# Tucker decomposition (High-order SVD)
@timer
def tucker_decomposition(tensor: np.ndarray, rank: list[float], cutoff: float) -> np.ndarray:
    shape = tensor.shape  # Tensor shape [n1, n2, ..., nd]
    order = len(shape)    # order d
    
    # Compute the factor matrix A(1), A(2), ..., A(d) by SVD
    factor = []
    output_rank = []
    for n in range(order):
        # X(n): Matricization of the input tensor along different modes
        X = tl.base.unfold(tensor, n) 
        
        # Singular value decomposition and truncation
        U, S, Vh = svd(X)  # SVD of the unfolded matrix
        truncS = S[np.where(S > cutoff)] # Truncate singular values by cutoff
        trunc_rank = len(truncS)  # Truncation dimension (rank)
        
        # A(n) <- R_n leading left singular vectors of X(n)
        if (trunc_rank > rank[n]):
            A = U[:, 0:rank[n]]
            output_rank.append(rank[n])
        else:
            A = U[:, 0:trunc_rank]
            output_rank.append(trunc_rank)
        factor.append(A)
            
    # Compute the core tensor
    # core = tensor *_1 A(1)T *_2 A(2)T ... *_d A(d)T  
    core = mode_dot(tensor, factor[0], 0, transpose = True)
    for n in range(1, order):
        core = mode_dot(core, factor[n], n, transpose = True)

    return core, factor, output_rank

def unit_test_1():
    # Test of a synthetic tucker tensor
    # Test synthetic tensor settings
    print("Unit test 1 starts!")
    start_t = tm.time()
    shape = [20, 30, 20]  # Tensor shape n1 * n2 * n3
    rank = [15, 30, 18]   # Tucker rank 
    random_state = 10  # Random seed
    tucker_factor = random_tucker(shape, rank, random_state=random_state)
    tensor = tucker_to_tensor(tucker_factor)
    
    # Tucker decomposition
    input_rank = [20, 40, 20]
    cutoff = 1e-10
    core, factor, output_rank = tucker_decomposition(tensor, input_rank, cutoff)
    recon_error = recon_error_eval(core, factor, tensor)
    
    # Result display
    end_t = tm.time()
    print(f"Unit test 1 ends! It took {end_t - start_t} seconds")
    print(f"Tensor shape = {shape},\nTucker factor rank = {output_rank},")
    print(f"Reconstruction error = {recon_error}\n")
    return

def unit_test_2():
    # Test of a synthetic tucker tensor
    # Test synthetic tensor settings
    print("Unit test 2 starts!")
    start_t = tm.time()
    shape = [20, 50, 10, 30] # Tensor shape n1 * n2 * n3 * n4
    rank = [15, 30, 7, 25]   # Tucker rank 
    random_state = 20  # Random seed
    tucker_factor = random_tucker(shape, rank, random_state=random_state)
    tensor = tucker_to_tensor(tucker_factor)
    
    # Tucker decomposition
    input_rank = [50, 50, 50, 50]
    cutoff = 1e-10
    core, factor, output_rank = tucker_decomposition(tensor, input_rank, cutoff)
    recon_error = recon_error_eval(core, factor, tensor)
    
    # Result display
    end_t = tm.time()
    print(f"Unit test 2 ends! It took {end_t - start_t} seconds")
    print(f"Tensor shape = {shape},\nTucker factor rank = {output_rank},")
    print(f"Reconstruction error = {recon_error}\n")
    return

def unit_test_3():
    # Test of a completely random tensor
    print("Unit test 3 starts!")
    start_t = tm.time()
    shape = [50, 60, 100]  # Tensor shape n1 * n2 * n3
    seed(20)  # Random seed
    tensor = rand(shape[0], shape[1], shape[2])
      
    # Tucker decomposition
    input_rank = [100, 100, 100]
    cutoff = 1e-10
    core, factor, output_rank = tucker_decomposition(tensor, input_rank, cutoff)
    recon_error = recon_error_eval(core, factor, tensor)
    
    # Result display
    end_t = tm.time()
    print(f"Unit test 3 ends! It took {end_t - start_t} seconds")
    print(f"Tensor shape = {shape},\nTucker factor rank = {output_rank},")
    print(f"Reconstruction error = {recon_error}\n")
    return

#unit_test_1()
#unit_test_2()
#unit_test_3()

