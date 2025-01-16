import numpy as np
import time as tm
import h5py
import tensorly as tl

from tensor_train import tensor_train_decomposition, tensor_train_reconstruction
from cp_decomposition import CP_decomposition, CP_reconstruction
from tucker_decomposition import tucker_decomposition, tucker_reconstruction

# Specify the path to your HDF5 file
hdf5_file_path = '/home/jwang96/datasets/NVB_C009_l10n512_S12345T692_z42.hdf5'

def recon_error_eval(recon_tensor: np.array, tensor: np.array) -> float:
    rel_error = tl.norm(recon_tensor - tensor) / tl.norm(tensor)
    return rel_error

def read_hdf5(hdf5_file_path, data_field, chunk=True):
    # Open the HDF5 file
    with h5py.File(hdf5_file_path, 'r') as f:
        # List all groups in the file (optional, for exploration)
        print("Groups in the file:", list(f.keys()))
    
        # Assuming you know the dataset name, replace 'dataset_name' with your actual dataset name
        #dataset_name = 'native_fields/dark_matter_density'
        dataset_name = data_field 
    
        # Read the dataset into a NumPy array
        data_array = np.array(f[dataset_name])

    if chunk:
        chunk_size=(128, 256, 256)
        shape = data_array.shape
        chunks_z, chunks_y, chunks_x = (int(np.ceil(s / c)) for s, c in zip(shape, chunk_size))
        for z in range(chunks_z):
            z_start, z_end = z * chunk_size[0], min((z + 1) * chunk_size[0], shape[0])
            for y in range(chunks_y):
                y_start, y_end = y * chunk_size[1], min((y + 1) * chunk_size[1], shape[1])
                for x in range(chunks_x):
                    x_start, x_end = x * chunk_size[2], min((x + 1) * chunk_size[2], shape[2])
                    chunk_data = data_array[z_start:z_end, y_start:y_end, x_start:x_end]
    
    data_array = chunk_data
    # Now 'data_array' is a NumPy array containing your HDF5 data
    print("Shape of the data:", data_array.shape)
    print("Data type:", data_array.dtype)
    
    # Example: Print the first few elements of the array
    print("First few elements:", data_array.flatten()[:5])
    return data_array

if __name__ == "__main__":

    data_field = 'native_fields/dark_matter_density'
    data_array = read_hdf5(hdf5_file_path, data_field)

    # tensor train method
    #r_max = 200
    for r_max in [1]: # T, 2, 4, 8, 16, 32, 64, 128
        cutoff = 1e-10
        print(f"r_max:{r_max}")
        tt_factor = tensor_train_decomposition(data_array, r_max, cutoff)
        tt_recon = tensor_train_reconstruction(tt_factor)
        recon_error = recon_error_eval(tt_recon, data_array)
        print(f"TT Reconstruction error = {recon_error}\n")

    # cp method
    #rank = 200
    for rank in [1]: #, 2, 4, 8, 16, 32, 64, 128
        print(f"rank:{rank}")
        cp_result = CP_decomposition(data_array, rank)
        cp_recon = CP_reconstruction(cp_result)
        recon_error = recon_error_eval(cp_recon, data_array)
        print(f"CP Reconstruction error = {recon_error}\n")

    # tucker method
    #rank = [100, 100, 100]
    for rank in [[4, 4, 4]]: #, [8, 8, 8], [16, 16, 16], [32, 32, 32], [64, 64, 64], [128, 128, 128]
        print(f"rank:{rank}")
        cutoff = 1e-10
        core, factor, output_rank = tucker_decomposition(data_array, rank, cutoff)
        tucker_recon = tucker_reconstruction(core, factor)
        recon_error = recon_error_eval(tucker_recon, data_array)
        print(f"Tucker Reconstruction error = {recon_error}\n")
