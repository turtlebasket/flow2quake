import numpy as np

def create_subsidence_matrix_fromGF(MESH, B, GF_vertical_displacement) :
    """Computes the matrix by which we multiply the pressure to find the subsidence
     this one already accounts for compressibility"""

    tolerance = 0.500 #dim change to km

    correspondance_matrix = np.zeros((len(B), len(MESH)))

    for ii in range(len(B)):
        #For each reservoir's block, we search the closest point of MESH
        indx = np.argmin(np.sqrt((MESH[:,0] - B['X'].iloc[ii]/1000)**2 + (MESH[:,1] - B['Y'].iloc[ii]/1000)**2)) #dim change
        val  = np.min(np.sqrt((MESH[:,0] - B['X'].iloc[ii]/1000)**2 + (MESH[:,1] - B['Y'].iloc[ii]/1000)**2)) #dim change

        if val < tolerance:
            correspondance_matrix[ii, indx] = 1.0


    G_new_mesh = np.matmul(GF_vertical_displacement, correspondance_matrix) * 1e6 #because the pressure is now in MPa

    np.save ('../Simulation_results/Greens_functions/GF_vert_disp_new_mesh.npy', G_new_mesh)
    
    return G_new_mesh