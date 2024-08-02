import numpy as np

def create_subsidence_matrix(MESH, RES, CM_grd, B, G_vertical_displacement, block_size,folder='reservoir_data') :
    """Computes the matrix by which we multiply the pressure to find the subsidence"""

    tolerance = block_size/1000 #0.500 #dim change

    CM = np.zeros(len(MESH))
    #Adjusts the compressibility values given in RES to the new MESH : 
    #For each point of MESH, searches the closest point of the mesh used in RES.
    #If the distance is inferior to a value given in tolerance, the corresponding value is added to CM

    for ii in range(len(MESH)):
        #Old version
        indx = np.argmin(np.sqrt((RES['X'].flatten()/1000 - MESH[ii,0])**2 + (RES['Y'].flatten()/1000 - MESH[ii,1])**2)) #dim change
        val  = np.min(np.sqrt((RES['X'].flatten()/1000 - MESH[ii,0])**2 + (RES['Y'].flatten()/1000 - MESH[ii,1])**2)) #dim change

        if val > tolerance:
            CM[ii] = 0
        else:
            CM[ii] = CM_grd.flatten()[indx-1]
    Cm_matrix = np.diag(CM/(5.5e-5))

    correspondance_matrix = np.zeros((len(B), len(MESH)))

    for ii in range(len(B)):
        #For each reservoir's block, we search the closest point of MESH

        indx = np.argmin(np.sqrt((MESH[:,0] - B['X'].iloc[ii]/1000)**2 + (MESH[:,1] - B['Y'].iloc[ii]/1000)**2)) #dim change
        val  = np.min(np.sqrt((MESH[:,0] - B['X'].iloc[ii]/1000)**2 + (MESH[:,1] - B['Y'].iloc[ii]/1000)**2)) #dim change

        if val < tolerance:
            correspondance_matrix[ii, indx] = 1.0


    G_new_mesh = np.matmul(G_vertical_displacement, correspondance_matrix)
    C_matrix = np.matmul(G_new_mesh, Cm_matrix) * 1e6 #1e6 because the pressure is now in MPa

    np.save (folder+'C_matrix.npy', C_matrix)
    
    return C_matrix