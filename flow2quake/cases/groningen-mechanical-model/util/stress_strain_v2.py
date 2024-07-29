import numpy as np
#import pylab as p
import math
import matplotlib.cm
import pandas as pd
from matplotlib.font_manager import FontProperties
import sys
import os
from scipy.interpolate import griddata
from glob import glob
from datetime import datetime,timedelta
import matplotlib.pylab as plt
from matplotlib.patches import Circle, Wedge, Polygon
import torch
from astropy.convolution import Gaussian2DKernel, convolve
import math 
from tqdm import tqdm
from time import time

pd.options.mode.chained_assignment = None

π = math.pi


def import_RES_from_files(RES_folder):
    import os
    # assign directory
    directory = RES_folder
    keys2=[]
    RES2 = {}
    # iterate over files in
    # that directory
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        keys2.append(filename[:-2])
        RES2[filename[:-2]]=np.load(open(f,'rb'),allow_pickle=True)
    return RES2

def create_groningen_object(RES_folder) :
    print ('Creating Groningen object...', end = " ")
    grg = Groningen(RES_folder)

    BLOCKS            = grg.BLOCKS
    POINTS            = BLOCKS[['X','Y','Z']]
    POINTS['X']       = BLOCKS['X'] + BLOCKS['wX']/2
    POINTS['Y']       = BLOCKS['Y'] + BLOCKS['wY']/2
    POINTS['Z']       = grg.BLOCKS['Z']-10

    grg.StrainVolume = StrainVolume()
    grg.StrainVolume.LoadBlocks(BLOCKS)
    grg.StrainVolume.LoadPoints(POINTS)

    print ('[DONE]')

    return grg

def create_groningen_object_depth_no_GF(RES_folder,GF_folder,depth=-10) :
    print ('Creating Groningen object...', end = " ")
    grg = Groningen(RES_folder)

    BLOCKS            = grg.BLOCKS
    POINTS            = BLOCKS[['X','Y','Z']]
    POINTS['X']       = BLOCKS['X'] + BLOCKS['wX']/2
    POINTS['Y']       = BLOCKS['Y'] + BLOCKS['wY']/2
    POINTS['Z']       = grg.BLOCKS['Z']+depth

    grg.StrainVolume = StrainVolume()
    grg.StrainVolume.LoadBlocks(BLOCKS)
    grg.StrainVolume.LoadPoints(POINTS)

    print ('[DONE]')

    return grg

def create_groningen_object_depth_with_GF(RES_folder,GF_folder,depth=-10) :
    print ('Creating Groningen object...', end = " ")
    grg = Groningen(RES_folder)

    BLOCKS            = grg.BLOCKS
    POINTS            = BLOCKS[['X','Y','Z']]
    POINTS['X']       = BLOCKS['X'] + BLOCKS['wX']/2
    POINTS['Y']       = BLOCKS['Y'] + BLOCKS['wY']/2
    POINTS['Z']       = grg.BLOCKS['Z']+depth

    grg.StrainVolume = StrainVolume()
    grg.StrainVolume.LoadBlocks(BLOCKS)
    grg.StrainVolume.LoadPoints(POINTS)

    print('Groningen Object Created')
    print('...')
    print('Computing Greens function')

    grg.StrainVolume.compute(GreenFunction=True, dv='cpu')
    torch.save (grg.StrainVolume.Disp, GF_folder+f'G_matrix_disp_{depth}m.pt')
    torch.save (grg.StrainVolume.σ, GF_folder+f'G_matrix_sigma_{depth}m.pt')
    print(f'Greens Functions computed for depth = {-depth} above reservoir.')
    print ('[DONE]')

    return grg

def compute_surfaceGF_z(RES_folder,GF_folder, depth_below_surface=1) :
    print(f'Computing surface displacement Greens function at {depth_below_surface}m below surface')

    grg = Groningen(RES_folder)
    BLOCKS            = grg.BLOCKS
    POINTS            = BLOCKS[['X','Y','Z']]
    POINTS['X']       = BLOCKS['X'] + BLOCKS['wX']/2
    POINTS['Y']       = BLOCKS['Y'] + BLOCKS['wY']/2
    POINTS['Z']       = depth_below_surface

    grg.StrainVolume = StrainVolume()
    grg.StrainVolume.LoadBlocks(BLOCKS)
    grg.StrainVolume.LoadPoints(POINTS)

    grg.StrainVolume.compute(GreenFunction=True, dv='cpu')
    Surf_GF = grg.StrainVolume.Disp.numpy()[:,:,-1]
    np.save (GF_folder+f'SurfaceGreensFunction_vertical_displacement_only_{depth_below_surface}m.npy',Surf_GF)
    print(f'Surface Greens Functions computed at {depth_below_surface}m depth.')
    print(f'Saved to numpy array for the vertical displacement only')
    print ('[DONE]')

    return grg

def compute_green_functions(groningen_object,GF_folder) :
    """Computes and saves the matrix we use to compute the stress tensor"""
    grg = groningen_object
    print ('Computing GreenFunction')
    grg.StrainVolume.compute(GreenFunction=True, dv='cpu')
    torch.save (grg.StrainVolume.Disp, GF_folder+f'G_matrix_disp.pt')
    torch.save (grg.StrainVolume.σ, GF_folder+f'G_matrix_sigma.pt')


def compute_coulomb_stress(groningen_object, RESdata_folder, MESH, pressures_list,GF_folder, activate_cuda=False) :
    """Compute the coulomb stress using the matrix computed with compute_green_functions()"""
    
    grg = groningen_object

    print ('Loading G matrix...')
    grg.StrainVolume.σ = torch.load(GF_folder+f'G_matrix_sigma_{depth}m.pt')
    grg.StrainVolume.Disp = torch.load(GF_folder+f'G_matrix_disp_{depth}m.pt')
    print ('G matrix loaded')

    RES = import_RES_from_files(RESdata_folder)
    correspondance_matrix = np.zeros((RES['X'].size, len(MESH)))
    tolerance=0.5

    for ii in range(RES['X'].size):
        #For each reservoir's block, we search the closest point of MESH
        indx = np.argmin(np.sqrt((MESH[:,0] - RES['X'].flatten()[ii]/1000)**2 + (MESH[:,1] - RES['Y'].flatten()[ii]/1000)**2)) #dim change
        val  = np.min(np.sqrt((MESH[:,0] - RES['X'].flatten()[ii]/1000)**2 + (MESH[:,1] - RES['Y'].flatten()[ii]/1000)**2)) #dim change
        if val < tolerance:
            correspondance_matrix[ii, indx] = 1.0

    pres_res_mesh = np.zeros((RES['X'].shape[0], RES['X'].shape[1], len(pressures_list)))
    for i in range (pres_res_mesh.shape[2]) :
        pres_res_mesh[:, :, i] = np.matmul(correspondance_matrix, (pressures_list[i]-pressures_list[0])*1e6).reshape(RES['X'].shape)

    values = grg.DepthSlice_FullTEvolution_V2('./', pres_res_mesh, depth = depth, activate_cuda=activate_cuda)
    max_coulomb_stress = values['GridMCL']


    return max_coulomb_stress

def compute_coulomb_stress_depth(groningen_object, RESdata_folder, MESH, pressures_list,GF_folder,activate_cuda=False, depth = 10) :
    """Compute the coulomb stress using the matrix computed with compute_green_functions()"""
    
    grg = groningen_object

    print ('Loading G matrix...')
    grg.StrainVolume.σ = torch.load(GF_folder+f'G_matrix_sigma_{depth}m.pt')
    grg.StrainVolume.Disp = torch.load(GF_folder+f'G_matrix_disp_{depth}m.pt')
    print ('G matrix loaded')

    RES = import_RES_from_files(RESdata_folder)
    correspondance_matrix = np.zeros((RES['X'].size, len(MESH)))
    tolerance=0.5

    for ii in range(RES['X'].size):
        #For each reservoir's block, we search the closest point of MESH
        indx = np.argmin(np.sqrt((MESH[:,0] - RES['X'].flatten()[ii]/1000)**2 + (MESH[:,1] - RES['Y'].flatten()[ii]/1000)**2)) #dim change
        val  = np.min(np.sqrt((MESH[:,0] - RES['X'].flatten()[ii]/1000)**2 + (MESH[:,1] - RES['Y'].flatten()[ii]/1000)**2)) #dim change
        if val < tolerance:
            correspondance_matrix[ii, indx] = 1.0

    pres_res_mesh = np.zeros((RES['X'].shape[0], RES['X'].shape[1], len(pressures_list)))
    for i in range (pres_res_mesh.shape[2]) :
        pres_res_mesh[:, :, i] = np.matmul(correspondance_matrix, (pressures_list[i]-pressures_list[0])*1e6).reshape(RES['X'].shape)

    values = grg.DepthSlice_FullTEvolution_V2('./', pres_res_mesh, depth = depth, activate_cuda=activate_cuda)
    max_coulomb_stress = values['GridMCL']

    return max_coulomb_stress


class StrainVolume:
    def __init__(self):
        """
            Initiate stage
        """


    def _elastic_constant(self):
        """
            Determining the elastic constants

        """
        self.G  = self.G*(self.Cm*0 +1)
        self.E = 2*self.G*(1. + self.ν)                           # Youngs Modulus
        self.K = (2.*self.G*(1+self.ν))/(3.*(1-2.*self.ν))        # Bulk   Modulus

        self.Fσ = -(self.α*self.Cm*self.G)/(2.*π) # Stress Factor
        self.Fu = -(self.α*self.Cm)/(4.*π)        # Displacement Factor


    def _BoundaryPerturbation(self,BoundaryTolerance,tol=1e-3):
        '''
            ADD PERTURBATION IF YOU ARE ON THE BOUNDARY !! 
        '''
        xmin = self.vertices[:,0,0]; xmax = self.vertices[:,1,0];
        ymin = self.vertices[:,0,1]; ymax = self.vertices[:,2,1];
        zmax = -self.vertices[:,0,2]; zmin = -self.vertices[:,4,2];
        InVol       = torch.zeros(self.obspoints.shape[0],dtype=torch.bool)
        InVol_value = torch.zeros(self.obspoints.shape[0])
        
        for jj in tqdm(range(self.vertices.shape[0]), desc='Observation Points Boundary Check'):
            
            # -------- Perturbing Points on Volume Boundary -------
            #XminIndex :  obspoints where x=xmin[jj], y in [ymin[jj] ; ymax[jj]] and z in [zmin[jj] ; zmax[jj]]
            #--> selection of points in the xmin face of the block
            XminIndex = torch.where((xmin[jj] == self.obspoints[:,0]) & (ymin[jj] <= self.obspoints[:,1]) & (self.obspoints[:,1] <= ymax[jj]) & (zmax[jj] <= self.obspoints[:,2]) & (self.obspoints[:,2]<= zmin[jj]))
            XmaxIndex = torch.where((xmax[jj] == self.obspoints[:,0]) & (ymin[jj] <= self.obspoints[:,1]) & (self.obspoints[:,1] <= ymax[jj]) &  (zmax[jj] <= self.obspoints[:,2]) & (self.obspoints[:,2]<= zmin[jj]))
            YminIndex = torch.where((xmin[jj] <= self.obspoints[:,0]) & (self.obspoints[:,0] <= xmax[jj]) & (ymin[jj] == self.obspoints[:,1]) &  (zmax[jj] <= self.obspoints[:,2]) & (self.obspoints[:,2]<= zmin[jj]))
            YmaxIndex = torch.where((xmin[jj] <= self.obspoints[:,0]) & (self.obspoints[:,0] <= xmax[jj]) & (ymax[jj] == self.obspoints[:,1]) &  (zmax[jj] <= self.obspoints[:,2]) & (self.obspoints[:,2]<= zmin[jj]))
            ZminIndex = torch.where((xmin[jj] <= self.obspoints[:,0]) & (self.obspoints[:,0] <= xmax[jj]) & (ymin[jj] <= self.obspoints[:,1]) & (self.obspoints[:,1] <= ymax[jj]) &  (zmin[jj] == self.obspoints[:,2]))
            ZmaxIndex = torch.where((xmin[jj] <= self.obspoints[:,0]) & (self.obspoints[:,0] <= xmax[jj]) & (ymin[jj] <= self.obspoints[:,1]) & (self.obspoints[:,1] <= ymax[jj]) &  (zmax[jj] == self.obspoints[:,2]))
                
            self.XminIndex = XminIndex

            try: 
                self.obspoints[XminIndex,0] = self.obspoints[XminIndex,0] + tol 
            except: 
                pass
            try: 
                self.obspoints[XmaxIndex,0] = self.obspoints[XmaxIndex,0] + tol 
            except: 
                pass
            try: 
                self.obspoints[YminIndex,1] = self.obspoints[YminIndex,1] + tol 
            except: 
                pass
            try: 
                self.obspoints[YmaxIndex,1] = self.obspoints[YmaxIndex,1] + tol 
            except: 
                pass
            try: 
                self.obspoints[ZminIndex,2] = self.obspoints[ZminIndex,2] + tol 
            except: 
                pass
            try: 
                self.obspoints[ZmaxIndex,2] = self.obspoints[ZmaxIndex,2] + tol 
            except: 
                pass

            # -------- Determining if points within volume -------
            InVolIndex = torch.where((xmin[jj] <= self.obspoints[:,0]) & 
                            (self.obspoints[:,0] <= xmax[jj]) &
                            (ymin[jj] <= self.obspoints[:,1]) &
                            (self.obspoints[:,1] <= ymax[jj]) & 
                            (zmax[jj] <= self.obspoints[:,2]) &
                            (self.obspoints[:,2]<= zmin[jj]))

            try:
                InVol[InVolIndex]       = True
            except:
                pass

        #InVol : is True for every observation point inside a block
        #InVol_value stores the pressure for each point
        return InVol,InVol_value


    def _fFunc(self,x,y,z,R):
        return z*torch.atan((x*y)/(z*R)) - x*torch.log(torch.abs(R + y)) - y*torch.log(torch.abs(R + x))


    def LoadBlocks(self,BLOCKS,PoissonRatio=0.25,BiotCoef=1.0,ShearModulus=6e9):
        """
            Function to determine features for a series of strain volumes

            Input:
                BLOCKS - Pandas Array with headings 'X','Y','Z','wX','wY','wZ','dP'

            Return:
                self.vertices   -
                self.dP         -
                self.BlockThick - 
        """ 
        # ----- Defining vertices -----


        xmin = np.array(BLOCKS['X']);  xmax = np.array(BLOCKS['X'] + BLOCKS['wX'])
        ymin = np.array(BLOCKS['Y']);  ymax = np.array(BLOCKS['Y'] + BLOCKS['wY'])
        zmin = np.array(BLOCKS['Z']);  zmax = np.array((BLOCKS['Z'] + BLOCKS['wZ']))

        if type(BLOCKS) == pd.core.series.Series:
            self.vertices = torch.zeros(1,8,3)
        elif type(BLOCKS) == pd.core.frame.DataFrame:
            self.vertices = torch.zeros(len(BLOCKS),8,3)
        #for each block, self.vertices contains the 3 coordinates of each of the 8 vertices
        self.vertices[:,0,:] = torch.Tensor(np.vstack((xmin, ymin, -zmin)).transpose())
        self.vertices[:,1,:] = torch.Tensor(np.vstack((xmax, ymin, -zmin)).transpose())
        self.vertices[:,2,:] = torch.Tensor(np.vstack((xmin, ymax, -zmin)).transpose())
        self.vertices[:,3,:] = torch.Tensor(np.vstack((xmax, ymax, -zmin)).transpose())
        self.vertices[:,4,:] = torch.Tensor(np.vstack((xmin, ymin, -zmax)).transpose())
        self.vertices[:,5,:] = torch.Tensor(np.vstack((xmax, ymin, -zmax)).transpose())
        self.vertices[:,6,:] = torch.Tensor(np.vstack((xmin, ymax, -zmax)).transpose())
        self.vertices[:,7,:] = torch.Tensor(np.vstack((xmax, ymax, -zmax)).transpose())

        # ----- Defining vertices -----
        if type(BLOCKS) == pd.core.series.Series:
            self.dP         = torch.Tensor(np.array([BLOCKS['dP/Strain']]))
            self.BlockThick = torch.Tensor(np.array([BLOCKS['wZ']]))
            self.Cm         = torch.Tensor(np.array([BLOCKS['Cm']]))
        elif type(BLOCKS) == pd.core.frame.DataFrame:
            self.dP         = torch.Tensor(BLOCKS['dP/Strain'].to_numpy())
            self.BlockThick = torch.Tensor(BLOCKS['wZ'].to_numpy())
            self.Cm         = torch.Tensor(BLOCKS['Cm'].to_numpy())


        # ----- Determining the Elastic Constants for the blocks
        self.ν = PoissonRatio # Posson's ratio
        self.α = BiotCoef     # Biot's coefficient
        self.G = ShearModulus # Shear Modulus
        self._elastic_constant() #computes shear modulus, young's modulus & bulk modulus from PoissonRatio & BiotCoef


    def LoadPoints(self,POINTS,BoundaryCheck=False,BoundaryTolerance=1e-3):       
        
        # Loading the point locations and removing coordinate system
        self.obspoints = torch.Tensor(np.array(POINTS[['X','Y','Z']])) #POINTS = pd.DataFrame(POINTS_np,columns=['X','Y','Z'])
        #self.obspoints[:,-1] = -self.obspoints[:,-1] 

        # Determining any points that are on the block boundaries or lie within the blocks  
        self.obspoints_inBlock,self.obspoints_inBlock_value = self._BoundaryPerturbation(BoundaryTolerance,tol=BoundaryTolerance)

    def compute(self,verbrose=True,dv='cpu',GreenFunction=False):
        """
        Changes the coordinates before calling the function self.PointValues
        Changes them back again to go back to the previous coordinates system
        """

        # -- Removal of coordinate system --
        self.origin_correction = torch.mean(self.obspoints, dim=0); self.origin_correction[-1] = 0.
        self.obspoints        -= self.origin_correction
        self.vertices         -= self.origin_correction[None,None,:]

        if verbrose:
            print('Computing Strains for Blocks to Points')
            print('  Total of {} Blocks'.format(len(self.dP)))
            print('  Total of {} Points'.format(len(self.obspoints)))
        
        self.PointValues(device=dv,GreenFunction=GreenFunction)


        # -- Applying back the region location correction
        self.obspoints      += self.origin_correction
        self.vertices       += self.origin_correction[None,None,:]

    def PointValues(self,device='cpu',GreenFunction=False):
        """
            Computes strain at the point location due to the volume changes within the series of 
            blocks according the the Kuvshinov formulation
        """
        dv = torch.device(device)
        self.Disp      = torch.zeros(self.vertices.shape[0],self.obspoints.shape[0],3,device=dv)
        self.σ         = torch.zeros(self.vertices.shape[0],self.obspoints.shape[0],6,device=dv)
        Sσ             = torch.Tensor([-1, 1, 1, -1, 1, -1, -1, 1]).to(dv)
        self.vertices  = self.vertices.to(dv)
        self.obspoints = self.obspoints.to(dv)
        #VJC added for computation in cuda
        self.obspoints_inBlock = self.obspoints_inBlock.to(dv)


        for ii in tqdm(range(len(self.vertices)),desc='Computing Disp and σ'):
            block = self.vertices[ii,...]
            for jj, vertx in enumerate(block):
                x0 = torch.zeros(self.obspoints.shape[0],3,device=dv); x0[:,-1] = -self.obspoints[:,-1]
                vertx=vertx[None,:]
                x  = vertx - self.obspoints - x0
                ζp = x[...,2] + x0[...,2]
                ζm = x[...,2] - x0[...,2]
                rp = torch.sqrt((x[...,0] - x0[...,0])**2 + (x[...,1] - x0[...,1])**2 + ζp**2)
                rm = torch.sqrt((x[...,0] - x0[...,0])**2 + (x[...,1] - x0[...,1])**2 + ζm**2)


                # -- Displacements --
                self.Disp[ii,:,0] +=  Sσ[jj]*(self._fFunc(x[...,1],ζm,x[...,0],rm) 
                                  + (3. - 4.*self.ν)*self._fFunc(x[...,1], ζp,  x[...,0], rp) 
                                  - 2*x0[...,2]*torch.log(torch.abs(rp + x[...,1])))
                self.Disp[ii,:,1] +=  Sσ[jj]*(self._fFunc(x[...,0],ζm,x[...,1],rm)
                                  + (3. - 4.*self.ν)*self._fFunc(x[...,0], ζp,  x[...,1], rp)
                                  - 2*x0[...,2]*torch.log(torch.abs(rp + x[...,0])))
                self.Disp[ii,:,2] += -Sσ[jj]*(self._fFunc(x[...,0],x[...,1],ζm,rm)
                                  + (3. - 4.*self.ν)*self._fFunc(x[...,0],x[...,1], ζp, rp) 
                                  + 2 * x0[...,2] * torch.atan((ζp*rp)/(x[...,0]*x[...,1])) )

                # -- Stresses --
                self.σ[ii,:,0] += Sσ[jj]*(
                          - torch.atan((x[...,0]*rm)/(x[...,1]*ζm))
                          - (3.-4.*self.ν)*torch.atan((x[...,0]*rp)/(x[...,1]*ζp))
                          + 4.*self.ν*torch.atan((ζp*rp)/(x[...,0]*x[...,1]))
                          + 2.*x0[...,2]*x[...,0]*x[...,1]/(torch.pow(x[...,0],2) + torch.pow(ζp,2))/rp
                          - 2.*self.obspoints_inBlock.int())
                # σYY
                self.σ[ii,:,1] += Sσ[jj]*(
                          - torch.atan((x[...,1]*rm)/(x[...,0]*ζm))
                          - (3.-4.*self.ν)*torch.atan((x[...,1]*rp)/(x[...,0]*ζp))
                          + 4*self.ν*torch.atan((ζp*rp)/(x[...,0]*x[...,1]))
                          + 2*x0[...,2]*x[...,0]*x[...,1]/(torch.pow(x[...,1],2)+torch.pow(ζp,2))/rp
                          - 2.*self.obspoints_inBlock.int())
                # σZZ
                self.σ[ii,:,2] += Sσ[jj]*(
                               - torch.atan((ζm*rm)/(x[...,0]*x[...,1]))
                               + torch.atan((ζp*rp)/(x[...,0]*x[...,1]))
                               - 2*x0[...,2]*x[...,0]*x[...,1]/rp*(1/(torch.pow(x[...,0],2) + torch.pow(ζp,2)) + 1/(torch.pow(x[...,1],2) + torch.pow(ζp,2))))
                
                # σXY
                self.σ[ii,:,3] += -Sσ[jj]*(torch.log(abs(rm + ζm))
                               + (3.-4.*self.ν)*torch.log(abs(rp+ζp))
                               + (2*x0[...,2]/rp))
                # σXZ
                self.σ[ii,:,4] += Sσ[jj]*(torch.log(torch.abs((rm+x[...,1])/(rp+x[...,1])))
                               + 2*x0[...,2]*x[...,1]*ζp/(torch.pow(x[...,0],2)+torch.pow(ζp,2))/rp)
                # σYZ
                self.σ[ii,:,5] += Sσ[jj]*(torch.log(torch.abs((rm+x[...,0])/(rp+x[...,0])))
                             + 2*x0[...,2]*x[...,0]*ζp/(torch.pow(x[...,1],2)+torch.pow(ζp,2))/rp)


        if GreenFunction == True:
            self.Disp = self.Disp*(-self.Fu[:,None,None])
            self.σ    = self.σ*(self.Fσ[:,None,None]) # Minus for compressive sense

        else:
            self.Disp = torch.sum(self.Disp*(-self.Fu[:,None,None]*self.dP[:,None,None]),dim=0)
            self.σ    = torch.sum(self.σ*(self.Fσ[:,None,None]*self.dP[:,None,None]),dim=0) # Minus for compressive sense            


class Groningen:
    def __init__(self,ReservoirFolder):


        # Loading information into block info
        self.RES                  = import_RES_from_files(ReservoirFolder)
        FilterDistance=.75*1000
        from shapely.geometry import Point
        from shapely.geometry.polygon import Polygon
        MotifReservoir = np.zeros((self.RES['X'].shape))*np.nan
        polygon = Polygon(np.array(self.RES['Outline'][['X','Y']]))

        for ii in range(MotifReservoir.shape[0]):
            for jj in range(MotifReservoir.shape[1]):
                point = Point(self.RES['X'][ii,jj],self.RES['Y'][ii,jj])
                #mindist : distance du point le plus proche de RES['Outline']
                mindist = np.min(np.sqrt((self.RES['Outline']['X'] - self.RES['X'][ii,jj])**2 + (self.RES['Outline']['Y'] - self.RES['Y'][ii,jj])**2))              
                if polygon.contains(point) and (mindist>FilterDistance):
                    MotifReservoir[ii,jj] = 1
        from matplotlib.patches import Circle, Wedge, Polygon
        self.RES['Motif'] = MotifReservoir #MotifReservoir contains ones for each reservoir pointfar from the outline and NaNs outside.
        del MotifReservoir

        #self.BLOCKS is a dataframe conteaining position, dimensions, pressure and compressibility for each block.
        self.BLOCKS               = pd.DataFrame(np.zeros((len(self.RES['X'].flatten()),8)),columns=['X','wX','Y','wY','Z','wZ','dP/Strain','Cm'])
        self.BLOCKS['X']          = self.RES['X'].flatten()
        self.BLOCKS['Y']          = self.RES['Y'].flatten()
        self.BLOCKS['wX']         = self.BLOCKS['wY'] = 500
        self.BLOCKS['Z']          = self.RES['RES']['Depth'].flatten()
        self.BLOCKS['wZ']         = self.RES['RES']['Thickness'].flatten()
        self.BLOCKS['dP/Strain']  = (self.RES['PRESSURE']['TimeEvolution'][:,:,-1].flatten())
        self.BLOCKS['Cm']         = ((self.RES['Cm']['RAD'] + self.RES['Cm']['OL'] + self.RES['Cm']['InSARGPS'])/3).flatten()
        self.BLOCKS               = self.BLOCKS.dropna();
        #self.BLOCKS               = self.BLOCKS.reset_index(drop=True)


    def SingleBlock(self,PATH,index=0,sep=5,extr=100):
        BLOCK = self.BLOCKS.iloc[index] #Block contains X, dX, Y, dY, Z, dZ, dP/dStrain, Cm for a given block

        #Creation of a meshgrid on X and Z of the block's size + an extra space on the sides
        X,Z = np.meshgrid(np.arange(BLOCK['X']-extr,BLOCK['X']+BLOCK['wX']+extr,sep),np.arange(BLOCK['Z']-extr,BLOCK['Z']+BLOCK['wZ']+extr,sep))
        #POINTS_np contains the meshgrid coordinates
        POINTS_np = np.zeros((len(X.flatten()),3))
        POINTS_np[:,0] = X.flatten()
        POINTS_np[:,1] = BLOCK['Y']+BLOCK['wY']/2 #Y is fixed in the center of the block : cross-section over Y + wY/2
        POINTS_np[:,2] = Z.flatten()
        POINTS = pd.DataFrame(POINTS_np,columns=['X','Y','Z'])

        #Call the StrainVolume class for calculations
        self.StrainVolume = StrainVolume() #Create the object
        self.StrainVolume.LoadBlocks(BLOCK) #Create self.vertices (coordinates of blocks summits),
        #, self.dP, self.BlockThick and self.Cm plus the elastic constants
        self.StrainVolume.LoadPoints(POINTS)
        self.StrainVolume.compute() #Calculates deformations and stresses for each observation point

        # # -- Putting Output into Numpy array for plotting
        Disp            = self.StrainVolume.Disp.numpy()
        Stress          = np.zeros((self.StrainVolume.σ.shape[0],3,3)) #
        Stress[...,0,0] = self.StrainVolume.σ[:,0]                     # Sxx
        Stress[...,1,1] = self.StrainVolume.σ[:,1]                     # Syy
        Stress[...,2,2] = self.StrainVolume.σ[:,2]                     # Szz
        Stress[...,1,0] = Stress[...,0,1] = self.StrainVolume.σ[:,3]   # Sxy
        Stress[...,2,0] = Stress[...,0,2] = self.StrainVolume.σ[:,4]   # Sxz
        Stress[...,2,1] = Stress[...,1,2] = self.StrainVolume.σ[:,5]   # Syz


        # -- Plotting Displacement, Stress and Strain --
        plt.clf();plt.close('all')
        fig,axs = plt.subplots(1,3,figsize=(15,5))
        Dir = ['X','Y','Z']
        for ii in range(len(axs)):
            # Plotting the Block Outline
            xbl = np.array([BLOCK['X'],BLOCK['wX']+BLOCK['X'],BLOCK['wX']+BLOCK['X'],BLOCK['X'],BLOCK['X']])/1000
            ybl = np.array([BLOCK['Z'],BLOCK['Z'],BLOCK['wZ']+BLOCK['Z'],BLOCK['wZ']+BLOCK['Z'],BLOCK['Z']])/1000
            axs[ii].plot(xbl,ybl,color='w',linestyle='--',linewidth=2.0)
            #Plot displacements in each direction
            quad = axs[ii].scatter(self.StrainVolume.obspoints[:,0].numpy()/1000,self.StrainVolume.obspoints[:,2].numpy()/1000,15,self.StrainVolume.Disp[...,ii],marker='s',cmap='bwr'); 
            axs[ii].set_title('Displacement {}'.format(Dir[ii]))
            plt.colorbar(quad,ax=axs[ii],orientation='horizontal',label='Displacement')
            axs[ii].set_aspect('equal')
            axs[ii].set_ylabel('Depth (km)')
            axs[ii].set_xlabel('RDX (km)')
            axs[ii].set_xlim([self.StrainVolume.obspoints[:,0].numpy().min()/1000,self.StrainVolume.obspoints[:,0].numpy().max()/1000])
            axs[ii].set_ylim([self.StrainVolume.obspoints[:,2].numpy().min()/1000,self.StrainVolume.obspoints[:,2].numpy().max()/1000])
            axs[ii].invert_yaxis()
        plt.savefig('{}/SingleBlock_Index{}_Displacement.png'.format(PATH,index))


        fig,axs = plt.subplots(2,3,figsize=(15,10))
        Dir = ['Sxx','Syy','Szz','Sxy','Sxz','Syz']
        for ii in range(self.StrainVolume.σ.shape[-1]):
            # Plotting the Block Outline
            indy = (ii//3)
            indx = ii-indy*3
            xbl = np.array([BLOCK['X'],BLOCK['wX']+BLOCK['X'],BLOCK['wX']+BLOCK['X'],BLOCK['X'],BLOCK['X']])/1000
            ybl = np.array([BLOCK['Z'],BLOCK['Z'],BLOCK['wZ']+BLOCK['Z'],BLOCK['wZ']+BLOCK['Z'],BLOCK['Z']])/1000
            axs[indy,indx].plot(xbl,ybl,color='w',linestyle='--',linewidth=2.0)
            quad = axs[indy,indx].scatter(self.StrainVolume.obspoints[:,0].numpy()/1000,self.StrainVolume.obspoints[:,2].numpy()/1000,15,self.StrainVolume.σ[...,ii]/1e6,marker='s',cmap='bwr')#,vmin=-abs(self.StrainVolume.σ[...,ii]/1e9).max(),vmax=abs(self.StrainVolume.σ[...,ii]/1e9).max()); 
            axs[indy,indx].set_title('Stresss {}'.format(Dir[ii]))
            plt.colorbar(quad,ax=axs[indy,indx],orientation='horizontal',label='Stress (MPa)')
            axs[indy,indx].set_aspect('equal')
            axs[indy,indx].set_ylabel('Depth (km)')
            axs[indy,indx].set_xlabel('RDX (km)')
            axs[indy,indx].invert_yaxis()
        plt.savefig('{}/SingleBlock_Index{}_Stress.png'.format(PATH,index))

    def CrossSection(self,PATH,sep=10,ypos=590540,xlim=[257000,259000],dep_ext=300,dist=275,mu=0.66,Strike=270,Dip=85):

        # ========= Defining Points =======

        BLOCKS = self.BLOCKS[(self.BLOCKS['Y'] >= (ypos-dist)) & 
                             (self.BLOCKS['Y'] <= (ypos+dist)) & 
                             (self.BLOCKS['X'] >= (xlim[0]-dist)) & 
                             (self.BLOCKS['X'] <= (xlim[1]+dist))].reset_index(drop=True)

        zlim   = [BLOCKS['Z'].min()-dep_ext,BLOCKS['Z'].max()+BLOCKS['wZ'].iloc[BLOCKS['Z'].argmax()]+dep_ext]
        X,Z = np.meshgrid(np.arange(xlim[0],xlim[1],sep),np.arange(zlim[0],zlim[1],sep))
        POINTS_np = np.zeros((len(X.flatten()),3))
        POINTS_np[:,0] = X.flatten()
        POINTS_np[:,1] = ypos
        POINTS_np[:,2] = Z.flatten()
        POINTS = pd.DataFrame(POINTS_np,columns=['X','Y','Z'])

        self.StrainVolume = StrainVolume()
        self.StrainVolume.LoadBlocks(BLOCKS)
        self.StrainVolume.LoadPoints(POINTS)
        self.StrainVolume.compute()

        # -- Plotting Displacement, Stress and Strain --
        fig,axs = plt.subplots(3,1,figsize=(15,5))
        Dir = ['X','Y','Z']
        for ii in range(len(axs)):
            # Plotting the Block Outline
            for bi in range(len(BLOCKS)):
                BLOCK = BLOCKS.iloc[bi]
                xbl = np.array([BLOCK['X'],BLOCK['wX']+BLOCK['X'],BLOCK['wX']+BLOCK['X'],BLOCK['X'],BLOCK['X']])/1000
                ybl = np.array([BLOCK['Z'],BLOCK['Z'],BLOCK['wZ']+BLOCK['Z'],BLOCK['wZ']+BLOCK['Z'],BLOCK['Z']])/1000
                axs[ii].plot(xbl,ybl,color='k',linestyle='--',linewidth=1.0)
            quad = axs[ii].scatter(self.StrainVolume.obspoints[:,0].numpy()/1000,self.StrainVolume.obspoints[:,2].numpy()/1000,5,self.StrainVolume.Disp[...,ii],cmap='bwr',marker='s'); 
            axs[ii].set_title('Displacement {}'.format(Dir[ii]))
            plt.colorbar(quad,ax=axs[ii],label='Displacement')
            axs[ii].set_aspect('equal')
            axs[ii].set_ylabel('Depth (km)')
            axs[ii].set_xlabel('RDX (km)')
            axs[ii].set_xlim([xlim[0]/1000,xlim[1]/1000]);axs[ii].set_ylim([zlim[0]/1000,zlim[1]/1000])
            axs[ii].invert_yaxis()
        plt.savefig('{}/CrossSection_Displacement.png'.format(PATH))

        fig,axs = plt.subplots(6,1,figsize=(15,10))
        Dir = ['Sxx','Syy','Szz','Sxy','Sxz','Syz']
        for ii in range(self.StrainVolume.σ.shape[-1]):
            for bi in range(len(BLOCKS)):
                BLOCK = BLOCKS.iloc[bi]
                xbl = np.array([BLOCK['X'],BLOCK['wX']+BLOCK['X'],BLOCK['wX']+BLOCK['X'],BLOCK['X'],BLOCK['X']])/1000
                ybl = np.array([BLOCK['Z'],BLOCK['Z'],BLOCK['wZ']+BLOCK['Z'],BLOCK['wZ']+BLOCK['Z'],BLOCK['Z']])/1000
                axs[ii].plot(xbl,ybl,color='k',linestyle='--',linewidth=1.0)

            quad = axs[ii].scatter(self.StrainVolume.obspoints[:,0].numpy()/1000,self.StrainVolume.obspoints[:,2].numpy()/1000,5,self.StrainVolume.σ[...,ii]/1e6,cmap='bwr',marker='s'); 
            axs[ii].set_title('Stresss {}'.format(Dir[ii]))
            plt.colorbar(quad,ax=axs[ii],label='Stress (MPa)')
            axs[ii].set_aspect('equal')
            axs[ii].set_ylabel('Depth (km)')
            axs[ii].set_xlabel('RDX (km)')
            axs[ii].set_xlim([xlim[0]/1000,xlim[1]/1000]);axs[ii].set_ylim([zlim[0]/1000,zlim[1]/1000])
            axs[ii].invert_yaxis()
        plt.savefig('{}/CrossSection_Stress.png'.format(PATH))



        Fault_normal = [-np.sin(Dip*(np.pi/180))*np.sin(Strike*(np.pi/180)),
                        np.sin(Dip*(np.pi/180))*np.cos(Strike*(np.pi/180)),
                        -np.cos(Dip*(np.pi/180))]
        Stress          = np.zeros((self.StrainVolume.σ.shape[0],3,3))
        Stress[...,0,0] = self.StrainVolume.σ[:,0]                     # Sxx
        Stress[...,1,1] = self.StrainVolume.σ[:,1]                     # Syy
        Stress[...,2,2] = self.StrainVolume.σ[:,2]                     # Szz
        Stress[...,1,0] = Stress[...,0,1] = self.StrainVolume.σ[:,3]   # Sxy
        Stress[...,2,0] = Stress[...,0,2] = self.StrainVolume.σ[:,4]   # Sxz
        Stress[...,2,1] = Stress[...,1,2] = self.StrainVolume.σ[:,5]   # Syz

        Stress_faultnormal  = np.dot(np.dot(Stress,Fault_normal),Fault_normal)
        Stress_faultshear   = np.sqrt(np.sum(np.dot(Stress,Fault_normal)**2,axis=-1) - Stress_faultnormal**2)
        Stress_faultcoulomb = (Stress_faultshear - mu*(Stress_faultnormal-self.StrainVolume.obspoints_inBlock_value.numpy()))/1e6

        Stress_maxshear  = 0.5*(np.max(np.linalg.eigvals(np.nan_to_num(Stress)),axis=-1) - np.min(np.linalg.eigvals(np.nan_to_num(Stress)),axis=-1))
        Stress_maxnormal = 0.5*(np.max(np.linalg.eigvals(np.nan_to_num(Stress)),axis=-1) + np.min(np.linalg.eigvals(np.nan_to_num(Stress)),axis=-1))
        Stress_maxcoulomb = (Stress_maxshear - mu*(Stress_maxnormal-self.StrainVolume.obspoints_inBlock_value.numpy()))/1e6
        Stress_maxcoulomb[Stress_maxcoulomb==np.inf]=0

        fig,axs = plt.subplots(2,1,figsize=(15,5))
        for bi in range(len(BLOCKS)):
            BLOCK = BLOCKS.iloc[bi]
            xbl = np.array([BLOCK['X'],BLOCK['wX']+BLOCK['X'],BLOCK['wX']+BLOCK['X'],BLOCK['X'],BLOCK['X']])/1000
            ybl = np.array([BLOCK['Z'],BLOCK['Z'],BLOCK['wZ']+BLOCK['Z'],BLOCK['wZ']+BLOCK['Z'],BLOCK['Z']])/1000
            axs[0].plot(xbl,ybl,color='k',linestyle='--',linewidth=.5)
        quad0 = axs[0].scatter(self.StrainVolume.obspoints[:,0].numpy()/1000,self.StrainVolume.obspoints[:,2].numpy()/1000,5,Stress_faultcoulomb,cmap='bwr',vmin=-np.nanstd(Stress_maxcoulomb)*3,vmax=np.nanstd(Stress_maxcoulomb)*3,marker='s'); 
        axs[0].set_title('Coulomb Stress on Strike={},Dip={} (MPa)'.format(Strike,Dip))
        plt.colorbar(quad0,ax=axs[0],label='Stress (MPa)')
        axs[0].set_aspect('equal')
        axs[0].set_ylabel('Depth (km)')
        axs[0].set_xlabel('RDX (km)')
        axs[0].set_xlim([xlim[0]/1000,xlim[1]/1000]);axs[0].set_ylim([zlim[0]/1000,zlim[1]/1000])
        axs[0].invert_yaxis()

        quad = axs[1].scatter(self.StrainVolume.obspoints[:,0].numpy()/1000.,self.StrainVolume.obspoints[:,2].numpy()/1000.,5,Stress_maxcoulomb,cmap='bwr',vmin=-np.nanstd(Stress_maxcoulomb)*3,vmax=np.nanstd(Stress_maxcoulomb)*3,marker='s'); 
        for bi in range(len(BLOCKS)):
            BLOCK = BLOCKS.iloc[bi]
            xbl = np.array([BLOCK['X'],BLOCK['wX']+BLOCK['X'],BLOCK['wX']+BLOCK['X'],BLOCK['X'],BLOCK['X']])/1000.
            ybl = np.array([BLOCK['Z'],BLOCK['Z'],BLOCK['wZ']+BLOCK['Z'],BLOCK['wZ']+BLOCK['Z'],BLOCK['Z']])/1000.
            axs[1].plot(xbl,ybl,color='k',linestyle='--',linewidth=.5)
        axs[1].set_title('Maximum Coulomb Stress (MPa)'.format(Strike,Dip))
        plt.colorbar(quad,ax=axs[1],label='Stress (MPa)')
        axs[1].set_aspect('equal')
        axs[1].set_ylabel('Depth (km)')
        axs[1].set_xlabel('RDX (km)')
        axs[1].set_xlim([xlim[0]/1000.,xlim[1]/1000.]);axs[1].set_ylim([zlim[0]/1000.,zlim[1]/1000.])
        axs[1].invert_yaxis()

        plt.savefig('{}/CrossSection_CS.png'.format(PATH))

    def DepthSlice(self,PATH,depth=0.0,mu=0.3,Strike=270,Dip=85,smoothing=6.0,GreenFunction=False,verbrose=True):
        BLOCKS            = self.BLOCKS
        #Observation points : top view of a cross section at a given depth
        #X and Y correspond to the Blocks' centers.
        POINTS            = BLOCKS[['X','Y','Z']]
        POINTS['X']       = BLOCKS['X'] + BLOCKS['wX']/2
        POINTS['Y']       = BLOCKS['Y'] + BLOCKS['wY']/2
        POINTS['Z']       = depth

        self.StrainVolume = StrainVolume()
        self.StrainVolume.LoadBlocks(BLOCKS)
        self.StrainVolume.LoadPoints(POINTS)
        self.StrainVolume.compute(GreenFunction=GreenFunction)

        if verbrose:
          Dir = ['X','Y','Z']
          fig,axs = plt.subplots(1,3,figsize=(15,5))
          for ii in range(self.StrainVolume.Disp.shape[-1]):
              p1     = axs[ii].add_patch(Polygon(np.array(self.RES['Outline'][['X','Y']]/1000),fill=False))
              GridV    = np.zeros(len(self.RES['X'].flatten()))*np.nan
              GridV[self.BLOCKS.index] = self.StrainVolume.Disp[...,ii]
              GridV    = GridV.reshape(self.RES['X'].shape)

              quad1 = axs[ii].pcolormesh(RES['X']/1000,RES['Y']/1000,GridV,vmin=-np.nanmax(abs(GridV)),vmax=np.nanmax(abs(GridV)),cmap='bwr',clip_path=p1,clip_on=True)
              axs[ii].set_aspect('equal')
              axs[ii].set_xlabel('RDX (km)')
              axs[ii].set_ylabel('RDY (km)')
              plt.colorbar(quad1,ax=axs[ii],label='Displacement {} (m)'.format(Dir[ii]),orientation='horizontal')
          plt.title('Displacement')
          plt.savefig('{}/DepthSlice_Displacement.png'.format(PATH))

          kernel = Gaussian2DKernel(smoothing)
          Dir = ['X','Y','Z']
          fig,axs = plt.subplots(1,3,figsize=(15,5))
          for ii in range(self.StrainVolume.Disp.shape[-1]):
              p1     = axs[ii].add_patch(Polygon(np.array(self.RES['Outline'][['X','Y']]/1000),fill=False))
              GridV    = np.zeros(len(self.RES['X'].flatten()))*np.nan
              GridV[self.BLOCKS.index] = self.StrainVolume.Disp[...,ii]
              GridV    = GridV.reshape(self.RES['X'].shape)
              GridV    = convolve(GridV*self.RES['Motif'],kernel,nan_treatment='fill')
              quad1 = axs[ii].pcolormesh(RES['X']/1000,RES['Y']/1000,GridV,vmin=-np.nanmax(abs(GridV)),vmax=np.nanmax(abs(GridV)),cmap='bwr',clip_path=p1,clip_on=True)
              axs[ii].set_aspect('equal')
              axs[ii].set_xlabel('RDX (km)')
              axs[ii].set_ylabel('RDY (km)')
              plt.colorbar(quad1,ax=axs[ii],label='Displacement {} (m)'.format(Dir[ii]),orientation='horizontal')
          plt.title('Displacement')
          plt.savefig('{}DepthSlice_Smoothed_Displacement.png'.format(PATH))

          kernel = Gaussian2DKernel(smoothing)
          fig,axs = plt.subplots(2,3,figsize=(15,10))
          Dir = ['Sxx','Syy','Szz','Sxy','Sxz','Syz']
          for ii in range(self.StrainVolume.σ.shape[-1]):
              # Plotting the Block Outline
              indy = (ii//3)
              indx = ii-indy*3
              p1     = axs[indy,indx].add_patch(Polygon(np.array(self.RES['Outline'][['X','Y']]/1000),fill=False))
              GridV    = np.zeros(len(self.RES['X'].flatten()))*np.nan
              GridV[self.BLOCKS.index] = self.StrainVolume.σ[...,ii]/1e6
              GridV    = GridV.reshape(self.RES['X'].shape)
              GridV    = convolve(GridV*self.RES['Motif'],kernel,nan_treatment='fill')
              quad = axs[indy,indx].pcolormesh(RES['X']/1000,RES['Y']/1000,GridV,vmin=-np.nanmax(abs(GridV)),vmax=np.nanmax(abs(GridV)),cmap='bwr',clip_path=p1,clip_on=True)#,vmin=-abs(self.StrainVolume.σ[...,ii]/1e9).max(),vmax=abs(self.StrainVolume.σ[...,ii]/1e9).max()); 
              axs[indy,indx].set_title('Stresss {}'.format(Dir[ii]))
              plt.colorbar(quad,ax=axs[indy,indx],orientation='horizontal',label='Stress {} (MPa)'.format(ii))
              axs[indy,indx].set_aspect('equal')
              axs[indy,indx].set_ylabel('Depth (km)')
              axs[indy,indx].set_xlabel('RDX (km)')
          plt.savefig('{}DepthSlice_Smoothed_Stress.png'.format(PATH))

          fig,axs = plt.subplots(2,3,figsize=(15,10))
          Dir = ['Sxx','Syy','Szz','Sxy','Sxz','Syz']
          for ii in range(self.StrainVolume.σ.shape[-1]):
              # Plotting the Block Outline
              indy = (ii//3)
              indx = ii-indy*3
              p1     = axs[indy,indx].add_patch(Polygon(np.array(self.RES['Outline'][['X','Y']]/1000),fill=False))
              GridV    = np.zeros(len(self.RES['X'].flatten()))*np.nan
              GridV[self.BLOCKS.index] = self.StrainVolume.σ[...,ii]/1e6
              GridV    = GridV.reshape(self.RES['X'].shape)
              quad = axs[indy,indx].pcolormesh(RES['X']/1000,RES['Y']/1000,GridV,vmin=-np.nanmax(abs(GridV)),vmax=np.nanmax(abs(GridV)),cmap='bwr',clip_path=p1,clip_on=True)#,vmin=-abs(self.StrainVolume.σ[...,ii]/1e9).max(),vmax=abs(self.StrainVolume.σ[...,ii]/1e9).max()); 
              axs[indy,indx].set_title('Stresss {}'.format(Dir[ii]))
              plt.colorbar(quad,ax=axs[indy,indx],orientation='horizontal',label='Stress {} (MPa)'.format(ii))
              axs[indy,indx].set_aspect('equal')
              axs[indy,indx].set_ylabel('Depth (km)')
              axs[indy,indx].set_xlabel('RDX (km)')
          plt.savefig('{}DepthSlice_Stress.png'.format(PATH))


          Fault_normal = [-np.sin(Dip*(np.pi/180))*np.sin(Strike*(np.pi/180)),
                          np.sin(Dip*(np.pi/180))*np.cos(Strike*(np.pi/180)),
                          -np.cos(Dip*(np.pi/180))]
          Stress          = np.zeros((self.StrainVolume.σ.shape[0],3,3))
          Stress[...,0,0] = self.StrainVolume.σ[:,0]                     # Sxx
          Stress[...,1,1] = self.StrainVolume.σ[:,1]                     # Syy
          Stress[...,2,2] = self.StrainVolume.σ[:,2]                     # Szz
          Stress[...,1,0] = Stress[...,0,1] = self.StrainVolume.σ[:,3]   # Sxy
          Stress[...,2,0] = Stress[...,0,2] = self.StrainVolume.σ[:,4]   # Sxz
          Stress[...,2,1] = Stress[...,1,2] = self.StrainVolume.σ[:,5]   # Syz

          #Stresses: normal, shear, and Coulomb
          #Normal stress : n.σ.n
          Stress_faultnormal  = np.dot(np.dot(Stress,Fault_normal),Fault_normal)
          #Shear stress : σ.n - (n.σ.n)n
          Stress_faultshear   = np.sqrt(np.sum(np.dot(Stress,Fault_normal)**2,axis=-1) - Stress_faultnormal**2)
          #obspoints_inBlock_value contains the pressure values in the reservoir
          #Coulomb stress : =tau - friction coefficient * (sigma_N -p)
          Stress_faultcoulomb = (Stress_faultshear - mu*(Stress_faultnormal-self.StrainVolume.obspoints_inBlock_value.numpy()))/1e6

          GridFCL    = np.zeros(len(self.RES['X'].flatten()))*np.nan
          GridFCL[self.BLOCKS.index] = Stress_faultcoulomb
          GridFCL    = GridFCL.reshape(self.RES['X'].shape)

          #Max shear : (max eigenvalue - min eigenvalue )/2
          Stress_maxshear  = 0.5*(np.max(np.linalg.eigvals(np.nan_to_num(Stress)),axis=-1) - np.min(np.linalg.eigvals(np.nan_to_num(Stress)),axis=-1))
          #Max normal
          Stress_maxnormal = 0.5*(np.max(np.linalg.eigvals(np.nan_to_num(Stress)),axis=-1) + np.min(np.linalg.eigvals(np.nan_to_num(Stress)),axis=-1))
          #Coulomb stress
          Stress_maxcoulomb = (Stress_maxshear - mu*(Stress_maxnormal-self.StrainVolume.obspoints_inBlock_value.numpy()))/1e6
          Stress_maxcoulomb[Stress_maxcoulomb==np.inf]=0

          GridMCL    = np.zeros(len(self.RES['X'].flatten()))*np.nan
          GridMCL[self.BLOCKS.index] = Stress_maxcoulomb
          GridMCL    = GridMCL.reshape(self.RES['X'].shape)
          
          #PLOT FAULT COULOMB STRESS
          fig,axs = plt.subplots(1,2,figsize=(15,5))
          p1    = axs[0].add_patch(Polygon(np.array(self.RES['Outline'][['X','Y']]/1000),fill=False))
          quad0 = axs[0].pcolormesh(RES['X']/1000,RES['Y']/1000,GridFCL*self.RES['Motif'],cmap='bwr',vmin=-np.nanmax(abs(GridFCL)),vmax=np.nanmax(abs(GridFCL)),clip_path=p1,clip_on=True); 
          axs[0].set_title('Fault CoulombStress (MPa)')
          plt.colorbar(quad0,ax=axs[0],label='Stress (MPa)')
          axs[0].set_aspect('equal')
          axs[0].set_ylabel('RDY(km)')
          axs[0].set_xlabel('RDX (km)')

          #PLOT MAXIMUM COULOMB STRESS
          p1    = axs[1].add_patch(Polygon(np.array(self.RES['Outline'][['X','Y']]/1000),fill=False))
          quad1 = axs[1].pcolormesh(RES['X']/1000,RES['Y']/1000,GridMCL*self.RES['Motif'],cmap='bwr',vmin=-np.nanmax(abs(GridMCL)),vmax=np.nanmax(abs(GridMCL)),clip_path=p1,clip_on=True); 
          axs[1].set_title('Maximum CoulombStress (MPa)')
          plt.colorbar(quad1,ax=axs[1],label='Stress (MPa)')
          axs[1].set_aspect('equal')
          axs[1].set_ylabel('RDY(km)')
          axs[1].set_xlabel('RDX (km)')
          plt.savefig('{}DepthSlice_CS.png'.format(PATH))

          #PLOT SMOOTHED MAXIMUM COULOMB STRESS
          kernel = Gaussian2DKernel(smoothing)
          GridFCL = convolve(GridFCL*self.RES['Motif'],kernel,nan_treatment='fill')
          GridMCL = convolve(GridMCL*self.RES['Motif'],kernel,nan_treatment='fill')
          fig,axs = plt.subplots(1,2,figsize=(15,5))
          p1    = axs[0].add_patch(Polygon(np.array(self.RES['Outline'][['X','Y']]/1000),fill=False))
          quad0 = axs[0].pcolormesh(RES['X']/1000,RES['Y']/1000,GridFCL,cmap='bwr',vmin=-np.nanmax(abs(GridFCL)),vmax=np.nanmax(abs(GridFCL)),clip_path=p1,clip_on=True); 
          axs[0].set_title('Fault CoulombStress (MPa)')
          plt.colorbar(quad0,ax=axs[0],label='Stress (MPa)')
          axs[0].set_aspect('equal')
          axs[0].set_ylabel('RDY(km)')
          axs[0].set_xlabel('RDX (km)')
          p1    = axs[1].add_patch(Polygon(np.array(self.RES['Outline'][['X','Y']]/1000),fill=False))
          quad1 = axs[1].pcolormesh(RES['X']/1000,RES['Y']/1000,GridMCL,cmap='bwr',vmin=-np.nanmax(abs(GridMCL)),vmax=np.nanmax(abs(GridMCL)),clip_path=p1,clip_on=True); 
          axs[1].set_title('Maximum CoulombStress (MPa)')
          plt.colorbar(quad1,ax=axs[1],label='Stress (MPa)')
          axs[1].set_aspect('equal')
          axs[1].set_ylabel('RDY(km)')
          axs[1].set_xlabel('RDX (km)')
          plt.savefig('{}DepthSlice_Smoothed_CS.png'.format(PATH))

    def DepthSlice_FullTEvolution_V2(self,PATH, PRES, activate_cuda=False, depth=0.0,mu=0.3,Strike=270,Dip=85,smoothing=6.0,verbrose=False,GreenFunction=False):        

        if activate_cuda :
            print ('Compute displacement and stress on cuda...')
            disp_cuda = self.StrainVolume.Disp.to('cuda')
            sigma_cuda = self.StrainVolume.σ.to('cuda')
            pres_cuda = torch.tensor(PRES, device='cuda')
            DISP_f     = torch.zeros((self.StrainVolume.Disp.shape[1],PRES.shape[2],self.StrainVolume.Disp.shape[2]), device='cuda')
            STRESS_f   = torch.zeros((self.StrainVolume.Disp.shape[1],PRES.shape[2],self.StrainVolume.σ.shape[2]), device='cuda')

            for ii in tqdm(range(PRES.shape[2]),desc='Time-Evolution Values'):
                DISP_f[:,ii,:] = torch.sum(disp_cuda*pres_cuda[...,ii].flatten()[self.BLOCKS.index,None,None],dim=0)
                STRESS_f[:,ii,:] = torch.sum(sigma_cuda*pres_cuda[...,ii].flatten()[self.BLOCKS.index,None,None],dim=0)

            DISP_f = DISP_f.cpu()
            STRESS_f = STRESS_f.cpu()

        else :
            print ('Compute displacement and stress on cpu... (could be long, set activate_cuda=True if you have a gpu)')
            DISP_f     = torch.zeros((self.StrainVolume.Disp.shape[1],PRES.shape[2],self.StrainVolume.Disp.shape[2]))
            STRESS_f   = torch.zeros((self.StrainVolume.Disp.shape[1],PRES.shape[2],self.StrainVolume.σ.shape[2]))

            for ii in tqdm(range(PRES.shape[2]),desc='Time-Evolution Values'):
                DISP_f[:,ii,:]   = torch.sum(self.StrainVolume.Disp*PRES[...,ii].flatten()[self.BLOCKS.index,None,None],dim=0)
                STRESS_f[:,ii,:] = torch.sum(self.StrainVolume.σ*PRES[...,ii].flatten()[self.BLOCKS.index,None,None],dim=0)


        DISP   = np.zeros((len(self.RES['X'].flatten()),DISP_f.shape[-2],DISP_f.shape[-1]))*np.nan
        DISP[self.BLOCKS.index,...] = DISP_f
        DISP   = DISP.reshape((self.RES['X'].shape[0],self.RES['X'].shape[1],DISP_f.shape[-2],DISP_f.shape[-1]))

        STRESSA  = np.zeros((len(self.RES['X'].flatten()),STRESS_f.shape[-2],STRESS_f.shape[-1]))*np.nan
        STRESSA[self.BLOCKS.index,...] = STRESS_f
        STRESSA  = STRESSA.reshape((self.RES['X'].shape[0],self.RES['X'].shape[1],STRESS_f.shape[-2],STRESS_f.shape[-1]))
        STRESS  = np.zeros(STRESSA.shape[:-1] + (3,3))
        STRESS[...,0,0] = STRESSA[...,0]
        STRESS[...,1,1] = STRESSA[...,1]
        STRESS[...,2,2] = STRESSA[...,2]
        STRESS[...,1,0] = STRESS[...,0,1] = STRESSA[...,3]
        STRESS[...,2,0] = STRESS[...,0,2] = STRESSA[...,4]
        STRESS[...,2,1] = STRESS[...,1,2] = STRESSA[...,5]

        PRES_p  = np.zeros((len(self.RES['X'].flatten())))*np.nan
        PRES_p[self.BLOCKS.index] = self.StrainVolume.obspoints_inBlock_value
        PRES_p  = PRES_p.reshape((self.RES['X'].shape))
        PRES_p[PRES_p>0] = 1
        PRES_p  = PRES*PRES_p[:,:,None]

        Fault_normal = [-np.sin(Dip*(np.pi/180))*np.sin(Strike*(np.pi/180)),
            np.sin(Dip*(np.pi/180))*np.cos(Strike*(np.pi/180)),
            -np.cos(Dip*(np.pi/180))]

        Stress_faultnormal  = np.dot(np.dot(STRESS,Fault_normal),Fault_normal)
        Stress_faultshear   = np.sqrt(np.sum(np.dot(STRESS,Fault_normal)**2,axis=-1) - Stress_faultnormal**2)
        CS_FAULT            = (Stress_faultshear - mu*(Stress_faultnormal-PRES_p))/1e6
        del Stress_faultnormal,Stress_faultshear

        Stress_maxshear     = 0.5*(np.max(np.linalg.eigvals(np.nan_to_num(STRESS)),axis=-1) - np.min(np.linalg.eigvals(np.nan_to_num(STRESS)),axis=-1))
        Stress_maxnormal    = 0.5*(np.max(np.linalg.eigvals(np.nan_to_num(STRESS)),axis=-1) + np.min(np.linalg.eigvals(np.nan_to_num(STRESS)),axis=-1))
        
        #MAX COULOMB STRESS : VALENTIN'S MODIFICATION to compute it from sigma1 and sigma3 directly.
        failure_angle = np.arctan(mu)
        sig1 = np.min(np.linalg.eigvals(np.nan_to_num(STRESS)),axis=-1)
        sig3 = np.max(np.linalg.eigvals(np.nan_to_num(STRESS)),axis=-1) #shape (97,99,nb_iterations)
        CS_MAX = -( sig1*np.sin(failure_angle) + 0.5*(sig3-sig1)*(np.sin(failure_angle)-1)) * 1e-6
        CS_MAX[CS_MAX==np.inf]=0

        #Added part
        GridMCL    = np.zeros((self.RES['X'].shape[0], self.RES['X'].shape[1], CS_MAX.shape[2]))*np.nan
        for i in range (GridMCL.shape[2]):
            GridMCL[:, :, i] = CS_MAX[:, :, i].reshape(self.RES['X'].shape)
            GridMCL[:, :, i] = GridMCL[:, :, i]*self.RES['Motif']


        from copy import copy
        CS_MAX_filt   = copy(CS_MAX)
        CS_FAULT_filt = copy(CS_FAULT)
        kernel = Gaussian2DKernel(smoothing)
        for ii in range(CS_FAULT_filt.shape[-1]):
            CS_MAX_filt[...,ii]   = convolve(CS_MAX_filt[...,ii]*self.RES['Motif'],kernel,nan_treatment='fill')
            CS_FAULT_filt[...,ii] = convolve(CS_FAULT_filt[...,ii]*self.RES['Motif'],kernel,nan_treatment='fill')


        Values = {}
        Values['Pressure']     = PRES
        Values['Displacement'] = DISP
        Values['Stress']       = STRESS
        Values['Pressure at points']               = PRES_p
        Values['Coulomb Stress Fault']             = {}
        Values['Coulomb Stress Fault']['Mu']       = mu
        Values['Coulomb Stress Fault']['Strike']   = Strike
        Values['Coulomb Stress Fault']['Dip']      = Dip
        Values['Coulomb Stress Fault']['RAW']      = CS_FAULT
        Values['Coulomb Stress Fault']['Filtered'] = CS_FAULT_filt
        Values['Max Coulomb Stress']               = {}
        Values['Max Coulomb Stress']['Mu']         = mu
        Values['Max Coulomb Stress']['RAW']        = CS_MAX
        Values['Max Coulomb Stress']['Filtered']   = CS_MAX_filt

        #Added line
        Values['GridMCL'] = GridMCL


        return Values



