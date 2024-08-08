import pandas as pd
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy
import numpy as np
import math
import os
import matplotlib.pyplot as plt
from colorama import Fore, Back, Style
from scipy.interpolate import RBFInterpolator
from mpl_toolkits.mplot3d import Axes3D
import time
from tqdm import tqdm
from mshr import*
import random
import pickle as pkl
from util.auxiliary_classes import*

########################################################################################################################
# Extract points, pressure and h of a given iteration from calculated and stored data 

def extract_points_p_h(iteration):
    reader_p = vtk.vtkXMLUnstructuredGridReader()
    reader_p.SetFileName(file_name_p(iteration))
    reader_p.Update()  # Needed because of GetScalarRange
    points = vtk_to_numpy(reader_p.GetOutput().GetPoints().GetData()).tolist()
    pressure_int = (1e-5*vtk_to_numpy(reader_p.GetOutput().GetPointData().GetArray(0))).tolist()
    reader_h = vtk.vtkXMLUnstructuredGridReader()
    reader_h.SetFileName(file_name_h(iteration))
    reader_h.Update()  # Needed because of GetScalarRange
    h = vtk_to_numpy(reader_h.GetOutput().GetPointData().GetArray(0)).tolist()
    return(points, pressure_int, h)

########################################################################################################################
# Find names of files which correspond to a given iteration

def file_name_p(iteration):
    if iteration == 0 : return(f"../data-out/reservoir/output/p/output_p000000.vtu")
    digits = int(math.log10(iteration))+1
    car = ""
    for i in range(6-digits): car += "0"
    return(f"../data-out/reservoir/output/p/output_p" + car + str(iteration)+ ".vtu")

def file_name_h(Number):
    if Number == 0 : return(f"../data-out/reservoir/output/h/output_h000000.vtu")
    digits = int(math.log10(Number))+1
    car = ""
    for i in range(6-digits): car += "0"
    return(f"../data-out/reservoir/output/h/output_h" + car + str(Number)+ ".vtu")

########################################################################################################################
# Convert time t in seconds to calendar format (min, hour, day, year)

def day_format(t):
    min = 60 ; hour = 60*min ; day = 24*hour ; year = 365*day
    if t<min : return(str(int(t)) + ' s')
    elif t<hour : return(str(int(t/min)) + ' min')
    elif t<day : return(str(int(10*t/hour)/10) + ' h')
    elif t<year : return(str(int(10*t/day)/10) + ' days')
    else : return(str(int(100*t/year)/100) + ' years')

########################################################################################################################
# Create the mesh of the problem
# domain, Nm, Nhr, Nst, Sources_g, sigma, T_tot, phi, sg, rho_g0, H must be provided

def create_mesh(domain, Nm, Nhr, Nst, Sources_g, sigma, T_tot, phi, sg, rho_g0, H):
    mesh = generate_mesh(domain, Nm)
    for i in range(Nhr):
        cell_markers = MeshFunction("bool", mesh, mesh.topology().dim())
        for c in cells(mesh):
            cell_markers[c] = False
            for source in Sources_g :
                p = source[0]
                Injected_mass = np.matmul(np.array(source[1]), np.array(source[2])[:, 1]-np.array(source[2])[:, 0])
                Hydraulic_radius = np.sqrt(Injected_mass/(phi*sg*rho_g0*H))
                if c.midpoint().distance(p) < 4*Hydraulic_radius: cell_markers[c] = True
        mesh = refine(mesh, cell_markers)
    for i in range(Nst):
        cell_markers = MeshFunction("bool", mesh, mesh.topology().dim())
        for c in cells(mesh):
            cell_markers[c] = False
            for source in Sources_g :
                p = source[0]
                if c.midpoint().distance(p) < 5*sigma: cell_markers[c] = True
        mesh = refine(mesh, cell_markers)
    return(mesh)

########################################################################################################################
# Define the true physical value for the thickness of the gas (if h<0 : h=0, if h>H : h=H, else : keep value of h)
# H must be provided

def true_h(H, h): return((H+(h+abs(h))/2 - abs(H-(h+abs(h))/2))/2)

########################################################################################################################
# Find extrem values of h and p at a given iteration

def min_max_pressure_h(iteration, H, g, sg, Nz, rho_g, rho_w, List_t):
    points_init, pressure_int, h = extract_points_p_h(iteration)
    H_data = H.vector().get_local()
    points = [] ; pressure = []
    for i in range(len(points_init)):
        dz = H_data[i]/Nz
        z = H_data[i]-h[i]
        points.append([points_init[i][0], points_init[i][1], z])
        pressure.append(pressure_int[i])
        p_int = pressure_int[i]
        p = p_int ; z += dz
        while z<H_data[i]: # we start from the interface 
            points.append([points_init[i][0], points_init[i][1], z])
            rho_avg = sg*rho_g(p) + (1-sg)*rho_w(p)
            pressure.append(p - 1e-5*rho_avg*g*dz)
            p -= 1e-5*rho_avg*g*dz
            z += dz      
        z = H_data[i]-h[i]-dz ; p = p_int
        while z>0: # we start from the interface
            points.append([points_init[i][0], points_init[i][1], z])
            pressure.append(p + 1e-5*rho_w(p)*g*dz)
            p += 1e-5*rho_w(p)*g*dz
            z -= dz
    return(np.min(pressure), np.max(pressure), np.min(h), np.max(h))

########################################################################################################################
# Calculate extrem values for h and p during the entire simulations
# Nt, H, g, sg, Nz, rho_g, rho_w, List_t must be provided

def min_max_pressure_h_total(Nt, H, g, sg, Nz, rho_g, rho_w, List_t):
    p_min, p_max, h_min, h_max = 1e10,0,1e10,0
    print('Searching for extrem values of p and h...')
    for i in tqdm(range(Nt)):
        p_min_t, p_max_t, h_min_t, h_max_t = min_max_pressure_h(i, H, g, sg, Nz, rho_g, rho_w, List_t)
        if p_min_t < p_min : p_min = p_min_t
        if p_max_t > p_max : p_max = p_max_t
        if h_min_t < h_min : h_min = h_min_t
        if h_max_t > h_max : h_max = h_max_t
    return(p_min, p_max, h_min, h_max)

########################################################################################################################
# Determine if we inject or not at time t for a given injection profile (which corresponds to a source)

def bin_inject(t, Inj_profile):
    if len(Inj_profile) == 0 : return 0
    elif t>=Inj_profile[0] and t<=Inj_profile[1] : return 1
    return 0

########################################################################################################################
# Define source function at time t from the list with sources data (Sources_g)
# ME1, sigma must be provided

def s(Sources_g, t, ME1, sigma):
    alpha = 1/sigma**2 ; K = 1/(np.pi*sigma**2)
    f_g = '0'
    s_g_ = Function(ME1)
    for source in Sources_g :
        x = source[0][0] ; y = source[0][1]
        for i in range(len(source[1])):
            rate = source[1][i] ; Inj_profile = source[2][i]
            bin = bin_inject(t, Inj_profile)
            f_g += '+' + str(bin) + '*' + str(rate) + '*K*exp(-alpha*((x[0]-'+str(x)+')*(x[0]-'+str(x)+')+(x[1]-'+str(y)+')*(x[1]-'+str(y)+')))'
    s_g = Expression(f_g, K=1/(np.pi*sigma**2), alpha=1/sigma**2, degree=2)
    return(s_g)

########################################################################################################################
# Define problem solved by the solver at a time t
# dt Sources_g, ME1, H, phi, sg, g, rho_g, rho_w, rho_g0, rho_w0, mug, muw, cr, cg, cw, u, p, h, p0, h0, du, bcs, sigma, v_p, v_h, k, krg,
# krw must be provided

def define_problem(dt, t, Sources_g, ME1, H, phi, sg, g, rho_g, rho_w, rho_g0, rho_w0, mug, muw, cr, cg, cw, u, p, h, p0, h0, du, bcs, sigma, v_p, v_h, k, krg, krw):
    s_g_ = Function(ME1)
    s_g_.interpolate(s(Sources_g, t, ME1, sigma))
    
    L_p = phi*((cr*rho_g(p)+cg*rho_g0)*sg*h*(p-p0) + rho_g(p)*sg*(h-true_h(H, h0)))*v_p*dx \
    + dt*rho_g(p)*k*krg/mug(p)*h*inner(nabla_grad(p) - rho_g0*g*nabla_grad(h), nabla_grad(v_p))*dx \
    - dt*s_g_*v_p*dx

    L_h = phi*((cr*rho_w(p) + rho_w0*cw)*(H-sg*h)*(p-p0) - rho_w(p)*sg*(h-true_h(H, h0)))*v_h*dx \
    + dt*rho_w(p)*k*krw/muw(p)*(H-h)*inner(nabla_grad(p) - rho_w0*g*nabla_grad(h), nabla_grad(v_h))*dx \

    L = L_h + L_p
    J = derivative(L, u, du) # Compute directional derivative about u in the direction of du (Jacobian)
    
    return(VFEModel(J, L, bcs))

########################################################################################################################
# Calculating theoritical injected (or extracted) mass. Positive for injected, negative for extracted

def injected_mass(t, Sources_g):
    amount = 0
    for source in Sources_g:
        New_inj_profile = []
        if len(source[2]) != 0:
            for i in range(len(source[2])):
                Inj_profile = source[2][i] ; rate = source[1][i]
                if Inj_profile[1]<= t: amount += rate*(Inj_profile[1]-Inj_profile[0])
                elif Inj_profile[0]<= t: amount += rate*(t-Inj_profile[0])
    return(amount)

########################################################################################################################
# Create functions for permeability, porosity, thickness and depth

def H_function(mesh, H_points, H_field, ME1):
    H = Function(ME1)
    interp = RBFInterpolator(H_points, H_field)
    H_data = interp(ME1.tabulate_dof_coordinates())
    H_moy = np.nanmean(H_data)
    for l in range(ME1.dim()):
        if np.isnan(H_data[l]): # if there is a NaN value, we give the precedent value to the vector.
            H.vector()[l] = H_moy
            print('Warning : NaN found after interpolation, replacing by mean value')
        else: H.vector()[l] = H_data[l]
    return(H)

def k_function(mesh, k_points, k_field, ME1):
    k = Function(ME1)
    interp = RBFInterpolator(k_points, k_field)
    k_data = interp(ME1.tabulate_dof_coordinates())
    k_moy = np.nanmean(k_data)
    for l in range(ME1.dim()):
        if np.isnan(k_data[l]): # if there is a NaN value, we give the precedent value to the vector.
            k.vector()[l] = k_moy
            print('Warning : NaN found after interpolation, replacing by mean value')
        else: k.vector()[l] = k_data[l]
    return(k)

def phi_function(mesh, phi_points, phi_field, ME1):
    phi = Function(ME1)
    interp = RBFInterpolator(phi_points, phi_field)
    phi_data = interp(ME1.tabulate_dof_coordinates())
    phi_moy = np.nanmean(phi_data)
    for l in range(ME1.dim()):
        if np.isnan(phi_data[l]): # if there is a NaN value, we give the precedent value to the vector.
            phi.vector()[l] = phi_moy
            print('Warning : NaN found after interpolation, replacing by mean value')
        else: phi.vector()[l] = phi_data[l]
    return(phi)

def depth_function(mesh, depth_points, depth_field, ME1):
    depth = Function(ME1)
    interp = RBFInterpolator(depth_points, depth_field)
    depth_data = interp(ME1.tabulate_dof_coordinates())
    depth_moy = np.nanmean(depth_data)
    for l in range(ME1.dim()):
        if np.isnan(depth_data[l]): # if there is a NaN value, we give the precedent value to the vector.
            depth.vector()[l] = depth_moy
            print('Warning : NaN found after interpolation, replacing by mean value')
        else: depth.vector()[l] = depth_data[l]
    return(depth)

########################################################################################################################
# Create list of average pressure for each time step (for mechanical model). List is exported in folder.
# Nt, H, Nz, rho_g, rho_w, sg, g must be provided

def extract_avg_pressure_list(Nt, H, Nz, rho_g, rho_w, sg, g, folder):
    H_max = np.amax(H.vector().get_local())
    pressure_list = []
    h_list = []
    avg_pressure_list = []
    reader_p = vtk.vtkXMLUnstructuredGridReader()
    reader_p.SetFileName(file_name_p(0))
    reader_p.Update()
    points = vtk_to_numpy(reader_p.GetOutput().GetPoints().GetData()).tolist()
    H_data = H.compute_vertex_values().tolist()
    for i in range(Nt):
        reader_p.SetFileName(file_name_p(i))
        reader_p.Update()  # Needed because of GetScalarRange
        pressure_int = vtk_to_numpy(reader_p.GetOutput().GetPointData().GetArray(0)).tolist()
        pressure_list.append(pressure_int)
        reader_h = vtk.vtkXMLUnstructuredGridReader()
        reader_h.SetFileName(file_name_h(i))
        reader_h.Update()  # Needed because of GetScalarRange
        h = vtk_to_numpy(reader_h.GetOutput().GetPointData().GetArray(0)).tolist()
        h_list.append(h)
    for i in tqdm(range(Nt)):
        dz = H_data[i]/Nz
        avg_pressure = []
        for j in range(len(points)):
            integral = 0
            p_int = pressure_list[i][j]
            z = H_max-h_list[i][j] ; p = p_int
            integral += p*dz
            z += dz
            while z<H_max: # we start from the interface
                rho_avg = sg*rho_g(p) + (1-sg)*rho_w(p)
                p -= rho_avg*g*dz
                integral += p*dz  
                z += dz
            z = H_max-h_list[i][j] ; p = p_int
            z -= dz
            while z>H_max-H_data[i]-dz: # we start from the interface
                p += rho_w(p)*g*dz
                integral += p*dz
                z -= dz
            avg_pressure.append(integral/H_data[i])
        avg_pressure_list.append(avg_pressure)

    np.save(folder + 'avg_pressure_list.npy',np.array(avg_pressure_list))