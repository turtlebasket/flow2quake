import vtkmodules as vtk
from vtkmodules.util.numpy_support import vtk_to_numpy
import numpy as np
import matplotlib.pyplot as plt
from colorama import Fore, Back, Style
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
import time
from CoolProp.CoolProp import PropsSI
from auxiliary_classes import *
from auxiliary_functions import *


########################################################################################################################
# plot the Area Of Interest
def plot_AOI(ax, AOI, color):
    ax.plot(np.array(AOI)[:, 0], np.array(AOI)[:, 1], color=color, linewidth=3)


########################################################################################################################
# plot the pressure at the interface for a given iteration
# Nx, Ny = number of cells used for the interpolation
# AOI, p_min, p_max, List_t, ME1 must be provided


def plot_p_int(iteration, Nx, Ny, AOI, p_min, p_max, List_t, ME1):
    points = extract_points_p_h(iteration)[0]
    pressure = extract_points_p_h(iteration)[1]
    points = [[points[i][0], points[i][1]] for i in range(len(points))]

    interp = RBFInterpolator(points, pressure)
    Data = interp(ME1.tabulate_dof_coordinates())

    P = Function(ME1)
    P_moy = np.nanmean(Data)  # Average on the 2D domain
    for l in range(ME1.dim()):
        if np.isnan(
            Data[l]
        ):  # if there is a NaN value, we give the precedent value to the vector.
            P.vector()[l] = P_moy
            print("Warning : NaN found after interpolation, replacing by mean value")
        else:
            P.vector()[l] = Data[l]

    print(
        "Maximum pressure at the interface at t = "
        + day_format(List_t[iteration])
        + " : "
        + str(np.amax(pressure))
        + " bar"
    )
    print("After interpolation : " + str(np.amax(Data)) + " bar")
    fig = plt.figure(figsize=(10, 10))
    plt.colorbar(
        plot(P, mode="color", cmap="jet", vmin=p_min, vmax=p_max),
        orientation="vertical",
    )
    plt.title("Pressure at the interface (bar) at t = " + day_format(List_t[iteration]))
    ax = plt.gca()
    plot_AOI(ax, np.array(AOI), "red")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Distance (m)")


########################################################################################################################
# plot the h for a given iteration
# Nx, Ny = number of cells used for the interpolation
# AOI, h_min, h_max, List_t, ME1 must be provided


def plot_h(iteration, Nx, Ny, AOI, h_min, h_max, List_t, ME1):
    points = extract_points_p_h(iteration)[0]
    h = extract_points_p_h(iteration)[2]
    points = [[points[i][0], points[i][1]] for i in range(len(points))]

    interp = RBFInterpolator(points, h)
    Data = interp(ME1.tabulate_dof_coordinates())

    h_function = Function(ME1)
    h_moy = np.nanmean(Data)  # Average on the 2D domain
    for l in range(ME1.dim()):
        if np.isnan(
            Data[l]
        ):  # if there is a NaN value, we give the precedent value to the vector.
            h_function.vector()[l] = h_moy
            print("Warning : NaN found after interpolation, replacing by mean value")
        else:
            h_function.vector()[l] = Data[l]

    print(
        "Maximum pressure at the interface at t = "
        + day_format(List_t[iteration])
        + " : "
        + str(np.amax(h))
        + " bar"
    )
    print("After interpolation : " + str(np.amax(Data)) + " bar")
    fig = plt.figure(figsize=(10, 10))
    plt.colorbar(
        plot(h_function, mode="color", cmap="YlGnBu", vmin=h_min, vmax=h_max),
        orientation="vertical",
    )
    plt.title("Thickness of gas (m) at t = " + day_format(List_t[iteration]))
    ax = plt.gca()
    plot_AOI(ax, np.array(AOI), "red")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Distance (m)")
    # ax.set_xlim([6.4e4, 6.6e4]) ; ax.set_ylim([8.6e4, 8.8e4])


########################################################################################################################
# average pressure (along the vertical) in the reservoir at a given iteration
# Nx, Ny, Nz = number of cells used for the interpolation
# List_t, H, rho_g, rho_w, g, sg, p_min, p_max, AOI, ME1


def plot_p_avg(
    iteration, Nx, Ny, Nz, List_t, H, rho_g, rho_w, g, sg, p_min, p_max, AOI, ME1
):
    H_max = np.amax(H.vector().get_local())
    points_init, pressure_int, h = extract_points_p_h(iteration)
    points = []
    P_avg_list = []
    H_data = H.compute_vertex_values().tolist()

    print("Calculating vertical pressure in the reservoir...")
    for i in tqdm(range(len(points_init))):
        dz = H_data[i] / Nz
        integral = 0
        points.append([points_init[i][0], points_init[i][1]])
        p_int = pressure_int[i]
        z = H_max - h[i]
        p = p_int
        integral += p * dz
        z += dz
        while z < H_max:  # we start from the interface
            rho_avg = sg * rho_g(p) + (1 - sg) * rho_w(p)
            p -= 1e-5 * rho_avg * g * dz
            integral += p * dz
            z += dz
        z = H_max - h[i]
        p = p_int
        z -= dz
        while z > H_max - H_data[i]:  # we start from the interface
            p += 1e-5 * rho_w(p) * g * dz
            integral += p * dz
            z -= dz
        P_avg_list.append(integral / H_data[i])

    interp = RBFInterpolator(points, P_avg_list)
    Data = interp(ME1.tabulate_dof_coordinates())
    P_avg_function = Function(ME1)
    P_avg_moy = np.nanmean(Data)  # Average on the 2D domain
    for l in range(ME1.dim()):
        if np.isnan(
            Data[l]
        ):  # if there is a NaN value, we give the precedent value to the vector.
            P_avg_function.vector()[l] = P_avg_moy
            print("Warning : NaN found after interpolation, replacing by mean value")
        else:
            P_avg_function.vector()[l] = Data[l]

    print(
        "Maximum average pressure at t = "
        + day_format(List_t[iteration])
        + " : "
        + str(np.amax(P_avg_list))
        + " bar"
    )
    print("After interpolation : " + str(np.amax(Data)) + " bar")
    fig = plt.figure(figsize=(10, 10))
    plt.colorbar(
        plot(P_avg_function, mode="color", cmap="jet", vmin=p_min, vmax=p_max),
        orientation="vertical",
    )
    plt.title("Average pressure (bar) at t = " + day_format(List_t[iteration]))
    ax = plt.gca()
    plot_AOI(ax, np.array(AOI), "red")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Distance (m)")


########################################################################################################################
# plot the pressure at the interface for several iterations
# N = number of plots desired
# Nx, Ny = number of cells used for the interpolation
# Nt, p_min, p_max, List_t, AOI, ME1 must be provided


def multiplot_p_int(N, Nt, Nx, Ny, p_min, p_max, List_t, AOI, ME1):
    if np.sqrt(N) - int(np.sqrt(N)) < 1e-14:
        Nc = int(np.sqrt(N))
    else:
        Nc = int(np.sqrt(N)) + 1

    if N > Nt:
        print(Fore.RED + "Number of plots superior to number of iterations")
        return ()
    elif N != 1:
        n = (Nt - 1) // (N - 1)  # number of steps between 2 consecutive plots
    else:
        n = Nt
    Nl = Nc - (Nc**2 - N) // Nc

    fig_p, axs = plt.subplots(Nl, Nc, gridspec_kw={"wspace": 0, "hspace": 0.2})
    fig_p.suptitle("Pressure at the interface (bar)")
    fig_p.set_size_inches([13 * Nc / Nl, 14])

    num_plot = 1
    k = 0
    while num_plot <= N:
        if k % n == 0:
            points = extract_points_p_h(k)[0]
            pressure = extract_points_p_h(k)[1]
            points = [[points[i][0], points[i][1]] for i in range(len(points))]

            interp = RBFInterpolator(points, pressure)
            Data = interp(ME1.tabulate_dof_coordinates())

            P = Function(ME1)
            P_moy = np.nanmean(Data)  # Average on the 2D domain
            for l in range(ME1.dim()):
                if np.isnan(
                    Data[l]
                ):  # if there is a NaN value, we give the precedent value to the vector.
                    P.vector()[l] = P_moy
                    print(
                        "Warning : NaN found after interpolation, replacing by mean value"
                    )
                else:
                    P.vector()[l] = Data[l]

            plt.subplot(Nl, Nc, num_plot)
            ax = plt.gca()
            im = plot(P, mode="color", cmap="jet", vmin=p_min, vmax=p_max)
            plot_AOI(ax, AOI, "red")
            plt.title(day_format(List_t[k]))
            if Nl != 1 or Nc != 1:
                for ax in axs.flat:
                    ax.set(xlabel="Distance (m)", ylabel="Distance (m)")
                for ax in axs.flat:
                    ax.label_outer()
            num_plot += 1
        k += 1
    cbax = fig_p.add_axes([0.93, 0.15, 0.02, 0.7])
    fig_p.colorbar(im, cax=cbax)


########################################################################################################################
# plot h for several iterations
# N = number of plots desired
# Nx, Ny = number of cells used for the interpolation
# Nt, h_min, h_max, List_t, AOI, ME1 must be provided


def multiplot_h(N, Nt, Nx, Ny, h_min, h_max, List_t, AOI, ME1):
    if np.sqrt(N) - int(np.sqrt(N)) < 1e-14:
        Nc = int(np.sqrt(N))
    else:
        Nc = int(np.sqrt(N)) + 1

    if N > Nt:
        print(Fore.RED + "Number of plots superior to number of iterations")
        return ()
    elif N != 1:
        n = (Nt - 1) // (N - 1)  # number of steps between 2 consecutive plots
    else:
        n = Nt
    Nl = Nc - (Nc**2 - N) // Nc

    fig_h, axs = plt.subplots(Nl, Nc, gridspec_kw={"wspace": 0, "hspace": 0.2})
    fig_h.suptitle("Thickness of the gas layer (m)")
    fig_h.set_size_inches([13 * Nc / Nl, 14])

    num_plot = 1
    k = 0
    while num_plot <= N:
        if k % n == 0:
            points = extract_points_p_h(k)[0]
            h = extract_points_p_h(k)[2]
            points = [[points[i][0], points[i][1]] for i in range(len(points))]

            interp = RBFInterpolator(points, h)
            Data = interp(ME1.tabulate_dof_coordinates())

            h_function = Function(ME1)
            h_moy = np.nanmean(Data)  # Average on the 2D domain
            for l in range(ME1.dim()):
                if np.isnan(
                    Data[l]
                ):  # if there is a NaN value, we give the precedent value to the vector.
                    h_function.vector()[l] = h_moy
                    print(
                        "Warning : NaN found after interpolation, replacing by mean value"
                    )
                else:
                    h_function.vector()[l] = Data[l]

            plt.subplot(Nl, Nc, num_plot)
            ax = plt.gca()
            im = plot(h_function, mode="color", cmap="YlGnBu", vmin=h_min, vmax=h_max)
            plot_AOI(ax, AOI, "red")
            plt.title(day_format(List_t[k]))
            if Nl != 1 or Nc != 1:
                for ax in axs.flat:
                    ax.set(xlabel="Distance (m)", ylabel="Distance (m)")
                for ax in axs.flat:
                    ax.label_outer()
            num_plot += 1
        k += 1
    cb_ax_h = fig_h.add_axes([0.93, 0.15, 0.02, 0.7])
    fig_h.colorbar(im, cax=cb_ax_h)


########################################################################################################################
# slice plot of the pressure and h for a given iteration at the x_pos position
# Ny, Nz = number of cells used for the interpolation (and the vertical integration of the pressure)
# List_t, H, rho_g, rho_w, g, sg, p_min, p_max, ymin, ymax, zmin, zmax, Outline must be provided


def xslice_plot_p_h(
    x_pos,
    iteration,
    Ny,
    Nz,
    List_t,
    H,
    rho_g,
    rho_w,
    g,
    sg,
    p_min,
    p_max,
    ymin,
    ymax,
    zmin,
    zmax,
    Outline,
):
    H_max = np.nanmax(H.vector())
    points_init, pressure_int, h_init = extract_points_p_h(iteration)
    points1 = []
    pressure1 = []
    h1 = []
    H_data1 = []
    H_data_init = H.compute_vertex_values().tolist()
    for i in range(len(points_init)):
        if abs(points_init[i][0] - x_pos) < (Outline[1][0] - Outline[0][0]) / 10:
            points1.append(points_init[i])
            pressure1.append(pressure_int[i])
            h1.append(h_init[i])
            H_data1.append(H_data_init[i])
    points2 = []
    pressure2 = []
    Y = np.linspace(
        np.nanmin(np.array(points1)[:, 1]), np.nanmax(np.array(points1)[:, 1]), int(Ny)
    )
    Z = np.linspace(0, H_max, int(Nz))
    print("Calculating vertical pressure in the reservoir...")
    for y in tqdm(Y):
        min = 8e4
        index = 0
        for i in range(len(points1)):
            err = np.sqrt((x_pos - points1[i][0]) ** 2 + (y - points1[i][1]) ** 2)
            if err < min:
                min = err
                index = i
        z = H_max - h1[index]
        dz = H_data1[index] / Nz
        points2.append([points1[index][1], H_max - z])
        p_int = pressure1[index]
        pressure2.append(p_int)
        p = p_int
        z += dz
        while z <= H_max:  # we start from the interface
            points2.append([points1[index][1], H_max - z])
            rho_avg = sg * rho_g(p) + (1 - sg) * rho_w(p)
            pressure2.append(p - 1e-5 * rho_avg * g * dz)
            p -= 1e-5 * rho_avg * g * dz
            z += dz
        z = H_max - h1[index] - dz
        p = p_int
        while z >= H_max - H_data1[index]:  # we start from the interface
            points2.append([points1[index][1], H_max - z])
            pressure2.append(p + 1e-5 * rho_w(p) * g * dz)
            p += 1e-5 * rho_w(p) * g * dz
            z -= dz
        while z >= 0:
            points2.append([points1[index][1], H_max - z])
            pressure2.append(math.nan)
            z -= dz

    max_x = np.amax(np.array(points1)[:, 0])
    min_x = np.amin(np.array(points1)[:, 0])
    X_h = np.linspace(min_x, max_x, int(Ny))
    Y_h = np.linspace(Outline[0][1], Outline[2][1], int(Ny))
    print("Interpolation...")
    grid_y, grid_z = np.meshgrid(Y, Z)
    grid_x_h, grid_y_h = np.meshgrid(X_h, Y_h)
    grid_h = griddata(
        np.array(points1)[:, :2], h1, (grid_x_h, grid_y_h), method="cubic"
    )  # interpolate the data
    grid_H = griddata(
        np.array(points1)[:, :2], H_data1, (grid_x_h, grid_y_h), method="cubic"
    )  # interpolate the data
    grid_p = griddata(
        np.array(points2), pressure2, (grid_y, grid_z), method="linear"
    )  # interpolate the data
    h_int = np.array(grid_h)[:, int(Ny * (x_pos - min_x) / (max_x - min_x))]
    H_plot = np.array(grid_H)[:, int(Ny * (x_pos - min_x) / (max_x - min_x))]
    P = np.array(grid_p)
    print(
        "Maximum pressure at x = "
        + str(x_pos)
        + " m"
        + " at t = "
        + day_format(List_t[iteration])
        + " : "
        + str(np.nanmax(pressure2))
        + " bar"
    )
    print("After interpolation : " + str(np.nanmax(P)) + " bar")
    print(
        "Maximum thickness at x = "
        + str(x_pos)
        + " m"
        + " at t = "
        + day_format(List_t[iteration])
        + " : "
        + str(np.nanmax(h1))
        + " m"
    )
    print("After interpolation : " + str(np.nanmax(h_int)) + " m")

    fig, ax_left = plt.subplots(figsize=[15, 4])
    ax_right = ax_left.twinx()
    ax_left.set_xlabel("Distance in m")
    ax_left.set_ylabel("Depth in m")
    fig.suptitle(
        "Pressure (bar) at t = "
        + day_format(List_t[iteration])
        + " at x = "
        + str(x_pos)
        + " m"
    )
    ax_left.set_ylim([zmax, zmin])
    cbax = fig.add_axes([0.93, 0.12, 0.02, 0.76])
    plt.colorbar(
        ax_left.pcolormesh(Y, Z, P, cmap="jet", shading="auto", vmin=p_min, vmax=p_max),
        ax=ax_left,
        cax=cbax,
    )
    ax_right.plot(Y_h, h_int, color="black", linewidth=1)
    ax_right.plot(Y_h, H_plot, color="black", linewidth=2)
    ax_right.set_ylim([H_max - zmin, H_max - zmax])
    ax_right.get_yaxis().set_visible(False)
    ax_right.set_xlim([ymin, ymax])
    ax_left.set_xlim([ymin, ymax])


########################################################################################################################
# slice plot of the pressure and h for a given iteration at the y_pos position
# Ny, Nz = number of cells used for the interpolation (and the vertical integration of the pressure)
# List_t, H, rho_g, rho_w, g, p_min, p_max, xmin, xmax, ymin, ymax, Outline must be provided


def yslice_plot_p_h(
    y_pos,
    iteration,
    Nx,
    Nz,
    List_t,
    H,
    rho_g,
    rho_w,
    g,
    sg,
    p_min,
    p_max,
    xmin,
    xmax,
    zmin,
    zmax,
    Outline,
):
    H_max = np.nanmax(H.vector())
    points_init, pressure_int, h_init = extract_points_p_h(iteration)
    points1 = []
    pressure1 = []
    h1 = []
    H_data1 = []
    H_data_init = H.compute_vertex_values().tolist()
    for i in range(len(points_init)):
        if abs(points_init[i][1] - y_pos) < (Outline[2][1] - Outline[0][1]) / 10:
            points1.append(points_init[i])
            pressure1.append(pressure_int[i])
            h1.append(h_init[i])
            H_data1.append(H_data_init[i])
    points2 = []
    pressure2 = []
    X = np.linspace(
        np.nanmin(np.array(points1)[:, 0]), np.nanmax(np.array(points1)[:, 0]), int(Nx)
    )
    Z = np.linspace(0, H_max, int(Nz))
    print("Calculating vertical pressure in the reservoir...")
    for x in tqdm(X):
        min = 8e4
        index = 0
        for i in range(len(points1)):
            err = np.sqrt((x - points1[i][0]) ** 2 + (y_pos - points1[i][1]) ** 2)
            if err < min:
                min = err
                index = i
        dz = H_data1[index] / Nz
        z = H_max - h1[index]
        points2.append([points1[index][0], H_max - z])
        p_int = pressure1[index]
        pressure2.append(p_int)
        p = p_int
        z += dz
        while z < H_max:  # we start from the interface
            points2.append([points1[index][0], H_max - z])
            rho_avg = sg * rho_g(p) + (1 - sg) * rho_w(p)
            pressure2.append(p - 1e-5 * rho_avg * g * dz)
            p -= 1e-5 * rho_avg * g * dz
            z += dz
        z = H_max - h1[index] - dz
        p = p_int
        while z >= H_max - H_data1[index]:  # we start from the interface
            points2.append([points1[index][0], H_max - z])
            pressure2.append(p + 1e-5 * rho_w(p) * g * dz)
            p += 1e-5 * rho_w(p) * g * dz
            z -= dz
        while z >= 0:
            points2.append([points1[index][0], H_max - z])
            pressure2.append(math.nan)
            z -= dz

    max_y = np.amax(np.array(points1)[:, 1])
    min_y = np.amin(np.array(points1)[:, 1])
    X_h = np.linspace(Outline[0][0], Outline[1][0], int(Nx))
    Y_h = np.linspace(min_y, max_y, int(Nx))
    print("Interpolation...")
    grid_x, grid_z = np.meshgrid(X, Z)
    grid_x_h, grid_y_h = np.meshgrid(X_h, Y_h)
    grid_h = griddata(
        np.array(points1)[:, :2], h1, (grid_x_h, grid_y_h), method="cubic"
    )  # interpolate the data
    grid_p = griddata(
        np.array(points2), pressure2, (grid_x, grid_z), method="linear"
    )  # interpolate the data
    grid_H = griddata(
        np.array(points1)[:, :2], H_data1, (grid_x_h, grid_y_h), method="cubic"
    )  # interpolate the data
    H_plot = np.array(grid_H)[int(Nx * (y_pos - min_y) / (max_y - min_y))]
    h_int = np.array(grid_h)[int(Nx * (y_pos - min_y) / (max_y - min_y))]
    P = np.array(grid_p)
    print(
        "Maximum pressure at y = "
        + str(y_pos)
        + " m"
        + " at t = "
        + day_format(List_t[iteration])
        + " : "
        + str(np.nanmax(pressure2))
        + " bar"
    )
    print("After interpolation : " + str(np.nanmax(P)) + " bar")
    print(
        "Maximum thickness at x = "
        + str(y_pos)
        + " m"
        + " at t = "
        + day_format(List_t[iteration])
        + " : "
        + str(np.nanmax(h1))
        + " m"
    )
    print("After interpolation : " + str(np.nanmax(h_int)) + " m")
    fig, ax_left = plt.subplots(figsize=[15, 4])
    ax_right = ax_left.twinx()
    ax_left.set_xlabel("Distance in m")
    ax_left.set_ylabel("Depth in m")
    fig.suptitle(
        "Pressure (bar) at t = "
        + day_format(List_t[iteration])
        + " at y = "
        + str(y_pos)
        + " m"
    )
    ax_left.set_ylim([zmax, zmin])
    cbax = fig.add_axes([0.93, 0.12, 0.02, 0.76])
    plt.colorbar(
        ax_left.pcolormesh(X, Z, P, cmap="jet", shading="auto", vmin=p_min, vmax=p_max),
        ax=ax_left,
        cax=cbax,
    )
    ax_right.plot(X_h, h_int, color="black", linewidth=1)
    ax_right.plot(X_h, H_plot, color="black", linewidth=2)
    ax_right.set_ylim([H_max - zmin, H_max - zmax])
    ax_right.get_yaxis().set_visible(False)
    ax_right.set_xlim([xmin, xmax])
    ax_left.set_xlim([xmin, xmax])
    ax_left.get_yaxis().set_visible(True)


########################################################################################################################
# slice plot of h for a given iteration at the x_pos position
# Ny, Nz = number of cells used for the interpolation
# ymin, ymax, zmin, zmax, List_t, Outline must be provided


def xslice_plot_h(x_pos, iteration, Ny, Nz, ymin, ymax, zmin, zmax, List_t, Outline):
    points_init = extract_points_p_h(iteration)[0]
    h_init = extract_points_p_h(iteration)[2]
    points1 = []
    h1 = []

    for i in range(len(points_init)):
        if abs(points_init[i][0] - x_pos) < ((Outline[1][0] - Outline[0][0])) / 10:
            points1.append(points_init[i])
            h1.append(h_init[i])

    max_x = np.amax(np.array(points1)[:, 0])
    min_x = np.amin(np.array(points1)[:, 0])
    X_h = np.linspace(min_x, max_x, int(Ny))
    Y_h = np.linspace(Outline[0][1], Outline[2][1], int(Ny))

    grid_x_h, grid_y_h = np.meshgrid(X_h, Y_h)
    print("Interpolation...")
    grid_h = griddata(
        np.array(points1)[:, :2], h1, (grid_x_h, grid_y_h), method="linear"
    )  # interpolate the data
    H_int = np.array(grid_h)[:, int(Ny * (x_pos - min_x) / (max_x - min_x))]
    fig, ax = plt.subplots(1, figsize=(14, 5))
    ax.set_title(
        "Interface at t = "
        + day_format(List_t[iteration])
        + " at x = "
        + str(x_pos)
        + " m"
    )
    ax.plot(Y_h, H_int, color="black", linewidth=1)
    ax.set_xlim([ymin, ymax])
    ax.set_ylim([zmax, zmin])


########################################################################################################################
# slice plot of h for a given iteration at the y_pos position
# Nx, Nz = number of cells used for the interpolation
# xmin, xmax, zmin, zmax, List_t, Outline must be provided


def yslice_plot_h(y_pos, iteration, Nx, Ny, xmin, xmax, zmin, zmax, List_t, Outline):
    points_init = extract_points_p_h(iteration)[0]
    h_init = extract_points_p_h(iteration)[2]
    points1 = []
    h1 = []

    for i in range(len(points_init)):
        if abs(points_init[i][1] - y_pos) < (Outline[2][1] - Outline[0][1]) / 10:
            points1.append(points_init[i])
            h1.append(h_init[i])

    max_y = np.amax(np.array(points1)[:, 1])
    min_y = np.amin(np.array(points1)[:, 1])
    Y_h = np.linspace(min_y, max_y, int(Nx))
    X_h = np.linspace(Outline[0][1], Outline[2][1], int(Nx))

    grid_x_h, grid_y_h = np.meshgrid(X_h, Y_h)
    print("Interpolation...")
    grid_h = griddata(
        np.array(points1)[:, :2], h1, (grid_x_h, grid_y_h), method="linear"
    )  # interpolate the data
    H_int = np.array(grid_h)[int(Nx * (y_pos - min_y) / (max_y - min_y))]
    fig, ax = plt.subplots(1, figsize=(14, 5))
    ax.set_title(
        "Interface at t = "
        + day_format(List_t[iteration])
        + " at y = "
        + str(y_pos)
        + " m"
    )
    ax.plot(X_h, H_int, color="black", linewidth=1)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([zmax, zmin])


########################################################################################################################
# plot temporal profile of a source
# T_tot must be provided


def plot_source(source, T_tot, ax):
    if len(source[2]) == 0:
        X = [0, T_tot]
        Y = [0, 0]
    else:
        X = [0]
        Y = [0]
        for i in range(len(source[2])):
            Inj_profile = source[2][i]
            rate = source[1][i]
            X.append(Inj_profile[0])
            X.append(Inj_profile[0])
            X.append(Inj_profile[1])
            X.append(Inj_profile[1])
            Y.append(0)
            Y.append(rate)
            Y.append(rate)
            Y.append(0)
        X.append(T_tot)
        Y.append(0)
    ax.plot(X, Y)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("Injection rate (kg.s-1)")


########################################################################################################################
# plot wells positions
# Sources_g must be provided


def plot_wells(Sources_g, ax):
    for source in Sources_g:
        ax.plot(
            source[0][0],
            source[0][1],
            marker="o",
            markersize=2,
            markeredgecolor="red",
            markerfacecolor="red",
        )


########################################################################################################################
# Plot viscosities and densities
# T_min, T_max, p_min, p_max, N, p_ref, p_extrem must be provided


def plot_rho_mu(T_min, T_max, p_min, p_max, N, p_ref, p_extrem):
    T_list = np.linspace(273.15 + T_min, 273.15 + T_max, N)
    P = np.linspace(p_min, p_max, 100)
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    Y = []
    for T in T_list:
        for p in P:
            Y.append(PropsSI("D", "T", T, "P", p, "CO2"))  # gas density
        ax = axs[0, 0]
        ax.plot(1e-5 * P, Y, label=str("T = " + str(T - 273.15) + " °C"))
        Y = []
        ax.legend()
        ax.set_ylabel("Density (kg.m-3)")
        ax.set_title("Gas")
        ax.vlines(
            x=[p_ref * 1e-5, p_extrem * 1e-5],
            ymin=200,
            ymax=1000,
            colors="black",
            ls="--",
            lw=2,
        )

        for p in P:
            Y.append(1e3 * PropsSI("V", "T", T, "P", p, "CO2"))  # gas viscosity
        ax = axs[1, 0]
        ax.plot(1e-5 * P, Y, label=str("T = " + str(T - 273.15) + " °C"))
        Y = []
        ax.legend()
        ax.set_xlabel("Pressure (bar)")
        ax.set_ylabel("Viscosity (kPa.s)")
        ax.vlines(
            x=[p_ref * 1e-5, p_extrem * 1e-5],
            ymin=0.02,
            ymax=0.12,
            colors="black",
            ls="--",
            lw=2,
        )

        for p in P:
            Y.append(PropsSI("D", "T", T, "P", p, "WATER"))  # water density
        ax = axs[0, 1]
        ax.plot(1e-5 * P, Y, label=str("T = " + str(T - 273.15) + " °C"))
        Y = []
        ax.legend()
        ax.set_title("Water")
        ax.vlines(
            x=[p_ref * 1e-5, p_extrem * 1e-5],
            ymin=975,
            ymax=1012,
            colors="black",
            ls="--",
            lw=2,
        )

        for p in P:
            Y.append(1e3 * PropsSI("V", "T", T, "P", p, "WATER"))  # water viscosity
        ax = axs[1, 1]
        ax.plot(1e-5 * P, Y, label=str("T = " + str(T - 273.15) + " °C"))
        Y = []
        ax.legend()
        ax.set_xlabel("Pressure (bar)")
        ax.vlines(
            x=[p_ref * 1e-5, p_extrem * 1e-5],
            ymin=0.3,
            ymax=1.05,
            colors="black",
            ls="--",
            lw=2,
        )
    plt.show()
