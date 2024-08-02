import numpy as np
import pandas as pd
from fenics import *
from scipy import signal
from mshr import *

class DiffusionModel:
    """A class that computes the pressure depletion inside a reservoir given
    its physical properties (boundaries,thickness,permeability,porosity,
    temperature,initial pressure), the gas compositon (or Molar Weight)
    and the spatial and temporal parameters (grid,timestep and
    extracted volume)"""

    def __init__(self,
                 x: np.ndarray,
                 y: np.ndarray,
                 thickness: np.ndarray,
                 temperature: int,
                 initial_pressure: float,
                 gas_molecular_weight: float,
                 num_steps: int,
                 time_step: float,
                 extraction_data: pd.DataFrame,
                 wells: np.ndarray,
                 outline: pd.DataFrame,
                 p_new_init: np.ndarray = None,
                 begin: int = 0,
                 name: str = 'DiffusionModel'):
        """
        Initializes the reservoir model with invariant parameters.
        """

        # Model name and settings:
        self.name = name
        self.begin = begin

        # Input data:
        self.x_meshgrid = x
        self.y_meshgrid = y
        self.outline = outline
        self.reservoir_temperature = temperature  # In Kelvin
        self.gas_molecular_weight = gas_molecular_weight  # In kg/mol
        self.pressure_new_init = p_new_init
        self.extraction_data = extraction_data
        self.wells = wells

        # Mesh dimensions:
        self.dim_x = self.x_meshgrid.shape[0] - 1
        self.dim_y = self.x_meshgrid.shape[1] - 1

        # Time parameters:
        self.step_number = num_steps  # Number of time steps in months
        self.d_t = time_step  # Duration of a time step in seconds

        # Derived constants:
        self.d_rho = (self.gas_molecular_weight / (8.314 * self.reservoir_temperature)) * 1e6 / 3600**2  # dρ/dp
        self.rho_std_condition = self.rho(101325 * 1e-6, 293.25)  # Density in kg/m³

        # Create rectangular mesh for thickness:
        self.mesh = RectangleMesh(Point(self.x_meshgrid[0, 0], self.y_meshgrid[0, 0]),
                                  Point(self.x_meshgrid[-1, 0], self.y_meshgrid[0, -1]),
                                  self.dim_x, self.dim_y)
        self.w = FunctionSpace(self.mesh, 'CG', 1)
        self.coordinates = self.mesh.coordinates()
        self.intermediate_thickness = self.array_2d_to_fenics(thickness, 1e-6)

        # Create triangle mesh for flexibility:
        mesh_x = signal.resample(self.outline[2], 500)[::-1]
        mesh_y = signal.resample(self.outline[3], 500)[::-1]
        structure = [Point(x, y) for x, y in zip(mesh_x, mesh_y)]
        domain = Polygon(structure)
        self.mesh2 = generate_mesh(domain, 40)
        self.v = FunctionSpace(self.mesh2, 'CG', 1)
        self.coordinates2_ = self.mesh2.coordinates()

        # Interpolate thickness onto triangle mesh:
        self.thickness = Function(self.v)
        self.thickness.interpolate(self.intermediate_thickness)

        # Define initial pressure on the mesh:
        self.initial_pressure = interpolate(Constant(initial_pressure), self.v)


    def array_2d_to_fenics(self, numpy_2d_array: np.ndarray, error_avoir_number: float = None) -> Function:
        """
        Converts a 2D NumPy array to a Fenics array, handling potential errors and adjusting for mesh layout.

        Args:
            numpy_2d_array: The 2D NumPy array to convert.
            error_avoir_number: An optional value to add to all elements of the array for error mitigation.

        Returns:
            The converted Fenics array.
        """

        # Handle potential errors:
        numpy_2d_array = np.nan_to_num(numpy_2d_array)  # Replace NaNs with numerical values

        # Transpose the array to match Fenics mesh layout:
        numpy_2d_array = numpy_2d_array.T

        # Flatten the array for assignment to Fenics function:
        numpy_array = numpy_2d_array.reshape((self.dim_x + 1) * (self.dim_y + 1))

        # Add error mitigation value if specified:
        if error_avoir_number is not None:
            numpy_array = numpy_array + np.ones_like(numpy_array) * error_avoir_number

        # Create the Fenics function and assign values:
        fenics_array = Function(self.w)
        fenics_array.vector()[:] = numpy_array[dof_to_vertex_map(self.w)]

        return fenics_array

    def array_1d_to_fenics(self, numpy_1d_array: np.ndarray, error_avoid_number: float = None) -> Function:
        """
        Converts a 1D NumPy array to a Fenics array, optionally adding a value to all elements for error mitigation.

        Args:
            numpy_1d_array: The 1D NumPy array to convert.
            error_avoid_number: An optional value to add to all elements of the array for error mitigation.

        Returns:
            The converted Fenics array.
        """

        # Add error mitigation value if specified:
        if error_avoid_number is not None:
            numpy_1d_array = numpy_1d_array + np.ones_like(numpy_1d_array) * error_avoid_number

        # Create the Fenics function and assign values:
        fenics_array = Function(self.v)
        fenics_array.vector()[:] = numpy_1d_array[dof_to_vertex_map(self.v)]

        return fenics_array

    
    def rho(self,
            p,
            temperature: float = None):
        """Function that computes the density of a gas given the molecular
        weight of the gas, its pressure, and its temperature
        If no temperature is given, Temperature of Reservoir is by default
        Pressure given in MPa, result in kg.km-3"""
        if temperature is None:
            return p*1e6 * self.gas_molecular_weight / \
                (8.314 * self.reservoir_temperature) *1e9#dim change change2
    
        return p*1e6 * self.gas_molecular_weight / (8.314 * temperature) *1e9 #dim change change2

    def viscosity(self, p):
        """Compute the viscosity of a gas given its density.
        Empirical Formula from Lee–Gonzalez Semiempirical Method (1966)

        !!!!!!!Constants are set for Groningen parameters, they need
        to be changed if another case study!!!!!!!!"""
        vertex_values = p.compute_vertex_values(self.mesh2)
        rho_vertex = self.rho(vertex_values) / 1000 / 1e9 #pressure change : *(1e9*3600**2), change2

        viscosity = 137.071e-4 * np.exp(5.094 * rho_vertex **1.314) / 1000 * 1000*3600 #dim change 1e9 et 1000*3600
        fenics_viscosity = self.array_1d_to_fenics(viscosity)
        return fenics_viscosity
    
    def distance(self, array: np.ndarray, x: float, y: float):
        """
        Determines the index, distance, and coordinates of the point in the given array that is closest to a target point.
        Args:
            array: A 2D NumPy array of coordinates, where each row represents a point.
            x: The x-coordinate of the target point.
            y: The y-coordinate of the target point.
        Returns:
                - The index of the closest point in the array.
                - The distance between the target point and the closest point.
                - The x-coordinate of the closest point.
                - The y-coordinate of the closest point.
        """
        closest_index = None
        closest_distance = float('inf')  # Initialize with positive infinity
        closest_x = None
        closest_y = None

        for i, (point_x, point_y) in enumerate(array):
            distance_to_point = np.sqrt((x - point_x)**2 + (y - point_y)**2)
            if distance_to_point < closest_distance:
                closest_index = i
                closest_distance = distance_to_point
                closest_x = point_x
                closest_y = point_y

        return closest_index, closest_distance, closest_x, closest_y


    def well_pressure_difference(self,
                                 measurements_data: pd.DataFrame):
        """
        Calculates the total absolute difference between measured well pressures and calculated pressures
        at the closest points on the grid, as well as the number of valid measurements used.

        Args:
            measurements_data: A pandas DataFrame containing well pressure measurements.

        Returns:
            A tuple containing:
                - The total absolute well pressure difference.
                - The number of valid measurements used in the calculation.
        """

        # Find the indices of the closest grid points to each well:
        location = np.zeros(self.wells.shape[0])
        for k in range(self.wells.shape[0]):
            location[k] = self.distance(self.coordinates2_, int(Well[k][1]), int(Well[k][2]))[0]

        # Initialize variables for calculating the difference:
        well_difference = 0
        well_nb_measurements = 0

        # Determine the maximum number of time steps to consider:
        IHM = min(self.step_number, len(measurements_data))

        # Iterate through time steps and wells:
        for T in range(self.begin, IHM):
            for i in range(self.wells.shape[0]):
                # Check if the measurement is valid:
                if np.isfinite(measurements_data[self.wells[i][0]][T]):
                    # Calculate the absolute difference between measured and calculated pressures:
                    difference = abs(
                        measurements_data[self.wells[i][0]][T] - self.PArray_[int(location[i]), T]
                    )
                    well_difference += difference
                    well_nb_measurements += 1

        return well_difference, well_nb_measurements

    def diffusion_process_for_control(self,
                                        permeability: float,
                                        porosity: float,
                                        gassat: float,
                                        instantaneous_pressure: np.ndarray,
                                        instantaneous_extraction_data: np.ndarray):
        """
        Computes the diffusion model for a single iteration, given specific parameters
        and instantaneous pressure and extraction data. Returns the pressure field at time t+1.
        """

        # Initialize pressure array:
        self.PArray_ = np.zeros((self.coordinates2_.shape[0], self.step_number))

        # Set parameters and trial/test functions:
        permeability_fenics = Constant(permeability)
        self.permeability_ = interpolate(permeability_fenics, self.v)
        p = TrialFunction(self.v)
        p0i = self.array_1d_to_fenics(instantaneous_pressure)
        p0 = interpolate(p0i, self.v)
        v = TestFunction(self.v)

        # Define variational problem for pressure:
        a = ((porosity * self.d_rho * self.thickness) * p * (1e9 * 3600**2) * v * dx
             + (self.d_t * self.thickness * self.permeability_
                / self.viscosity(p0)) * dot(self.rho(p0) * grad(p * (1e9 * 3600**2)), grad(v)) * dx)
        L = ((porosity * self.d_rho * self.thickness) * p0 * (1e9 * 3600**2) * v * dx)

        # Define point sources for extraction:
        point_sources = []
        for i in range(self.wells.shape[0]):
            extraction_rate = -instantaneous_extraction_data[self.wells[i][0]]  # Convert to positive
            point_sources.append((Point(float(self.wells[i][1]), float(self.wells[i][2])),
                                  extraction_rate * self.d_t * self.rho_std_condition / gassat * 1e-9))
        ps = PointSource(self.v, point_sources)

        # Assemble and solve the system:
        b = assemble(L)
        A = assemble(a)
        ps.apply(b)
        self.p1_ = Function(self.v)
        self.p1_.assign(p0)
        solver = KrylovSolver('minres', 'hypre_euclid')
        solver.parameters["maximum_iterations"] = 1000
        solver.parameters["error_on_nonconvergence"] = False  # Set to True for debugging
        solver.solve(A, self.p1_.vector(), b)

        # Extract vertex values and return:
        self.vertex_values_P_ = self.p1_.compute_vertex_values(self.mesh2)
        return self.vertex_values_P_

