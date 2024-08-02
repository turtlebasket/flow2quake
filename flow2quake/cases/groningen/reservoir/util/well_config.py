import numpy as np
import pandas as pd

def create_well_configuration(well_configuration, Well, GasData):
    """
    Creates well configuration data based on the specified configuration type.

    Args:
        well_configuration (int): The type of well configuration to create.
        Well (np.ndarray): Array containing well data for Groningen real wells.
        GasData (pd.DataFrame): DataFrame containing gas data for Groningen wells.

    Returns:
        tuple: (wells_names, wells_locations, Well, initial_extraction_data)
    """

    wells_list = []  # List to store well information

    if well_configuration == 1:  # Unique well
        wells_names = ['Unique well']
        wells_locations = np.array([[252.500, 587.500]])
        initial_extraction_rate = [4e8 / (30.4375 * 24)]  # m3/hour

    elif well_configuration == 2:  # Two wells (north and south)
        wells_names = ['Extraction well', 'Injection well']
        wells_locations = np.array([[260.000, 575.000], [245.000, 600.000]])
        initial_extraction_rate = [4e8 / (30.4375 * 24), -4e8 / (30.4375 * 24)]

    elif well_configuration == 5:  # Five wells (central and surrounding)
        wells_names = ['Constant well', 'Variable well up_right', 'Variable well up_left',
                       'Variable well down_left', 'Variable well down_right']
        wells_locations = np.array([[252.500, 587.500],
                                    [257.500, 592.500], [247.500, 592.500],
                                    [247.500, 582.500], [257.500, 582.500]])
        initial_extraction_rate = [8e8 / (30.4375 * 24)] * 5

    elif well_configuration == 29:  # Groningen real wells
        wells_names = [Well[i, 0] for i in range(Well.shape[0])]
        wells_locations = np.array([[float(Well[i, 1]) / 1000, float(Well[i, 2]) / 1000]
                                    for i in range(Well.shape[0])])
        historic_extraction = np.nan_to_num(GasData['AverageWinter'].fillna(0).rolling(1).mean().to_numpy()) / (30.4375 * 24)
        initial_extraction_rate = historic_extraction[300, :].tolist()  # Extraction rate of a random month

    else:
        print('Wrong well_configuration')
        return None

    # Create the Well array directly using list comprehension
    Well = np.array([[name, str(loc[0]), str(loc[1])] for name, loc in zip(wells_names, wells_locations)])

    initial_extraction_data = pd.Series(data=initial_extraction_rate, index=wells_names)

    return wells_names, wells_locations, Well, initial_extraction_data