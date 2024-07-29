import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def UpdateGasExtractionData(
    GasData_file="./Data/WellInformation.npy",
    winter="ColdWinter",
    Excel_file="../../Workshop2021/UpdatedData/Gaswinning-maandelijks_2010_2022.xlsx",
    Extrapolate_increase=False,
    plot_YN=False):
    """
    Updates gas extraction data from an Excel file and optionally extrapolates values.

    Args:
        GasData_file (str): Path to the existing gas data file.
        winter (str): Specifies the winter type to use ('ColdWinter', 'AverageWinter', or 'HotWinter').
        Excel_file (str): Path to the Excel file containing updated data.
        Extrapolate_increase (bool): If True, extrapolates data beyond the last available date.
        plot_YN (bool): If True, generates a plot to visualize the updated data.

    Returns:
        pd.DataFrame: Updated gas extraction data.
    """

    # Load updated data from Excel file
    # For this to work, the second column of the file Gaswinning-maandelijks needs to contain the updated well clusters acronyms
    a = pd.read_excel(Excel_file)
    names1 = a[a.columns[0]][:-2]
    names2 = a[a.columns[1]][:-2]
    data = a[a.columns[2:-1]][:-2].T
    data.columns = names2
    idx = [type(data.index[ii]) != str for ii in range(len(data.index))]
    data = data[idx]
    # Load existing gas data
    GasData = np.load(GasData_file, allow_pickle=True).item()

    # Find matching indices for data alignment
    idx0 = np.argmin(abs(GasData["Date"] - data.index[0]))
    idx1 = np.argmin(abs(GasData["Date"] - data.index[-1]))

    # Update gas extraction data
    df1 = GasData[winter].copy()
    df1.iloc[idx0:idx1 + 1] = 0  # Clear existing data in the specified range
    for col in data.columns:
        df1[col].iloc[idx0:idx1 + 1] = data[col]  # Insert updated data

    # Extrapolate data if requested
    if Extrapolate_increase:
        df1.iloc[idx1 + 1 :] = df1.iloc[500 : 500 + len(df1.iloc[idx1 + 1 :])]

    # Generate plot to check consistency (if requested)
    if plot_YN:
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        for scenario in ["HotWinter", "AverageWinter", "ColdWinter"]:
            GasData[scenario].sum(axis=1).plot(
                x=GasData["Date"], ax=ax, ls="--", label=scenario
            )
        df1.sum(axis=1).plot(x=GasData["Date"], ax=ax, label="Corrected data")
        ax.axvline(x=idx1 + 1, c="r", ls="-.", label="End of data")
        ax.legend()
        ax.set_xlabel("Month since 1956")
        ax.set_ylabel("Total gas extraction")

    return df1



import os
def file_exists(file_path):
    """
    Checks if a file exists at the specified path.
    Args:
        file_path (str): The path to the file to check.
    Returns:
        bool: True if the file exists, False otherwise.
    """
    # Use the os.path.isfile function to determine if the file exists
    return os.path.isfile(file_path)
