import xarray as xr
import os
import pandas as pd

def DataFrame_to_CDF(data: pd.DataFrame, 
                     output_dir: str,
                     output_name: str):
    """
    This function converts a pandas DataFrame to a NetCDF file, allowing missing values (NaN).
    
    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame to be converted to NetCDF.
    output_dir : str
        The directory where the NetCDF file will be saved.
    output_name : str
        The name of the NetCDF file.
    ----------
    
    Data Requirements
    ----------
    - The DataFrame index must be named 'date'.
    - The DataFrame index must contain valid dates in the format '2000-01-01'.
    - Missing values (NaN) are allowed and will be retained in the NetCDF file.
    ----------
    """
    # Check if the index is named 'date'
    if data.index.name != 'date':
        raise ValueError("Index must be named 'date'")
    
    # Check if the index values are valid dates
    if not pd.to_datetime(data.index, errors='coerce').notna().all():
        raise ValueError("Index must contain valid dates in the format '2000-01-01'")
    
    # Ensure the directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert DataFrame to xarray Dataset and preserve NaN values
    ds = xr.Dataset.from_dataframe(data)
    ds = ds.fillna(float('nan'))  # Ensuring NaNs are explicitly preserved
    
    # Save to NetCDF
    ds.to_netcdf(f"{output_dir}/{output_name}.nc")
                    

def read_CDF(file_path: str) -> pd.DataFrame:
    """
    Reads a NetCDF file and converts it back to a pandas DataFrame.
    
    Parameters
    ----------
    file_path : str
        The path to the NetCDF file.
    
    Returns
    ----------
    pd.DataFrame
        A DataFrame containing the data from the NetCDF file.
    """
    ds = xr.open_dataset(file_path)
    return ds.to_dataframe()
