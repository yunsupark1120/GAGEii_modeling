import xarray as xr
import os
import pandas as pd

def DataFrame_to_CDF(data: pd.DataFrame, 
                     output_dir: str,
                     output_name: str):
    """
    This function converts a pandas DataFrame to a NetCDF file.
    
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
    - The DataFrame must not contain missing values.
    ----------
    
    """
    # Check if the index is named 'date'
    if data.index.name != 'date':
        raise ValueError("Index must be named 'date'")
    
    # Check if the index values are dates in the format '2000-01-01'
    if not pd.to_datetime(data.index, errors='coerce').notna().all():
        raise ValueError("Index must contain valid dates in the format '2000-01-01'")
    
    # Check for missing values in the dataframe
    if data.isnull().values.any():
        raise ValueError("DataFrame contains missing values")
    
    # Ensure the directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert DataFrame to xarray Dataset and save to NetCDF
    ds = xr.Dataset.from_dataframe(data)
    ds.to_netcdf(f"{output_dir}/{output_name}.nc")
                    

def read_CDF(file_path: str) -> pd.DataFrame:
    ds = xr.open_dataset(file_path)
    return ds.to_dataframe()
