import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from neuralhydrology.nh_run import eval_run
import pickle
from pathlib import Path
import os
from tqdm import tqdm
from scipy import stats


class Evaluator():
    
    """
    A class for evaluating a model trained by `Neuralhydrology` library
    -------------------
    Requirements:
        - A proper .txt file listing test basin id
        - A proper configuration (.yml) file specifing the test file and date
        - A folder consisting of .csv files of the test basin data with proper observed values
        - An attributes file (.csv) containing basin areas if area normalization was applied.
        
    -------------------
    Model Parameters:
        - run_dir: str
            Name of the directory for the target model
            
        - epoch_num: int
            The number of epoch to be evaluated
            
        - csv_dir: str
            The directory of the original csv files (for observed data)
            
        - eval_list: str
            The name of the .txt file consisting of the basin list to be evaluated
            
        - attributes_file: str
            Path to the CSV file containing basin attributes (e.g., area).
            Required if inverse area normalization is needed.
            
        - basin_area_scale_divisor: float
            The divisor used during the basin area normalization preprocessing step.
            
        - mean: float
            A mean value used for standardizing the target variable (after area norm and log transform).
            
        - var: float
            A variance value used for standardizing the target variable (after area norm and log transform).
            
        - test_start_date: str
            Date to start evaluation 'dd/mm/yyyy`
            
        - test_end_date: str
            Date to end evaluation 'dd/mm/yyyy'
            
        - skip_sim: bool
            Whether to skip simulation - default to false, enable this option when simulation is already done
            
        - apply_transformation: bool
            Whether to apply inverse transformation - default to true, enable this when the standardization is not applied to the data
            
        - target_var: str
            Name of the target variable - default to "discharge"
    
    -------------------   
    Member Functions:
        public:
        - plot_validation() -> None:
            plot the validation error progress during training
            
        - get_validation -> pd.DataFrame:
            returns a dataframe of validation errors for each epoch
            
        - get_prediction(basin_id: str) -> float, np.array:
            returns the simulated discharge for the specified basin
            
        - get_metrics() -> pd.DataFrame:
            returns a dataframe of the simulated nse & kge values for the listed basins
            
        - plot_prediction(basin_id: str) -> None: 
            plot the simulated vs observed discharge (or target) values for a single basin
            
        - plot_nse_distribution(ignore_neg = True) -> None:
            plot the distribution of nse values over multiple basins
            
        - plot_kge_distribution(ignore_neg = True) -> None:
            plot the distribution of kge values over multiple basins
            
        private:
        - __evalute_single__() -> float:
            evaluates a single basin and returns nse value
            
        - __evaluate__() -> pd.DataFrame:
            runs evaluation on the basins listed in eval_list and returns the result dataframe
            
        - __collect_validation__() -> pd.DataFrame:
            collect the mean, median, max values for validation metics for each epoch
    """
    
    def __init__(self, run_dir: str, 
                 epoch_num: int, 
                 csv_dir: str = "data/csv_files",
                 eval_list: str = r"basin_list\test.txt",
                 attributes_file: str = '../metadata/attributes.csv', # New parameter
                 basin_area_scale_divisor: float = 1000.0, # New parameter
                 mean: float = 0.8561527661255196,
                 var: float = 5.06157279557463,
                 test_start_date: str = '01/01/2011',
                 test_end_date: str = '31/12/2022',
                 skip_sim: bool = False,
                 apply_transformation: bool = True,
                 target_var: str = "discharge"
                 ):
        
        self.run_dir = Path("runs/" + run_dir)
        self.epoch_num = epoch_num
        self.csv_dir = Path(csv_dir)
        self.eval_list = Path(eval_list)
        self.attributes_file = Path(attributes_file) # Store new parameter
        self.basin_area_scale_divisor = basin_area_scale_divisor # Store new parameter
        self.mean = mean
        self.var = var
        self.test_start_date = pd.to_datetime(test_start_date, format='%d/%m/%Y')
        self.test_end_date = pd.to_datetime(test_end_date, format='%d/%m/%Y')
        self.skip_sim = skip_sim
        self.apply_transformation = apply_transformation
        self.target_var = target_var
        self.test_name = f"evaluate {run_dir} epoch {epoch_num}"

        if not self.run_dir.exists():
            raise FileNotFoundError(f"The specified run directory does not exist: {self.run_dir}")
        if not self.csv_dir.exists():
            raise FileNotFoundError(f"The specified CSV directory does not exist: {self.csv_dir}")
        if not self.eval_list.exists():
            raise FileNotFoundError(f"The specified evaluation list directory does not exist: {self.eval_list}")
        
        if self.apply_transformation: # Load attributes only if needed
            if not self.attributes_file.exists():
                raise FileNotFoundError(f"Attributes file not found: {self.attributes_file}")
            try:
                self.attributes_df = pd.read_csv(self.attributes_file)
                # Ensure gauge_id column exists before setting as index
                if 'gauge_id' not in self.attributes_df.columns:
                    raise KeyError("'gauge_id' column not found in attributes file. Please ensure it's present.")
                self.attributes_df.set_index('gauge_id', inplace=True)
            except KeyError as e:
                raise KeyError(f"{e} Check that '{self.attributes_file}' has a 'gauge_id' column.")
            except Exception as e:
                raise RuntimeError(f"Error loading attributes file {self.attributes_file}: {e}")
        else:
            self.attributes_df = None # Explicitly set to None if not loaded

        if not skip_sim:
            eval_run(run_dir = self.run_dir, period = "test")
            
        self.__result_df: pd.DataFrame = self.__evaluate__(self.eval_list)
        self.__validation_df: pd.DataFrame = self.__collect_validation__()
        
    
    def __evaluate_single__(self, basin_id: str): # basin_id is typically a string from results
        
        epoch_num_str = str(self.epoch_num)
        if len(epoch_num_str) == 1:
            epoch_folder_name = "model_epoch00" + epoch_num_str
        elif len(epoch_num_str) == 2:
            epoch_folder_name = "model_epoch0" + epoch_num_str
        else:
            epoch_folder_name = "model_epoch" + epoch_num_str

        results_file = self.run_dir / "test" / epoch_folder_name / "test_results.p"
        if not results_file.exists():
            # Fallback for older NeuralHydrology versions or different naming
            results_file_alt = self.run_dir / "test" / f"model_epoch{self.epoch_num:03d}" / "test_results.p"
            if results_file_alt.exists():
                results_file = results_file_alt
            else:
                raise FileNotFoundError(f"Could not find test_results.p in {results_file.parent} or {results_file_alt.parent}")

        with open(results_file, "rb") as fp:
            results = pickle.load(fp)

        qsim = results[basin_id]['1D']['xr']['discharge_sim'] # Assumes target is 'discharge'
        sim_normalized = qsim.values.copy() # Raw, normalized predictions
        dates = qsim['date'].values

        # Load and filter observed data first
        csv_file_path = self.csv_dir / f"{basin_id}.csv"
        df_obs = pd.read_csv(csv_file_path, index_col='date', parse_dates=True)
        if df_obs.index.tz is not None:
            df_obs.index = df_obs.index.tz_localize(None)
        df_obs = df_obs.loc[self.test_start_date:self.test_end_date, [self.target_var]]

        # Create DataFrame from predictions and align with observed data
        sim_df = pd.DataFrame({'date': pd.to_datetime(dates), 'simulated': sim_normalized.flatten()}).set_index('date')
        merged_df = pd.merge(df_obs, sim_df, left_index=True, right_index=True, how='inner')
        merged_df.dropna(subset=[self.target_var, 'simulated'], inplace=True)

        observed_values = merged_df[self.target_var].values
        simulated_values = merged_df['simulated'].values # These are still normalized
        aligned_dates = merged_df.index.values

        if self.apply_transformation:
            # Denormalize the ALIGNED subset of predictions
            sim = simulated_values.copy()
            
            # 1. Inverse Z-score normalization
            sim = (sim * np.sqrt(self.var)) + self.mean
            
            # 2. Inverse Area Normalization (Strategy 1)
            if self.attributes_df is not None:
                try:
                    basin_area = self.attributes_df.loc[str(basin_id), 'area']
                except KeyError:
                    print(f"Warning: Basin ID {basin_id} not found in attributes file. Skipping area denormalization.")
                    basin_area = np.nan

                if pd.isna(basin_area) or basin_area <= 0:
                    print(f"Warning: Invalid area ({basin_area}) for basin {basin_id}. Skipping area denormalization.")
                elif self.basin_area_scale_divisor == 0:
                    print(f"Warning: basin_area_scale_divisor is 0. Skipping area denormalization to avoid division by zero.")
                else:
                    scaled_basin_area = basin_area / self.basin_area_scale_divisor
                    sim = sim * scaled_basin_area
            else:
                print("Warning: attributes_df not loaded, cannot perform area denormalization.")
                
            # 3. Inverse Log transformation
            sim = np.exp(sim) - 1e-6  # EPSILON_S1
            
            # Outlier capping on the final, denormalized values
            sim[sim > 50000] = np.median(sim[sim <= 50000])
            
            simulated_values = sim # Replace with the final denormalized values

        if observed_values.size == 0 or simulated_values.size == 0:
            nse = np.nan
            kge = np.nan
        else:
            mean_observed = np.mean(observed_values)
            sum_squared_diff = np.sum((observed_values - simulated_values) ** 2)
            sum_squared_diff_mean = np.sum((observed_values - mean_observed) ** 2)
            nse = 1 - (sum_squared_diff / sum_squared_diff_mean) if sum_squared_diff_mean != 0 else np.nan
            nse = nse.item() if hasattr(nse, 'item') else nse
        
            try:
                r = np.corrcoef(observed_values, simulated_values)[0, 1]
                alpha = np.std(simulated_values) / np.std(observed_values) if np.std(observed_values) != 0 else np.nan
                beta = np.mean(simulated_values) / np.mean(observed_values) if np.mean(observed_values) != 0 else np.nan
                if pd.isna(r) or pd.isna(alpha) or pd.isna(beta):
                    kge = np.nan
                else:
                    kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
                kge = kge.item() if hasattr(kge, 'item') else kge
            except Exception as e:
                # print(f"KGE calculation error for {basin_id}: {e}")
                kge = np.nan

        return nse, kge, simulated_values, observed_values, aligned_dates

        
    def __evaluate__(self, eval_list_path: Path):
        with open(eval_list_path, 'r') as file:
            basin_ids = file.read().splitlines()
            
        nse_values = []
        kge_values = []
        
        print("Collecting NSE and KGE values")
        for basin_id in tqdm(basin_ids):
            try:
                nse, kge, _, _, _ = self.__evaluate_single__(basin_id)
            except Exception as e:
                print(f"Error evaluating basin {basin_id}: {e}")
                nse, kge = np.nan, np.nan
            nse_values.append(nse)
            kge_values.append(kge)
        
        result_df = pd.DataFrame({
            'basin_id': basin_ids, 
            'NSE': nse_values,
            'KGE': kge_values
        })
        
        result_df['Performance'] = result_df['NSE'].apply(
            lambda x: 'Excellent' if x > 0.75 else 
                    'Good' if x >= 0.36 else 
                    'Unsatisfactory' if x >= 0 else 
                    'Negative' if pd.notna(x) else 'N/A' # Handle NaN explicitly
        )
        
        result_df.loc[result_df['NSE'].isnull(), 'Performance'] = "N/A"
        
        return result_df
    
    def __collect_validation__(self):
        validation_folder = self.run_dir / "validation"
        epoches = []
        mean_nse_values = []
        median_nse_values = []
        max_nse_values = []
        # Add KGE metrics
        mean_kge_values = []
        median_kge_values = []
        max_kge_values = []


        print("Collecting validation metrics")
        for epoch in range(1, self.epoch_num + 1):
            epoches.append(epoch)
            epoch_folder_name = f"model_epoch{epoch:03d}"
            csv_file = validation_folder / epoch_folder_name / "validation_metrics.csv"
            
            if csv_file.exists():
                df = pd.read_csv(csv_file)
                
                nse_series = df["NSE"].dropna()
                kge_series = df.get("KGE", pd.Series(dtype='float64')).dropna() # Use .get for KGE

                mean_nse_values.append(nse_series.mean() if not nse_series.empty else np.nan)
                median_nse_values.append(nse_series.median() if not nse_series.empty else np.nan)
                max_nse_values.append(nse_series.max() if not nse_series.empty else np.nan)

                mean_kge_values.append(kge_series.mean() if not kge_series.empty else np.nan)
                median_kge_values.append(kge_series.median() if not kge_series.empty else np.nan)
                max_kge_values.append(kge_series.max() if not kge_series.empty else np.nan)
            else:
                mean_nse_values.append(np.nan)
                median_nse_values.append(np.nan)
                max_nse_values.append(np.nan)
                mean_kge_values.append(np.nan)
                median_kge_values.append(np.nan)
                max_kge_values.append(np.nan)

        validation_df = pd.DataFrame({
            "epoch": epoches,
            "mean_nse": mean_nse_values,
            "median_nse": median_nse_values,
            "max_nse": max_nse_values,
            "mean_kge": mean_kge_values,
            "median_kge": median_kge_values,
            "max_kge": max_kge_values,
        })

        return validation_df
    
    def __str__(self):
        self.print_summary()
        return self.test_name
    
    def plot_validation(self, plot_type: str = "Median", metric: str = "NSE"):
        
        valid_metrics = ["NSE", "KGE"]
        if metric.upper() not in valid_metrics:
            raise ValueError(f"Invalid metric: {metric}. Expected one of {valid_metrics}")

        column_name = ""
        if plot_type.lower() == "median":
            column_name = f"median_{metric.lower()}"
        elif plot_type.lower() == "mean":
            column_name = f"mean_{metric.lower()}"
        elif plot_type.lower() == "max":
            column_name = f"max_{metric.lower()}"
        else:
            raise ValueError(f"Incorrect plot type: {plot_type}. Expected 'Median', 'Mean', or 'Max'.")

        if column_name not in self.__validation_df.columns:
             raise ValueError(f"Validation data for '{column_name}' not found. Available columns: {self.__validation_df.columns.tolist()}")

        validation_values = self.__validation_df[column_name]
        
        plt.figure(figsize=(10, 6))
        epochs = self.__validation_df["epoch"] # Use epoch numbers from df
        plt.plot(epochs, validation_values, marker='o', label=f'{plot_type} {metric.upper()}')
        plt.xlabel("Epoch")
        plt.ylabel(f"{plot_type} {metric.upper()}")
        plt.title(f"Validation {plot_type} {metric.upper()} Progress")
        plt.grid(True)
        plt.legend()
        plt.show()
        
    def get_prediction(self, basin_id: str):
        nse, kge, pred, obs, dates = self.__evaluate_single__(basin_id) # obs and dates also returned
        return nse, kge, pred, obs, dates # Return all for potential use
    
    def get_metrics(self):
        return self.__result_df
        
    def get_validation(self):
        return self.__validation_df
    
    def plot_prediction(self, basin_id: str):
        try:
            nse, kge, sim, obs, date_aligned = self.__evaluate_single__(basin_id)
        except Exception as e:
            print(f"Error getting prediction for basin {basin_id}: {e}")
            return

        if obs is None or sim is None or date_aligned is None or len(obs) == 0:
            print(f"Not enough data to plot for basin {basin_id}.")
            return
        
        plt.figure(figsize=(14, 7))
        plt.plot(date_aligned, obs, label='Observed', alpha=0.7)
        plt.plot(date_aligned, sim, label='Simulated', alpha=0.7)
        plt.xlabel('Date')
        plt.ylabel(self.target_var.capitalize()) # Use target_var
        plt.title(f'Basin {basin_id} Observed vs Simulated\nNSE: {nse:.3f}, KGE: {kge:.3f}')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
    def print_summary(self):
        print(f"Summary for: {self.test_name}")
        print(f"\nNSE Summary Statistics: \n{self.__result_df['NSE'].replace([-np.inf, np.inf], np.nan).dropna().describe()}\n")
        print(f"KGE Summary Statistics: \n{self.__result_df['KGE'].replace([-np.inf, np.inf], np.nan).dropna().describe()}\n")
        print(f"Performance Summary (based on NSE): \n{self.__result_df['Performance'].value_counts(dropna=False)}\n") # show N/A counts
        
        
    def plot_nse_distribution(self, ignore_neg=True): 
        nse_values = self.__result_df['NSE'].replace([-np.inf, np.inf], np.nan).dropna()
        
        if ignore_neg:
            nse_values = nse_values[nse_values >= 0]
        
        if nse_values.empty:
            print("No NSE values to plot after filtering.")
            return

        plt.figure(figsize=(10, 6))
        plt.boxplot(nse_values, vert=True, patch_artist=True, labels=['NSE'])
        plt.title("Distribution of NSE Values Across Subbasins")
        plt.ylabel("NSE")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.hist(nse_values, bins=20, edgecolor='black', alpha=0.7) # Increased bins
        plt.title("Histogram of NSE Values Across Subbasins")
        plt.xlabel("NSE")
        plt.ylabel("Frequency")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()
        
    def plot_kge_distribution(self, ignore_neg=True):
        kge_values = self.__result_df['KGE'].replace([-np.inf, np.inf], np.nan).dropna()
        
        if ignore_neg:
            kge_values = kge_values[kge_values >= 0]

        if kge_values.empty:
            print("No KGE values to plot after filtering.")
            return
        
        plt.figure(figsize=(10, 6))
        plt.boxplot(kge_values, vert=True, patch_artist=True, labels=['KGE'])
        plt.title("Distribution of KGE Values Across Subbasins")
        plt.ylabel("KGE")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.hist(kge_values, bins=20, edgecolor='black', alpha=0.7) # Increased bins
        plt.title("Histogram of KGE Values Across Subbasins")
        plt.xlabel("KGE")
        plt.ylabel("Frequency")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()