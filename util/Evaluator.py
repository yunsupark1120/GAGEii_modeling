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

        - evaluate_extremes(high_flow_quantile=0.95, low_flow_quantile=0.10) -> pd.DataFrame:
            NEW: Performs stress tests on extreme high and low flow events.
            
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
                 attributes_file: str = '../metadata/attributes.csv',
                 basin_area_scale_divisor: float = 100.0,
                 mean: float = 0.8561527661255196,
                 var: float = 5.06157279557463,
                 test_start_date: str = '01/01/2011',
                 test_end_date: str = '31/12/2022',
                 skip_sim: bool = False,
                 apply_transformation: bool = True,
                 apply_basin_norm: bool = False,
                 target_var: str = "discharge"
                 ):
        
        self.run_dir = Path("runs/" + run_dir)
        self.epoch_num = epoch_num
        self.csv_dir = Path(csv_dir)
        self.eval_list = Path(eval_list)
        self.attributes_file = Path(attributes_file)
        self.basin_area_scale_divisor = basin_area_scale_divisor
        self.mean = mean
        self.var = var
        self.test_start_date = pd.to_datetime(test_start_date, format='%d/%m/%Y')
        self.test_end_date = pd.to_datetime(test_end_date, format='%d/%m/%Y')
        self.skip_sim = skip_sim
        self.apply_transformation = apply_transformation
        self.apply_basin_norm = apply_basin_norm
        self.target_var = target_var
        self.test_name = f"evaluate {run_dir} epoch {epoch_num}"

        if not self.run_dir.exists():
            raise FileNotFoundError(f"The specified run directory does not exist: {self.run_dir}")
        if not self.csv_dir.exists():
            raise FileNotFoundError(f"The specified CSV directory does not exist: {self.csv_dir}")
        if not self.eval_list.exists():
            raise FileNotFoundError(f"The specified evaluation list directory does not exist: {self.eval_list}")
        
        if self.apply_transformation and self.apply_basin_norm:
            if not self.attributes_file.exists():
                raise FileNotFoundError(f"Attributes file not found: {self.attributes_file}")
            try:
                self.attributes_df = pd.read_csv(self.attributes_file)
                if 'gauge_id' not in self.attributes_df.columns:
                    raise KeyError("'gauge_id' column not found in attributes file.")
                self.attributes_df.set_index('gauge_id', inplace=True)
            except Exception as e:
                raise RuntimeError(f"Error loading attributes file {self.attributes_file}: {e}")
        else:
            self.attributes_df = None

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
            
            # 2. Inverse Log transformation
            sim = np.exp(sim) - 1e-6  # EPSILON_S1
            
            # 3. Inverse Area Normalization 
            if self.apply_basin_norm:  # <--- Only apply if requested
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
    
    def plot_prediction(self, basin_id: str, legend_fontsize: int = 10, x_label_fontsize: int = 10, y_label_fontsize: int = 10):
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
        plt.xlabel('Date', fontsize = x_label_fontsize)
        plt.ylabel(self.target_var.capitalize(), fontsize = y_label_fontsize) # Use target_var
        plt.title(f'Basin {basin_id} Observed vs Simulated\nNSE: {nse:.3f}, KGE: {kge:.3f}')
        plt.legend(fontsize = legend_fontsize)
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

    # ------------------- NEW METHODS START HERE -------------------

    def __calculate_event_metrics(self, observed: np.ndarray, simulated: np.ndarray, quantile: float, event_type: str):
        """
        Private helper to calculate PBIAS and volume metrics for specific flow events.
        
        Args:
            observed: Array of observed streamflow.
            simulated: Array of simulated streamflow.
            quantile: The quantile to define the event threshold (e.g., 0.95 for high flow).
            event_type: Either 'high' or 'low'.
            
        Returns:
            A tuple containing (PBIAS, Volume_Fraction).
        """
        # Ensure there are non-zero flows to calculate quantiles on, avoiding warnings/errors
        valid_obs = observed[observed > 0]
        if len(valid_obs) == 0:
            return np.nan, np.nan

        threshold = np.quantile(valid_obs, quantile)
        
        if event_type == 'high':
            event_indices = observed >= threshold
        elif event_type == 'low':
            event_indices = observed <= threshold
        else:
            raise ValueError("event_type must be 'high' or 'low'")

        if not np.any(event_indices):
            return np.nan, np.nan  # No events of this type found in the basin

        obs_event = observed[event_indices]
        sim_event = simulated[event_indices]
        
        # Percentage Bias (PBIAS) for the event
        numerator = np.sum(sim_event - obs_event)
        denominator = np.sum(obs_event)
        pbia = (numerator / denominator) * 100 if denominator != 0 else np.nan

        # Fraction of Volume (FHV or FLV)
        # This is the sum of simulated flow during the event period divided by the total observed flow
        total_obs_volume = np.sum(observed)
        volume_fraction = np.sum(sim_event) / total_obs_volume if total_obs_volume != 0 else np.nan

        return pbia, volume_fraction

    def evaluate_extremes(self, high_flow_quantile: float = 0.95, low_flow_quantile: float = 0.10):
        """
        Performs stress tests on extreme events (high and low flows) using existing predictions.

        This function calculates event-based metrics to assess model performance during
        hydrologically critical periods, which may be obscured by overall metrics like NSE.

        Args:
            high_flow_quantile (float): The quantile of observed flow to define high-flow (flood) events. Defaults to 0.95.
            low_flow_quantile (float): The quantile of observed flow to define low-flow (drought) events. Defaults to 0.10.

        Returns:
            pd.DataFrame: A DataFrame containing basin IDs and the calculated metrics:
                          - PBIAS_high: Percentage Bias during high-flow events.
                          - FHV: Fraction of High-Flow Volume.
                          - PBIAS_low: Percentage Bias during low-flow events.
                          - FLV: Fraction of Low-Flow Volume.
        """
        basin_ids = self.__result_df['basin_id'].tolist()
        extreme_metrics = []

        print("Evaluating extreme event metrics...")
        for basin_id in tqdm(basin_ids):
            try:
                _, _, sim, obs, _ = self.__evaluate_single__(basin_id)
                
                # Check if there is enough data to calculate meaningful quantiles
                if obs is None or sim is None or len(obs) < 30:
                    metrics = {'basin_id': basin_id, 'PBIAS_high': np.nan, 'FHV': np.nan, 'PBIAS_low': np.nan, 'FLV': np.nan}
                    extreme_metrics.append(metrics)
                    continue

                # High flow metrics
                pbia_high, fhv = self.__calculate_event_metrics(obs, sim, high_flow_quantile, 'high')

                # Low flow metrics
                pbia_low, flv = self.__calculate_event_metrics(obs, sim, low_flow_quantile, 'low')

                metrics = {
                    'basin_id': basin_id,
                    'PBIAS_high': pbia_high,
                    'FHV': fhv,
                    'PBIAS_low': pbia_low,
                    'FLV': flv
                }
                extreme_metrics.append(metrics)

            except Exception as e:
                print(f"Error evaluating extremes for basin {basin_id}: {e}")
                metrics = {'basin_id': basin_id, 'PBIAS_high': np.nan, 'FHV': np.nan, 'PBIAS_low': np.nan, 'FLV': np.nan}
                extreme_metrics.append(metrics)

        return pd.DataFrame(extreme_metrics)
    
    def _calculate_error_by_quantile(self, observed: np.ndarray, simulated: np.ndarray, num_quantiles: int = 10):
        """
        Private helper to calculate bias for different flow quantiles.
        """
        # Create a DataFrame for easier processing
        df = pd.DataFrame({'observed': observed, 'simulated': simulated})
        
        # Define quantile bins based on the observed flow
        df['quantile_bin'] = pd.qcut(df['observed'], q=num_quantiles, labels=False, duplicates='drop')
        
        # Calculate bias (simulated - observed) for each row
        df['bias'] = df['simulated'] - df['observed']
        
        # Group by quantile bin and calculate the mean bias
        quantile_bias = df.groupby('quantile_bin')['bias'].mean()
        
        # Create labels for the plot
        labels = []
        quantiles = np.linspace(0, 1, num_quantiles + 1)
        for i in range(num_quantiles):
            labels.append(f'{quantiles[i]*100:.0f}-{quantiles[i+1]*100:.0f}%')
        
        # Ensure the labels match the calculated biases
        quantile_bias.index = labels[:len(quantile_bias)]
        
        return quantile_bias

    def plot_case_study(self, basin_id: str, storm_date_str: str, window_days: int = 10):
        """
        Generates a comprehensive multi-panel plot for a single basin case study.

        Args:
            basin_id (str): The ID of the basin to plot.
            storm_date_str (str): The center date for the storm window, e.g., '2015-06-20'.
            window_days (int): The number of days to show in the storm window plot.
        """
        try:
            nse, kge, sim, obs, dates = self.__evaluate_single__(basin_id)
            if obs is None:
                print(f"No data available for basin {basin_id}")
                return
        except Exception as e:
            print(f"Could not process basin {basin_id}: {e}")
            return

        # Create a 3-panel figure
        fig, axes = plt.subplots(3, 1, figsize=(12, 18), gridspec_kw={'height_ratios': [1, 1, 1]})
        fig.suptitle(f'Case Study: Basin {basin_id} (NSE: {nse:.3f}, KGE: {kge:.3f})', fontsize=16)

        # --- Panel 1: Full Hydrograph ---
        axes[0].plot(dates, obs, label='Observed', color='k', lw=1.5)
        axes[0].plot(dates, sim, label='Simulated', color='r', alpha=0.8, lw=1.5)
        axes[0].set_title('Full Period Hydrograph')
        axes[0].set_ylabel(f'{self.target_var.capitalize()} ($m^3/s$)')
        axes[0].legend()
        axes[0].grid(True, which='both', linestyle='--', linewidth=0.5)

        # --- Panel 2: Storm Response Window ---
        storm_center = pd.to_datetime(storm_date_str)
        start_date = storm_center - pd.Timedelta(days=window_days/2)
        end_date = storm_center + pd.Timedelta(days=window_days/2)

        df_full = pd.DataFrame({'date': dates, 'obs': obs, 'sim': sim}).set_index('date')
        df_storm = df_full.loc[start_date:end_date]

        axes[1].plot(df_storm.index, df_storm['obs'], 'o-', label='Observed', color='k', lw=2)
        axes[1].plot(df_storm.index, df_storm['sim'], 'o--', label='Simulated', color='r', alpha=0.8, lw=2)
        axes[1].set_title(f'Storm Response Window ({storm_date_str})')
        axes[1].set_ylabel(f'{self.target_var.capitalize()} ($m^3/s$)')
        axes[1].legend()
        axes[1].grid(True, which='both', linestyle='--', linewidth=0.5)
        fig.autofmt_xdate() # Improve date formatting

        # --- Panel 3: Errors by Flow Quantile ---
        quantile_bias = self._calculate_error_by_quantile(obs, sim)
        quantile_bias.plot(kind='bar', ax=axes[2], color='steelblue', edgecolor='k')
        axes[2].axhline(0, color='k', linestyle='--', linewidth=1)
        axes[2].set_title('Mean Error (Bias) by Flow Quantile')
        axes[2].set_xlabel('Observed Flow Quantile Bins')
        axes[2].set_ylabel('Bias ($Simulated - Observed$)')
        axes[2].grid(axis='y', linestyle='--', linewidth=0.5)
        plt.setp(axes[2].get_xticklabels(), rotation=45, ha="right")


        plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust for suptitle
        plt.show()
