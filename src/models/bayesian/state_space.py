"""
Bayesian State-Space Model for Chikungunya Early Warning

Hierarchical Negative Binomial state-space model with:
- Latent log-transmission risk Z_{d,t}
- AR(1) dynamics with climate forcing
- Partial pooling across districts

Reference: Phase 4 design specification
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import warnings

from src.config import get_repo_root
from src.common.thresholds import enforce_minimum_threshold

try:
    from cmdstanpy import CmdStanModel
    CMDSTAN_AVAILABLE = True
except ImportError:
    CMDSTAN_AVAILABLE = False
    warnings.warn("CmdStanPy not available. Install with: pip install cmdstanpy")

from ..base import BaseModel


class BayesianStateSpace(BaseModel):
    """
    Hierarchical Bayesian state-space model for outbreak prediction.
    
    Uses Stan for MCMC inference via CmdStanPy.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(name="bayesian_state_space", config=config)
        
        if not CMDSTAN_AVAILABLE:
            raise RuntimeError("CmdStanPy required but not available")
        
        # Model configuration
        self.n_warmup = config.get('n_warmup', 500) if config else 500
        self.n_samples = config.get('n_samples', 500) if config else 500
        self.n_chains = config.get('n_chains', 4) if config else 4
        self.seed = config.get('seed', 42) if config else 42
        self.adapt_delta = config.get('adapt_delta', 0.95) if config else 0.95  # FIX #1: Control sampler step size
        self.outbreak_percentile = config.get('outbreak_percentile') if config else None
        
        # Stan model path
        self.stan_file = config.get('stan_file', None) if config else None
        
        # Fitted objects
        self.model_ = None
        self.fit_ = None
        self.data_ = None
        self.district_map_ = None
        
    def _get_stan_file(self) -> Path:
        """Get path to Stan model file."""
        if self.stan_file:
            return Path(self.stan_file)
        
        # Default location(s): prefer repo root (shared across versions),
        # fall back to v6-local checkout if present.
        module_dir = Path(__file__).parent
        v6_root = module_dir.parent.parent.parent

        candidates = [
            get_repo_root() / "stan_models" / "hierarchical_ews_v01.stan",
            v6_root / "stan_models" / "hierarchical_ews_v01.stan",
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            "Stan model not found. Looked in: " + ", ".join(str(p) for p in candidates)
        )
    
    def _prepare_stan_data(
        self, 
        df: pd.DataFrame,
        feature_cols: list,
        target_col: str = 'cases',
        forecast_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Prepare data dictionary for Stan model including optional forecast.
        
        Args:
            df: DataFrame with features, target, and metadata (training data)
            feature_cols: Feature column names (used for temp anomaly)
            target_col: Column with case counts
            forecast_df: Optional DataFrame for forecasting (test data)
            
        Returns:
            Dictionary formatted for Stan with forecast inputs if provided
        """
        # FIX #5: Add deterministic sorting with unique row ID to prevent non-deterministic ordering
        # This ensures consistent district_id and time_idx assignment across runs
        df['_unique_row_id'] = range(len(df))
        
        # Sort by district and time, with unique row ID as tiebreaker
        sort_cols = ['state', 'district', 'year', 'week', '_unique_row_id']
        df = df.sort_values(sort_cols).reset_index(drop=True)
        
        # Validate week numbers before creating time indices
        if not df['week'].between(1, 53).all():
            invalid_weeks = df[~df['week'].between(1, 53)]['week'].unique()
            raise ValueError(f"Invalid week numbers found: {invalid_weeks}. Weeks must be 1-53.")
        
        # Create district IDs
        df['district_id'] = pd.factorize(df['state'] + '_' + df['district'])[0] + 1
        self.district_map_ = df[['state', 'district', 'district_id']].drop_duplicates()
        
        # FIX #5: Validate no duplicate district-week combinations (would cause misalignment)
        dups = df.groupby(['state', 'district', 'year', 'week']).size()
        if (dups > 1).any():
            dup_entries = dups[dups > 1].head()
            raise ValueError(
                f"Duplicate district-week combinations found. This indicates data quality issues.\n"
                f"Duplicates: {dup_entries.to_dict()}"
            )
        
        # Create time index (week within the dataset)
        df['time_idx'] = pd.factorize(
            df['year'].astype(str) + '_' + df['week'].astype(str).str.zfill(2)
        )[0] + 1
        
        N = len(df)
        D = df['district_id'].max()
        T_max = df['time_idx'].max()
        
        # Get cases (convert to integer counts)
        # If we have incidence per 100k, convert back to approximate counts
        if target_col == 'cases':
            y = df['cases'].fillna(0).astype(int).values
        else:
            y = df[target_col].fillna(0).astype(int).values
        
        # Get temperature anomaly (use feat_temp_anomaly if available)
        # Missing temperature data filtered out during valid_df preparation in training scripts
        if 'feat_temp_anomaly' in df.columns:
            temp_anomaly = df['feat_temp_anomaly'].values
        elif 'temp_celsius' in df.columns:
            # Compute simple anomaly: deviation from mean
            temp_anomaly = (df['temp_celsius'] - df['temp_celsius'].mean()).values
        else:
            # Fallback only if temp column completely missing
            temp_anomaly = np.zeros(N)
        
        # CRITICAL: Replace any NaN values with 0 (neutral anomaly)
        # NaN would cause Stan to fail with "log location parameter is nan"
        temp_anomaly = np.nan_to_num(temp_anomaly, nan=0.0)
        
        # Extract degree-days (mechanistic temperature threshold)
        if 'feat_degree_days_above_20' in df.columns:
            degree_days = df['feat_degree_days_above_20'].values
        elif 'degree_days_above_20' in df.columns:
            degree_days = df['degree_days_above_20'].values
        else:
            # Fallback: compute from temperature if available
            if 'temp_celsius' in df.columns:
                temp = df['temp_celsius'].values
                degree_days = np.maximum(temp - 20.0, 0.0) * 7  # Weekly degree-days
            else:
                degree_days = np.zeros(N)
        
        # CRITICAL: Replace any NaN values with 0 (neutral signal)
        degree_days = np.nan_to_num(degree_days, nan=0.0)
        
        # FIX #6: CRITICAL - Scale degree_days to prevent Stan overflow
        # Unscaled values up to 372.57 cause exp(log_mu) > 10^86 overflow
        # Scaling by 100 reduces range to [0, 3.73], preventing numerical issues
        degree_days = degree_days / 100.0
        
        # Extract rainfall persistence (4-week cumulative)
        if 'feat_rain_persist_4w' in df.columns:
            rain_persist = df['feat_rain_persist_4w'].values
        elif 'rain_persist_4w' in df.columns:
            rain_persist = df['rain_persist_4w'].values
        else:
            rain_persist = np.zeros(N)
        
        # CRITICAL: Replace any NaN values with 0 (neutral signal)
        rain_persist = np.nan_to_num(rain_persist, nan=0.0)
        
        stan_data = {
            'N': N,
            'D': D,
            'T_max': T_max,
            'district': df['district_id'].values,
            'time': df['time_idx'].values,
            'y': y,
            'temp_anomaly': temp_anomaly,
            'degree_days': degree_days,
            'rain_persist': rain_persist
        }
        
        # FIX #6: Input validation to catch issues before Stan
        # Validate all arrays are finite (no NaN, no Inf)
        assert np.isfinite(temp_anomaly).all(), "temp_anomaly contains non-finite values (NaN/Inf)"
        assert np.isfinite(degree_days).all(), "degree_days contains non-finite values (NaN/Inf)"
        assert np.isfinite(rain_persist).all(), "rain_persist contains non-finite values (NaN/Inf)"
        assert np.isfinite(y).all(), "case counts contain non-finite values (NaN/Inf)"
        
        # Validate ranges to prevent overflow
        if np.abs(degree_days).max() > 10:
            raise ValueError(
                f"degree_days too large after scaling (max={np.abs(degree_days).max():.2f}). "
                f"Expected < 10 after /100 scaling. Check feature engineering."
            )
        
        if np.abs(temp_anomaly).max() > 20:
            raise ValueError(
                f"temp_anomaly too large (max={np.abs(temp_anomaly).max():.2f}). "
                f"Expected |anomaly| < 20°C. Check data quality."
            )
        
        # Warn about data sparsity (increases Stan failure risk)
        coverage = N / (D * T_max)
        if coverage < 0.05:
            warnings.warn(
                f"Data is very sparse: {coverage*100:.1f}% coverage ({N} obs / {D}×{T_max} grid). "
                f"Latent states may be poorly constrained. Consider filtering to districts with "
                f"better temporal coverage or expect longer MCMC convergence times.",
                UserWarning
            )
        
        # Add forecast data if provided
        if forecast_df is not None:
            forecast_df = forecast_df.copy()
            
            # Map forecast districts to training district IDs
            forecast_df['district_key'] = forecast_df['state'] + '_' + forecast_df['district']
            train_district_map = dict(zip(
                self.district_map_['state'] + '_' + self.district_map_['district'],
                self.district_map_['district_id']
            ))
            
            forecast_df['district_id'] = forecast_df['district_key'].map(train_district_map)
            
            # Check for unseen districts
            unseen = forecast_df['district_id'].isna()
            if unseen.any():
                unseen_districts = forecast_df.loc[unseen, 'district_key'].unique()
                raise ValueError(
                    f"Forecast data contains {len(unseen_districts)} districts not in training: "
                    f"{list(unseen_districts)[:5]}..."
                )
            
            # Create time indices continuing from T_max
            # Map year-week to time indices
            train_time_map = dict(zip(
                df['year'].astype(str) + '_' + df['week'].astype(str).str.zfill(2),
                df['time_idx']
            ))
            
            # Get max training time
            max_train_time = df['time_idx'].max()
            
            # Create new time indices for forecast period
            forecast_df['time_key'] = (
                forecast_df['year'].astype(str) + '_' + 
                forecast_df['week'].astype(str).str.zfill(2)
            )
            
            # Assign sequential time indices starting after training
            unique_forecast_times = forecast_df['time_key'].unique()
            forecast_time_map = {}
            next_time_idx = max_train_time + 1
            
            for time_key in sorted(unique_forecast_times):
                if time_key not in train_time_map:
                    forecast_time_map[time_key] = next_time_idx
                    next_time_idx += 1
            
            forecast_df['time_idx'] = forecast_df['time_key'].map(forecast_time_map)
            
            # Get temperature anomaly for forecast
            if 'feat_temp_anomaly' in forecast_df.columns:
                temp_anomaly_forecast = forecast_df['feat_temp_anomaly'].values
            elif 'temp_celsius' in forecast_df.columns:
                # Use training mean for normalization
                train_temp_mean = df['temp_celsius'].mean() if 'temp_celsius' in df.columns else 0
                temp_anomaly_forecast = (forecast_df['temp_celsius'] - train_temp_mean).values
            else:
                temp_anomaly_forecast = np.zeros(len(forecast_df))
            
            # CRITICAL: Replace any NaN values with 0 (neutral anomaly) for forecasts too
            temp_anomaly_forecast = np.nan_to_num(temp_anomaly_forecast, nan=0.0)
            
            # Forecast degree-days
            if 'feat_degree_days_above_20' in forecast_df.columns:
                degree_days_forecast = forecast_df['feat_degree_days_above_20'].values
            elif 'temp_celsius' in forecast_df.columns:
                temp = forecast_df['temp_celsius'].values
                degree_days_forecast = np.maximum(temp - 20.0, 0.0) * 7
            else:
                degree_days_forecast = np.zeros(len(forecast_df))
            
            # CRITICAL: Replace any NaN values with 0 (neutral signal)
            degree_days_forecast = np.nan_to_num(degree_days_forecast, nan=0.0)
            
            # Forecast rain persistence
            if 'feat_rain_persist_4w' in forecast_df.columns:
                rain_persist_forecast = forecast_df['feat_rain_persist_4w'].values
            else:
                rain_persist_forecast = np.zeros(len(forecast_df))
            
            # CRITICAL: Replace any NaN values with 0 (neutral signal)
            rain_persist_forecast = np.nan_to_num(rain_persist_forecast, nan=0.0)
            
            stan_data.update({
                'N_forecast': len(forecast_df),
                'district_forecast': forecast_df['district_id'].astype(int).values,
                'time_forecast': forecast_df['time_idx'].astype(int).values,
                'temp_anomaly_forecast': temp_anomaly_forecast,
                'degree_days_forecast': degree_days_forecast,
                'rain_persist_forecast': rain_persist_forecast
            })
            
            # Store forecast mapping for later retrieval
            self.forecast_df_ = forecast_df
        else:
            # No forecast data - add empty arrays
            stan_data.update({
                'N_forecast': 0,
                'district_forecast': np.array([], dtype=int),
                'time_forecast': np.array([], dtype=int),
                'temp_anomaly_forecast': np.array([])
            })
            self.forecast_df_ = None
        
        self.data_ = stan_data
        return stan_data
    
    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        df: Optional[pd.DataFrame] = None,
        feature_cols: Optional[list] = None,
        forecast_df: Optional[pd.DataFrame] = None
    ) -> 'BayesianStateSpace':
        """
        Fit the Bayesian state-space model via MCMC.
        
        Args:
            X: Feature matrix (for API compatibility)
            y: Target vector (for API compatibility)
            df: Full DataFrame with metadata (preferred)
            feature_cols: Feature column names
            forecast_df: Optional test DataFrame for forecasting setup
            
        Returns:
            self
        """
        if df is None:
            raise ValueError("DataFrame with metadata required for Bayesian model")
        
        # Compile Stan model with C++ optimizations
        stan_file = self._get_stan_file()
        print(f"Compiling Stan model from {stan_file}...")
        cpp_options = {
            'STAN_OPENCL': 'FALSE',  # Metal not supported, disable OpenCL
            'CXXFLAGS': '-O3 -march=native -mtune=native'  # Aggressive CPU optimizations
        }
        print(f"Using C++ optimizations: {cpp_options['CXXFLAGS']}")
        self.model_ = CmdStanModel(
            stan_file=str(stan_file),
            cpp_options=cpp_options
        )
        
        # Prepare data (including forecast if provided)
        print("Preparing data for Stan...")
        stan_data = self._prepare_stan_data(df, feature_cols or [], forecast_df=forecast_df)
        
        print(f"Data summary: N={stan_data['N']}, D={stan_data['D']}, T_max={stan_data['T_max']}")
        if forecast_df is not None:
            print(f"Forecast setup: N_forecast={stan_data['N_forecast']}")
        
        # Run MCMC
        print(f"Running MCMC: {self.n_chains} chains, {self.n_warmup} warmup, {self.n_samples} samples...")
        self.fit_ = self.model_.sample(
            data=stan_data,
            chains=self.n_chains,
            iter_warmup=self.n_warmup,
            iter_sampling=self.n_samples,
            seed=self.seed,
            adapt_delta=self.adapt_delta,  # FIX #1: Add adapt_delta for better convergence
            show_progress=True
        )
        
        self.is_fitted = True
        return self
    
    def predict_proba(self, X: np.ndarray, df: Optional[pd.DataFrame] = None, use_forecast: bool = False) -> np.ndarray:
        """
        Get posterior predictive outbreak probabilities.
        
        Can return either:
        - Training predictions (y_rep) if use_forecast=False
        - Forecast predictions (y_forecast) if use_forecast=True
        
        NOTE: Validates that all test districts exist in training data to prevent
        silent misalignment that could compromise prediction quality.
        
        Args:
            X: Feature matrix (for API compatibility)
            df: DataFrame with metadata
            use_forecast: If True, use y_forecast; if False, use y_rep
            
        Returns:
            Array of outbreak probabilities
            
        Raises:
            ValueError: If test districts not seen during training or forecast not available
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Determine which predictions to use
        if use_forecast:
            if self.forecast_df_ is None:
                raise ValueError(
                    "Forecast predictions requested but model was not fitted with forecast_df. "
                    "Call fit() with forecast_df argument."
                )
            y_samples = self.fit_.stan_variable('y_forecast')  # Shape: (n_samples * n_chains, N_forecast)
            y_train = self.data_['y']
        else:
            # Use training predictions
            if df is not None:
                # Validate test districts exist in training
                test_districts = set(df['state'] + '_' + df['district'])
                train_districts = set(self.district_map_['state'] + '_' + self.district_map_['district'])
                unseen = test_districts - train_districts
                if unseen:
                    raise ValueError(
                        f"Test set contains {len(unseen)} districts not in training: "
                        f"{list(unseen)[:5]}{'...' if len(unseen) > 5 else ''}"
                    )
            
            y_samples = self.fit_.stan_variable('y_rep')  # Shape: (n_samples * n_chains, N)
            y_train = self.data_['y']
        
        # Compute probability of exceeding config-driven percentile of training data.
        # Use all cases (including zeros) and enforce minimum threshold 1.0 to
        # align with lead-time outbreak threshold logic.
        if self.outbreak_percentile is None:
            raise ValueError("outbreak_percentile must be provided via config")
        threshold = float(np.percentile(y_train, self.outbreak_percentile)) if len(y_train) else 1.0
        # Enforce minimum threshold to avoid classifying <1 case as outbreak
        threshold = enforce_minimum_threshold(threshold)
        
        # P(outbreak) = fraction of posterior samples exceeding threshold
        prob_outbreak = (y_samples > threshold).mean(axis=0)
        
        return prob_outbreak
    
    def forecast(
        self,
        test_df: Optional[pd.DataFrame] = None,
        n_draws: Optional[int] = None
    ) -> np.ndarray:
        """
        Get forecast predictions for test period.
        
        Extracts the y_forecast samples from the fitted model. The forecast
        must have been prepared during fit() by passing forecast_df argument.
        
        This propagates latent states Z forward from training period into test period
        using posterior samples of parameters (alpha, rho, sigma, beta_temp, phi).
        
        Args:
            test_df: Optional test DataFrame (for validation only)
            n_draws: Number of posterior draws to use (default: all)
            
        Returns:
            Array of shape (n_draws, N_forecast) with predicted case counts
            
        Raises:
            ValueError: If model not fitted with forecast capability
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        if self.forecast_df_ is None:
            raise ValueError(
                "Model was not fitted with forecast capability. "
                "Call fit() with forecast_df argument to enable forecasting."
            )
        
        # Validate test_df matches forecast setup if provided
        if test_df is not None:
            if len(test_df) != len(self.forecast_df_):
                warnings.warn(
                    f"test_df length ({len(test_df)}) differs from forecast setup "
                    f"({len(self.forecast_df_)}). Returning forecasts for fitted forecast data."
                )
        
        # Extract forecast samples
        y_forecast = self.fit_.stan_variable('y_forecast')  # (n_draws, N_forecast)
        
        if n_draws is not None and n_draws < y_forecast.shape[0]:
            # Subsample draws
            indices = np.random.choice(y_forecast.shape[0], size=n_draws, replace=False)
            y_forecast = y_forecast[indices]
        
        return y_forecast
    
    def forecast_proba(self, test_df: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Get outbreak probabilities for forecast period.
        
        Convenience method that calls predict_proba with use_forecast=True.
        
        Args:
            test_df: Optional test DataFrame (for validation)
            
        Returns:
            Array of outbreak probabilities for forecast samples
        """
        return self.predict_proba(X=None, df=test_df, use_forecast=True)

    def get_latent_risk_samples_per_observation(self) -> np.ndarray:
        """Return posterior samples of latent risk aligned to each observation row."""
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        if self.data_ is None:
            raise ValueError("Stan data not prepared.")

        Z = self.get_latent_states()  # (n_draws, D, T_max)
        district_idx = np.asarray(self.data_['district'], dtype=int) - 1
        time_idx = np.asarray(self.data_['time'], dtype=int) - 1
        return Z[:, district_idx, time_idx]

    def get_latent_risk_summary_per_observation(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (mean, sd) of latent risk Z for each observation row."""
        z_samples = self.get_latent_risk_samples_per_observation()
        return z_samples.mean(axis=0), z_samples.std(axis=0)
    
    def get_forecast_latent_risk_summary(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract latent risk Z summaries (mean, sd) for forecast period.
        
        This extracts Z_forecast from Stan's generated quantities block,
        which represents the latent transmission risk propagated forward
        from training into the test period using AR(1) dynamics.
        
        Returns:
            Tuple of (z_mean, z_sd) arrays with length N_forecast
            
        Raises:
            ValueError: If model not fitted with forecast capability
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        
        if self.forecast_df_ is None:
            raise ValueError(
                "Model was not fitted with forecast capability. "
                "Call fit() with forecast_df argument to enable forecasting."
            )
        
        # Extract Z_forecast from Stan: shape (n_draws, D, T_forecast)
        # This is the latent risk propagated into the forecast period
        try:
            Z_forecast = self.fit_.stan_variable('Z_forecast')
        except Exception as e:
            raise RuntimeError(
                f"Failed to extract Z_forecast from Stan model. "
                f"Ensure Stan model generates Z_forecast in generated quantities block."
            ) from e
        
        # Get forecast metadata
        district_forecast = self.data_['district_forecast']
        time_forecast = self.data_['time_forecast']
        T_max = self.data_['T_max']  # Max training time
        
        N_forecast = len(district_forecast)
        n_draws = Z_forecast.shape[0]
        
        # Extract Z for each forecast observation
        # Z_forecast[draw, district, time_offset] where time_offset = t - T_max
        z_forecast_samples = np.zeros((n_draws, N_forecast))
        
        for n in range(N_forecast):
            d = district_forecast[n] - 1  # Convert to 0-indexed
            t = time_forecast[n]
            t_offset = t - T_max - 1  # Offset into forecast period, 0-indexed
            
            # Extract samples for this observation
            z_forecast_samples[:, n] = Z_forecast[:, d, t_offset]
        
        # Compute mean and sd across draws
        z_mean = z_forecast_samples.mean(axis=0)
        z_sd = z_forecast_samples.std(axis=0)
        
        return z_mean, z_sd
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Get MCMC diagnostics.
        
        Returns:
            Dictionary with R-hat, ESS, divergences, etc.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        
        # Get summary statistics
        summary = self.fit_.summary()
        
        # Key parameters to check
        key_params = ['mu_alpha', 'sigma_alpha', 'rho', 'beta_temp', 'sigma', 'phi']
        
        # Get divergences from diagnostic output
        try:
            diag_output = self.fit_.diagnose()
            n_divergences = diag_output.count('divergent') if diag_output else 0
        except:
            n_divergences = "N/A"
        
        diagnostics = {
            'n_divergences': n_divergences,
            'max_rhat': summary['R_hat'].max(),
            'min_ess_bulk': summary['ESS_bulk'].min(),
            'min_ess_tail': summary['ESS_tail'].min(),
            'parameter_summary': {}
        }
        
        # Get summary for key parameters
        for param in key_params:
            if param in summary.index:
                row = summary.loc[param]
                diagnostics['parameter_summary'][param] = {
                    'mean': row['Mean'],
                    'std': row['StdDev'],
                    'rhat': row['R_hat'],
                    'ess_bulk': row['ESS_bulk']
                }
        
        return diagnostics
    
    def get_latent_states(self) -> np.ndarray:
        """
        Get posterior samples of latent states Z.
        
        Returns:
            Array of shape (n_samples, D, T_max)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        
        return self.fit_.stan_variable('Z')
    
    def get_posterior_predictive(self) -> np.ndarray:
        """
        Get posterior predictive samples.
        
        Returns:
            Array of shape (n_samples, N)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        
        return self.fit_.stan_variable('y_rep')
    
    def print_diagnostics(self) -> None:
        """Print formatted diagnostics summary."""
        diag = self.get_diagnostics()
        
        print("\n" + "=" * 50)
        print("MCMC DIAGNOSTICS")
        print("=" * 50)
        
        print(f"\nDivergences: {diag['n_divergences']}")
        print(f"Max R-hat: {diag['max_rhat']:.4f}")
        print(f"Min ESS (bulk): {diag['min_ess_bulk']:.0f}")
        print(f"Min ESS (tail): {diag['min_ess_tail']:.0f}")
        
        print("\nParameter Estimates:")
        print("-" * 50)
        print(f"{'Parameter':<15} {'Mean':>10} {'Std':>10} {'R-hat':>8} {'ESS':>8}")
        print("-" * 50)
        
        for param, vals in diag['parameter_summary'].items():
            print(f"{param:<15} {vals['mean']:>10.3f} {vals['std']:>10.3f} "
                  f"{vals['rhat']:>8.3f} {vals['ess_bulk']:>8.0f}")
        
        # Diagnostic flags
        print("\n" + "-" * 50)
        if diag['n_divergences'] > 0:
            print("⚠️  WARNING: Divergences detected!")
        if diag['max_rhat'] > 1.05:
            print("⚠️  WARNING: R-hat > 1.05 (chains may not have converged)")
        if diag['min_ess_bulk'] < 100:
            print("⚠️  WARNING: Low ESS (< 100)")
        
        if diag['n_divergences'] == 0 and diag['max_rhat'] <= 1.05 and diag['min_ess_bulk'] >= 100:
            print("✓ All diagnostics passed")
