/*
 * Hierarchical State-Space Model for Chikungunya Early Warning
 * Version: 0.1 (Minimal Working Model)
 * 
 * Latent state Z_{d,t} represents log-transmission risk.
 * Observations Y_{d,t} are cases (Negative Binomial).
 * 
 * Reference: Phase 4 design specification
 */

data {
  int<lower=1> N;                    // Total observations
  int<lower=1> D;                    // Number of districts
  int<lower=1> T_max;                // Maximum time points
  
  array[N] int<lower=1, upper=D> district;  // District index for each obs
  array[N] int<lower=1, upper=T_max> time;  // Time index for each obs
  array[N] int<lower=0> y;                  // Observed cases (counts)
  array[N] real temp_anomaly;               // Temperature anomaly (lagged)
  
  // Forecast inputs (for temporal prediction beyond training period)
  int<lower=0> N_forecast;               // Number of forecast observations
  array[N_forecast] int<lower=1, upper=D> district_forecast;
  array[N_forecast] int<lower=T_max+1> time_forecast;
  array[N_forecast] real temp_anomaly_forecast;
}

parameters {
  // Hierarchical intercepts (non-centered parameterization)
  real mu_alpha;                     // Population mean baseline
  real<lower=0, upper=5> sigma_alpha; // Between-district SD
  vector[D] alpha_raw;               // Raw district effects (std normal)
  
  // AR(1) coefficient
  real<lower=0, upper=0.99> rho;     // Persistence of risk
  
  // Climate effect
  real beta_temp;                    // Temperature coefficient
  
  // Process noise
  real<lower=0, upper=5> sigma;      // Innovation SD
  
  // Observation model
  real<lower=-20, upper=20> phi_raw; // Bounded to avoid softplus under/overflow
  
  // Latent states (excluding Z_0 which is fixed at 0)
  matrix[D, T_max] z_raw;            // Raw innovations for latent states
}

transformed parameters {
  // Non-centered district intercepts
  vector[D] alpha = mu_alpha + sigma_alpha * alpha_raw;

  // NegBin dispersion (softplus for stability)
  real phi = log1p_exp(phi_raw);
  
  // Latent log-risk states
  matrix[D, T_max] Z;
  
  // Initialize and propagate latent states
  for (d in 1:D) {
    // Z_{d,0} = 0 for identifiability (implicit, we start from t=1)
    // Z_{d,1} = alpha_d + sigma * z_raw[d,1]
    Z[d, 1] = alpha[d] + sigma * z_raw[d, 1];
    
    // AR(1) dynamics for t > 1
    for (t in 2:T_max) {
      Z[d, t] = alpha[d] + rho * (Z[d, t-1] - alpha[d]) + sigma * z_raw[d, t];
    }
  }
}

model {
  // ===== PRIORS =====
  
  // Hierarchical intercept
  mu_alpha ~ normal(0, 2);
  sigma_alpha ~ normal(0, 1);        // Half-normal (constrained positive)
  alpha_raw ~ std_normal();          // Non-centered parameterization
  
  // AR coefficient (expect persistence)
  rho ~ normal(0.7, 0.15);
  
  // Climate effect (weakly informative)
  beta_temp ~ normal(0, 0.5);
  
  // Process noise
  sigma ~ normal(0, 0.5);            // Half-normal
  
  // Observation dispersion (softplus transformation with correct Jacobian)
  // Correct Jacobian for softplus transformation: d/dx[log(1+exp(x))] = exp(x)/(1+exp(x))
  // Log-Jacobian = log(inv_logit(x)) = x - log(1+exp(x))
  target += gamma_lpdf(phi | 2, 0.5) + phi_raw - log1p_exp(phi_raw);
  
  // Latent state innovations
  to_vector(z_raw) ~ std_normal();
  
  // ===== LIKELIHOOD =====
  
  for (n in 1:N) {
    int d = district[n];
    int t = time[n];
    
    // Expected log-rate includes climate effect
    real log_mu = Z[d, t] + beta_temp * temp_anomaly[n];
    
    // Negative Binomial likelihood
    // Using mean-dispersion parameterization
    y[n] ~ neg_binomial_2_log(log_mu, phi);
  }
}

generated quantities {
  // Posterior predictive samples for training data
  array[N] int y_rep;
  
  // Log-likelihood for model comparison (WAIC/LOO)
  vector[N] log_lik;
  
  // Forecast for test period (temporal extrapolation)
  array[N_forecast] int y_forecast;          // Predicted cases
  vector[N_forecast] log_lik_forecast;       // Log-likelihood for forecast
  
  // FIX: Declare Z_forecast at top level so Stan saves it as output
  // Must be declared before use, with conditional initialization below
  matrix[D, N_forecast > 0 ? (max(time_forecast) - T_max) : 0] Z_forecast;
  
  // Training period posterior predictive
  for (n in 1:N) {
    int d = district[n];
    int t = time[n];
    real log_mu = Z[d, t] + beta_temp * temp_anomaly[n];
    
    // Posterior predictive
    y_rep[n] = neg_binomial_2_log_rng(log_mu, phi);
    
    // Log-likelihood
    log_lik[n] = neg_binomial_2_log_lpmf(y[n] | log_mu, phi);
  }
  
  // Forecast period: propagate latent states forward using AR(1)
  if (N_forecast > 0) {
    int T_forecast_max = max(time_forecast);
    // Z_forecast already declared above at top level
    
    // For each district, continue AR(1) dynamics forward
    for (d in 1:D) {
      for (t_new in (T_max+1):T_forecast_max) {
        int t_idx = t_new - T_max;
        real z_prev = (t_new == T_max + 1) ? Z[d, T_max] : Z_forecast[d, t_idx - 1];
        
        // AR(1): Z_t = alpha + rho * (Z_{t-1} - alpha) + sigma * epsilon_t
        Z_forecast[d, t_idx] = alpha[d] + rho * (z_prev - alpha[d]) + sigma * normal_rng(0, 1);
      }
    }
    
    // Generate forecast observations
    for (n in 1:N_forecast) {
      int d = district_forecast[n];
      int t = time_forecast[n];
      int t_idx = t - T_max;
      real log_mu = Z_forecast[d, t_idx] + beta_temp * temp_anomaly_forecast[n];
      
      y_forecast[n] = neg_binomial_2_log_rng(log_mu, phi);
      log_lik_forecast[n] = neg_binomial_2_log_lpmf(y_forecast[n] | log_mu, phi);
    }
  }
}
