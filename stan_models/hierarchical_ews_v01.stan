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
  
  // Observation dispersion (correct Jacobian for softplus)
  target += gamma_lpdf(phi | 2, 0.5) + log_inv_logit(phi_raw);
  
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
  // Posterior predictive samples
  array[N] int y_rep;
  
  // Log-likelihood for model comparison (WAIC/LOO)
  vector[N] log_lik;
  
  for (n in 1:N) {
    int d = district[n];
    int t = time[n];
    real log_mu = Z[d, t] + beta_temp * temp_anomaly[n];
    
    // Posterior predictive
    y_rep[n] = neg_binomial_2_log_rng(log_mu, phi);
    
    // Log-likelihood
    log_lik[n] = neg_binomial_2_log_lpmf(y[n] | log_mu, phi);
  }
}
