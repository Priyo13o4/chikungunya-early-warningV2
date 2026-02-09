"""
Data Loader for Chikungunya EWS - BLOCK 1: Data Acquisition

This module handles:
1. Loading EpiClim disease + climate data
2. Loading India Census 2011 population data
3. Merging datasets to create panel with incidence rates

Sources:
- EpiClim: https://www.epiclim.org/
- Census 2011: GitHub (nishusharma1608/India-Census-2011-Analysis)
"""
import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from rapidfuzz import fuzz, process


def parse_week_string(week_str: str) -> int:
    """
    Parse week string like "1st week", "2nd week", "10th week" to integer.
    
    Args:
        week_str: String like "1st week", "22nd week", etc.
        
    Returns:
        Integer week number (1-53)
    """
    if pd.isna(week_str):
        return np.nan
    
    # Extract number from string
    match = re.search(r'(\d+)', str(week_str))
    if match:
        return int(match.group(1))
    return np.nan


def load_epiclim(
    path: str,
    disease_filter: str = "Chikungunya",
    convert_temp: bool = True,
    imputation_strategy: str = "zero_fill"
) -> pd.DataFrame:
    """
    Load EpiClim CSV and filter for specified disease.
    
    Args:
        path: Path to Epiclim_Final_data.csv
        disease_filter: Disease name to filter (default: "Chikungunya")
        convert_temp: If True, convert Kelvin to Celsius
        imputation_strategy: Strategy for missing cases ('zero_fill', 'forward_fill', 'drop')
        
    Returns:
        DataFrame with columns: state, district, year, week, cases, deaths,
                               lat, lon, preci, lai, temp_celsius
    """
    df = pd.read_csv(path)
    
    # Filter for disease (strict match)
    df = df[df['Disease'] == disease_filter].copy()
    
    # Parse week string to integer
    df['week'] = df['week_of_outbreak'].apply(parse_week_string)
    
    # Convert cases to numeric (handle any non-numeric values)
    df['cases'] = pd.to_numeric(df['Cases'], errors='coerce')
    
    # Apply imputation strategy (config-driven)
    if imputation_strategy == "zero_fill":
        df['cases'] = df['cases'].fillna(0).astype(int)
    elif imputation_strategy == "forward_fill":
        df['cases'] = df.groupby(['state', 'district'])['cases'].ffill().fillna(0).astype(int)
    elif imputation_strategy == "drop":
        # Will be handled by caller
        pass
    else:
        raise ValueError(f"Unknown imputation_strategy: {imputation_strategy}")
    
    # Convert deaths to numeric
    df['deaths'] = pd.to_numeric(df['Deaths'], errors='coerce')
    
    # Temperature: Kelvin to Celsius
    if convert_temp and 'Temp' in df.columns:
        df['temp_celsius'] = df['Temp'] - 273.15
    else:
        df['temp_celsius'] = df.get('Temp', np.nan)
    
    # Standardize column names
    df = df.rename(columns={
        'state_ut': 'state',
        'district': 'district',
        'year': 'year',
        'Latitude': 'latitude',
        'Longitude': 'longitude',
        'preci': 'precipitation_mm',
        'LAI': 'lai'
    })
    
    # Select and order columns
    cols = [
        'state', 'district', 'year', 'week', 
        'cases', 'deaths',
        'latitude', 'longitude',
        'precipitation_mm', 'lai', 'temp_celsius'
    ]
    df = df[[c for c in cols if c in df.columns]]
    
    # Sort by state, district, year, week
    df = df.sort_values(['state', 'district', 'year', 'week']).reset_index(drop=True)
    
    return df


def load_census(path: str) -> pd.DataFrame:
    """
    Load India Census 2011 population data.
    
    Args:
        path: Path to india_census_2011_district.csv
        
    Returns:
        DataFrame with columns: state, district, population, district_code
    """
    df = pd.read_csv(path)
    
    # Standardize column names and case
    df = df.rename(columns={
        'State name': 'state',
        'District name': 'district', 
        'Population': 'population',
        'District code': 'district_code'
    })
    
    # Convert state/district to title case for matching
    df['state'] = df['state'].str.title()
    df['district'] = df['district'].str.title()
    
    # Select key columns
    cols = ['state', 'district', 'population', 'district_code']
    if 'Male' in df.columns and 'Female' in df.columns:
        cols.extend(['Male', 'Female'])
    
    return df[[c for c in cols if c in df.columns]]


def fuzzy_match_district(
    epiclim_district: str,
    epiclim_state: str,
    census_df: pd.DataFrame,
    score_threshold: int
) -> Tuple[Optional[str], Optional[int]]:
    """
    Fuzzy match district name from EpiClim to Census data.
    
    Args:
        epiclim_district: District name from EpiClim
        epiclim_state: State name from EpiClim (for filtering)
        census_df: Census DataFrame with 'state', 'district', 'population'
        score_threshold: Minimum fuzzy match score (config-driven)
        
    Returns:
        Tuple of (matched_district_name, population) or (None, None)
    """
    # Normalize inputs
    epiclim_district = str(epiclim_district).strip().title()
    epiclim_state = str(epiclim_state).strip().title()
    
    # Filter census to same state (fuzzy match state too)
    state_matches = census_df[
        census_df['state'].str.lower().str.contains(epiclim_state.lower()[:4], na=False)
    ]
    
    if len(state_matches) == 0:
        # Try fuzzy state match
        all_states = census_df['state'].unique()
        state_match = process.extractOne(epiclim_state, all_states, scorer=fuzz.ratio)
        if state_match and state_match[1] >= score_threshold:
            state_matches = census_df[census_df['state'] == state_match[0]]
    
    if len(state_matches) == 0:
        return None, None
    
    # Fuzzy match district within state
    districts = state_matches['district'].tolist()
    match = process.extractOne(epiclim_district, districts, scorer=fuzz.ratio)
    
    if match and match[1] >= score_threshold:
        matched_row = state_matches[state_matches['district'] == match[0]].iloc[0]
        return matched_row['district'], matched_row['population']
    
    return None, None


def merge_epiclim_census(
    epiclim_df: pd.DataFrame,
    census_df: pd.DataFrame,
    growth_rate: float = 0.02,
    target_year: int = 2020,
    score_threshold: int = None
) -> pd.DataFrame:
    """
    Merge EpiClim data with Census population data.
    Applies population growth projection from 2011 to target year.
    
    Args:
        epiclim_df: EpiClim DataFrame from load_epiclim()
        census_df: Census DataFrame from load_census()
        growth_rate: Annual population growth rate (default 2%)
        target_year: Year to project population to (default 2020)
        
    Returns:
        Merged DataFrame with population and incidence_per_100k
    """
    # Create unique district-state combinations
    districts = epiclim_df[['state', 'district']].drop_duplicates()
    
    # Match each district
    matched = []
    for _, row in districts.iterrows():
        if score_threshold is None:
            raise ValueError("score_threshold must be provided via config or caller")
        census_district, pop = fuzzy_match_district(
            row['district'], row['state'], census_df, score_threshold=score_threshold
        )
        matched.append({
            'state': row['state'],
            'district': row['district'],
            'census_district': census_district,
            'population_2011': pop
        })
    
    match_df = pd.DataFrame(matched)
    
    # Project population to target year
    years_since_2011 = target_year - 2011
    match_df['population'] = match_df['population_2011'] * ((1 + growth_rate) ** years_since_2011)
    match_df['population'] = match_df['population'].round().astype('Int64')  # Nullable int
    
    # Merge back to epiclim
    merged = epiclim_df.merge(
        match_df[['state', 'district', 'population', 'census_district']],
        on=['state', 'district'],
        how='left'
    )
    
    # Compute incidence per 100k
    merged['incidence_per_100k'] = (
        merged['cases'] / merged['population'] * 100000
    ).round(4)
    
    # Handle infinite/NaN incidence (zero population)
    merged['incidence_per_100k'] = merged['incidence_per_100k'].replace(
        [np.inf, -np.inf], np.nan
    )
    
    return merged


def build_panel(
    epiclim_path: str,
    census_path: str,
    disease: str = "Chikungunya",
    cases_imputation_strategy: str = "zero_fill",
    output_path: Optional[str] = None,
    score_threshold: Optional[int] = None
) -> pd.DataFrame:
    """
    Build complete panel dataset from raw sources.
    
    Args:
        epiclim_path: Path to EpiClim CSV
        census_path: Path to Census CSV
        disease: Disease to filter
        output_path: If provided, save panel to this path (parquet)
        
    Returns:
        Panel DataFrame ready for feature engineering
    """
    print(f"Loading EpiClim data for {disease}...")
    epiclim = load_epiclim(
        epiclim_path,
        disease_filter=disease,
        imputation_strategy=cases_imputation_strategy,
    )
    print(f"  → {len(epiclim)} rows loaded")
    
    print("Loading Census 2011 data...")
    census = load_census(census_path)
    print(f"  → {len(census)} districts loaded")
    
    print("Merging datasets with fuzzy matching...")
    if score_threshold is None:
        raise ValueError("score_threshold must be provided via config or caller")
    panel = merge_epiclim_census(epiclim, census, score_threshold=score_threshold)
    
    # Report matching stats
    matched = panel['population'].notna().sum()
    total = len(panel)
    print(f"  → {matched}/{total} rows matched ({100*matched/total:.1f}%)")
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(output_path, index=False)
        print(f"  → Saved to {output_path}")
    
    return panel


if __name__ == "__main__":
    # Quick test
    from src.config import get_project_root
    
    root = get_project_root()
    panel = build_panel(
        epiclim_path=root / "data/raw/Epiclim_Final_data.csv",
        census_path=root / "data/raw/india_census_2011_district.csv", 
        output_path=root / "data/processed/panel_chikungunya_v01.parquet"
    )
    print(f"\nPanel shape: {panel.shape}")
    print(panel.head())
