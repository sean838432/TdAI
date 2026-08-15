'''
This script sets up the training datasets for a Gradient Boosted Decision Tree (GBDT) algorithm designed to predict NBM Td error at 21z.
The training dataset comprises of 6 years of HRRR, NBM, and ASOS data from 2021-2026

CHANGES FROM TdAI v3.0:
1) The model is trained on KFVE, KHUL, KMLT, KGNR, and KBGR data (5 new stations) instead of just KCAR
2) HRRR top-layer (0-1cm) soil moisture is added as a predictor (MSLP was tried and dropped - ablation showed it added no more than noise-level value)
3) Removed cases where NBM Td error >= 3 and sub-800mb RH > 80% from the training dataset (to remove moist bust cases not driven by BL dynamics)

CHANGES FROM v2.2:
1) NBM Td is removed as a predictor (unnecessary since we have RH as a predictor)
2) Restricted the dataset to 2021-2026 which eliminates the use of HRRRv3 in 2020
3) Developed separate models for each operational cycle of TdAI (0245z_Day1, 0245z_Day2, 1445z_Day1, 1445z_Day2)

CHANGES FROM V2.1:
1) We train only on the 21z data. This provided big improvements because now we are only focusing on the time of maximum mixing and lowest RH
2) No re-distribution of the dataset is done (no random removal of quiet days and no sky/temp/RH/LPW filtering of the dataset)
3) Training data from 2020-2024 & 2026 is used. 2025 is the validation dataset
4) A smaller number of trees and smaller max depth of trees is used to account for the smaller dataset (resulting from only using 21z data). This prevents overfitting
5) An operational validation framework has been added to assess model performance if TdAI was run only under set weather conditions (i.e. when a bust was most likely)

    FEATURE VARIABLES:
        NBM Temperature (C)
        NBM RH (%)
        NBM Sky (%)
        NBM Mixing Height (100s of ft AGL)
        NBM Wind Speed (kts)
        NBM Wind Direction (deg)
        HRRR PWAT
        HRRR 1000mb-850mb Lapse Rate (C/km)
        HRRR 850mb-500mb Lapse Rate (C/km)
        HRRR 10cm Soil Moisture
        HRRR MSLP
        HRRR RH at all levels (%)
        Time of year

    OUTCOME VARIABLE:
        Td error from the 01z/13z NBM forecast

    WEIGHTING SCHEME:
        Td error 3-4 F: Weight of 2
        Td error >= 5 F: Weight of 5

    BUST PROPORTIONALITY SCHEME:
        None
'''

import numpy as np
import pandas as pd
import random
import os


####################################################################
#                                                                  #
#                             FUNCTIONS                            #
#                                                                  #
####################################################################


def seed_everything(seed=42):
    # 1. Set Python's built-in random seed
    random.seed(seed)
    # 2. Set Numpy's seed (Crucial for your jittering/noise)
    np.random.seed(seed)
    # 3. Set environment variable for any OS-level randomness
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"✅ Random state set at {seed}")

seed_everything(42)


def calculate_lpw_vectorized(df, all_levels):
    """
    Calculates Integrated Water Vapor (LPW) in millimeters (mm) across all
    specified pressure levels simultaneously using high-speed vectorized NumPy operations.
    Designed explicitly for input columns stored natively in Celsius.
    """
    g = 9.80665
    rho_w = 1000.0

    # Pre-allocate specific humidity arrays for each level across all rows
    q_matrix = []

    for lvl in all_levels:
        p = float(lvl)
        dpt_col = f'dpt_{lvl}'  # Ambient temp (t_col) is not mathematically required for LPW depth

        # --- AUGUST-ROCHE-MAGNUS FORMULATION (https://en.wikipedia.org/wiki/Clausius%E2%80%93Clapeyron_relation#Meteorology_and_climatology) ---
        # Takes dpt in Celsius
        e = 6.1094 * np.exp((17.625 * (df[dpt_col])) / (df[dpt_col] + 243.04))

        # Calculate mixing ratio (w) and specific humidity (q) https://glossary.ametsoc.org/wiki/specific-humidity/
        w = 0.622 * e / (p - e)
        q = w / (1.0 + w)

        q_matrix.append(q.values)

    # Shape of matrix: (num_levels, num_rows)
    q_matrix = np.array(q_matrix)

    # Initialize the integration output array
    lpw_total = np.zeros(len(df))

    # Hydrostatic integration layer-by-layer
    for i in range(len(all_levels) - 1):
        p_high = float(all_levels[i])   # Higher pressure (e.g., 1000 hPa)
        p_low = float(all_levels[i+1])  # Lower pressure (e.g., 925 hPa)

        # Pressure differential converted to Pascals (1 hPa = 100 Pa)
        dp = (p_high - p_low) * 100.0

        # Average specific humidity across the bounded layer
        q_avg = (q_matrix[i] + q_matrix[i+1]) / 2.0

        # Hydrostatic integration equation mapping to mm depth
        lpw_total += (q_avg * dp) / (g * rho_w) * 1000.0

    return lpw_total


def calculate_lapse_rate_vectorized(df, p_bottom, p_top):
  """
  Calculates the vertical temperature lapse rate (°C/km) between any two pressure
  levels dynamically using the Hypsometric Equation to determine layer thickness.

  Parameters:
  -----------
  df : pandas.DataFrame
      The pivoted HRRR DataFrame containing columns in 't_LEVEL' format (Celsius).
  p_bottom : int
      The pressure level at the bottom of the layer (e.g., 1000).
  p_top : int
      The pressure level at the top of the layer (e.g., 850).

  Returns:
  --------
  pandas.Series or np.nan
      A vectorized series representing the lapse rate in °C/km.
  """
  t_bottom_col = f't_{p_bottom}'
  t_top_col = f't_{p_top}'

  # 1. Enforce safety validation check for missing levels
  if t_bottom_col not in df.columns or t_top_col not in df.columns:
      return np.nan

  # 2. Extract ambient temperatures at the boundaries (assumed already in Celsius)
  t_bottom = df[t_bottom_col]
  t_top = df[t_top_col]

  # 3. Calculate raw temperature difference (Bottom - Top)
  # Positive = temperature cooling with height
  delta_t = t_bottom - t_top

  # 4. DYNAMIC GEOPOTENTIAL THICKNESS (The Hypsometric Equation)
  # Rd = 287.05 J/(kg*K) [Gas constant for dry air]
  # g  = 9.80665 m/s^2  [Standard gravitational acceleration]
  # Convert mean layer temperature to Kelvin
  t_mean_k = ((t_bottom + t_top) / 2.0) + 273.15

  # Calculate layer thickness in meters, then divide by 1000.0 for kilometers (dz_km)
  dz_meters = (287.05 * t_mean_k / 9.80665) * np.log(float(p_bottom) / float(p_top))
  dz_km = dz_meters / 1000.0

  # 5. Compute the lapse rate
  lapse_rate = delta_t / dz_km

  return lapse_rate


####################################################################
#                                                                  #
#                      COMPILE TRAINING DATASET                    #
#                                                                  #
####################################################################

base_path = "/home/sean834/TdAI/"
data_path = os.path.join(base_path, "data_download/")
trained_models_path = os.path.join(base_path, "model_training/training_dataset/")
STATIONS = ['CAR', 'HUL', 'MLT', 'GNR', 'BGR', 'FVE']

# 📅 Define dataset date thresholds (YYYY-MM-DD)
START_DATE = '2021-01-01' # To restrict dataset to HRRRv4 only
END_DATE = '2026-12-31'

# Define the 4 target operational cycles mapping inputs and target Day Offsets
CYCLES = [
    {
        'name': '15z_Day1',
        'hrrr_file': '12z_f09_Soundings.parquet',
        'nbm_file': '13z.csv',
        'target_day_offset': 0  # Day 1 (Valid Date == Init Date)
    },
    {
        'name': '15z_Day2',
        'hrrr_file': '12z_f33_Soundings.parquet',
        'nbm_file': '13z.csv',
        'target_day_offset': 1  # Day 2 (Valid Date == Init Date + 1)
    },
    {
        'name': '03z_Day1',
        'hrrr_file': '00z_f21_Soundings.parquet',
        'nbm_file': '01z.csv',
        'target_day_offset': 0  # Day 1 (Valid Date == Init Date)
    },
    {
        'name': '03z_Day2',
        'hrrr_file': '00z_f45_Soundings.parquet',
        'nbm_file': '01z.csv',
        'target_day_offset': 1  # Day 2 (Valid Date == Init Date + 1)
    }
]

print(f"🚀 Starting Master Compilation for {len(STATIONS)} stations across {len(CYCLES)} distinct models...")

for station in STATIONS:
    print(f"\n────────────────── Processing Station: K{station} ──────────────────")

    # Dictionary to collect compiled dataframes separated by cycle
    cycle_master_dfs = {cycle['name']: [] for cycle in CYCLES}

    # --- 1. LOAD ASOS GROUND TRUTH (Only needs to be done once per station) ---
    asos_path = os.path.join(data_path, f"ASOS_data/K{station}_2020_to_2026_asos.csv")
    if not os.path.exists(asos_path):
        print(f"⚠️ Missing ASOS file for K{station}. Skipping this station entirely.")
        continue

    print(f"   Loading ASOS ground truth...")
    asos_df = pd.read_csv(asos_path)
    asos_df['valid_time'] = pd.to_datetime(asos_df['valid_time']).dt.round('h')
    asos_df = asos_df[['valid_time', 'ASOS Dewpoint (F)']].dropna()

    # --- 2. LOOP THROUGH EACH PREDICTION CYCLE ---
    for cycle in CYCLES:

        c_name = cycle['name']
        print(f"\n   ⚙️ Structuring Cycle: {c_name}...")

        hrrr_path = os.path.join(data_path, f"HRRR_forecast_soundings/K{station}/K{station}_{cycle['hrrr_file']}")
        nbm_path = os.path.join(data_path, f"NBM_data/NBM_Master_Data_K{station}_{cycle['nbm_file']}")

        if not (os.path.exists(hrrr_path) and os.path.exists(nbm_path)):
            print(f"      ⚠️ Missing HRRR or NBM file for {c_name}. Skipping cycle.")
            continue

        try:
            # --- A. LOAD & PREPARE NBM DATA ---
            nbm_df = pd.read_csv(nbm_path)
            nbm_df['valid_time'] = pd.to_datetime(nbm_df['NBM Forecast Valid (UTC)'])
            nbm_df['init_time'] = pd.to_datetime(nbm_df['NBM Initialization Time (UTC)'])

            # Extract just the date components to calculate the Day Offset (0 for Day 1, 1 for Day 2)
            nbm_df['day_offset'] = (nbm_df['valid_time'].dt.normalize() - nbm_df['init_time'].dt.normalize()).dt.days

            # Strictly isolate the correct day's forecast based on our cycle definition
            nbm_df = nbm_df[nbm_df['day_offset'] == cycle['target_day_offset']].copy()

            if nbm_df.empty:
                print(f"      ⚠️ No NBM data found matching Day Offset {cycle['target_day_offset']}. Skipping.")
                continue

            # REPLACE SINGLE-HOUR (21z) SKY COVER WITH THE 15z-21z AVERAGE
            nbm_df['valid_date'] = nbm_df['valid_time'].dt.normalize()
            sky_window = (nbm_df['valid_time'].dt.hour >= 15) & (nbm_df['valid_time'].dt.hour <= 21)
            cloud_cover_avg_by_date = nbm_df[sky_window].groupby('valid_date')['NBM Cloud Cover (%)'].mean()
            nbm_df['NBM Cloud Cover (%)'] = nbm_df['valid_date'].map(cloud_cover_avg_by_date)
            nbm_df = nbm_df.drop(columns=['valid_date'])

            # --- B. LOAD & PIVOT HRRR SOUNDING SAMPLES ---
            hrrr_long = pd.read_parquet(hrrr_path)
            hrrr_long['valid_time'] = pd.to_datetime(hrrr_long['valid_time'])

            # Vectorized pivot mapping 3D profile to 2D features
            hrrr_pivoted = hrrr_long.pivot(
                index='valid_time',
                columns='isobaricInhPa',
                values=['t', 'dpt']
            )

            # Flatten MultiIndex columns
            hrrr_pivoted.columns = [f"{v}_{int(p)}" for v, p in hrrr_pivoted.columns]
            hrrr_pivoted = hrrr_pivoted.reset_index()

            # --- C. METEOROLOGICAL CONVERSIONS ---
            thermal_cols = [c for c in hrrr_pivoted.columns if c.startswith('t_') or c.startswith('dpt_')]
            hrrr_pivoted[thermal_cols] = hrrr_pivoted[thermal_cols] - 273.15

            all_levels = sorted(
                [int(col.split('_')[1]) for col in hrrr_pivoted.columns if col.startswith('t_')],
                reverse=True
            )

            hrrr_pivoted['hrrr_lpw (mm)'] = calculate_lpw_vectorized(hrrr_pivoted, all_levels)

            rh_features_dict = {}
            for lvl in all_levels:
                t_col = f't_{lvl}'
                dpt_col = f'dpt_{lvl}'
                if t_col in hrrr_pivoted.columns and dpt_col in hrrr_pivoted.columns:
                    es = np.exp((17.625 * hrrr_pivoted[t_col]) / (243.04 + hrrr_pivoted[t_col]))
                    e = np.exp((17.625 * hrrr_pivoted[dpt_col]) / (243.04 + hrrr_pivoted[dpt_col]))
                    rh_features_dict[f'rh_{lvl}'] = np.clip(100 * (e / es), 0.0, 100.0)

            new_features_df = pd.DataFrame(rh_features_dict, index=hrrr_pivoted.index)
            hrrr_pivoted = pd.concat([hrrr_pivoted, new_features_df], axis=1)

            hrrr_pivoted['1000mb-700mb Lapse Rate (C/km)'] = calculate_lapse_rate_vectorized(hrrr_pivoted, 1000, 700)
            hrrr_pivoted['700mb-500mb Lapse Rate (C/km)'] = calculate_lapse_rate_vectorized(hrrr_pivoted, 700, 500)

            # --- D. THE NBM-ASOS-HRRR MERGE ---
            station_df = pd.merge(nbm_df, asos_df, on='valid_time', how='inner')
            station_df = pd.merge(station_df, hrrr_pivoted, on='valid_time', how='inner')

            station_df['Target Error (F)'] = station_df['NBM Dewpoint (F)'] - station_df['ASOS Dewpoint (F)']

            # --- E. CALCULATE AND ADD NBM RH ---
            station_df = station_df[pd.to_datetime(station_df['valid_time']).dt.hour == 21].copy()

            nbm_tc = (station_df['NBM Temperature (F)'] - 32) * (5.0 / 9.0)
            nbm_tdc = (station_df['NBM Dewpoint (F)'] - 32) * (5.0 / 9.0)
            nbm_es = np.exp((17.625 * nbm_tc) / (243.04 + nbm_tc))
            nbm_e = np.exp((17.625 * nbm_tdc) / (243.04 + nbm_tdc))
            station_df['NBM RH (%)'] = np.clip(100 * (nbm_e / nbm_es), 0.0, 100.0)

            # --- F. ELIMINATE UNNECESSARY COLUMNS ---
            columns_to_drop = [
                'NBM Initialization Time (UTC)',
                'NBM Forecast Valid (UTC)',
                'ASOS Dewpoint (F)',
                'NBM Dewpoint (F)',
                'init_time',
                'day_offset'
            ]
            raw_sounding_thermals = [c for c in station_df.columns if c.startswith('t_') or c.startswith('dpt_')]
            columns_to_drop.extend(raw_sounding_thermals)

            station_df = station_df.drop(columns=columns_to_drop, errors='ignore')
            station_df = station_df.dropna()

            print(f"      ✅ Generated {len(station_df)} valid samples for {c_name}.")

            # Append strictly to this cycle's list
            cycle_master_dfs[c_name].append(station_df)

        except Exception as e:
            print(f"      ❌ Failed to process cycle {c_name} for K{station}: {e}")
            continue


    # --- 3. EXPORT 4 INDIVIDUAL DATASETS ---

    # Define your strict column hierarchy
    TARGET_COLUMN_ORDER = [
        'Target Error (F)',
        'NBM Temperature (F)',
        'NBM Cloud Cover (%)',
        'NBM Mixing Height (100s ft AGL)',
        'NBM Wind Speed (kts)',
        'NBM Wind Direction (deg)',
        'NBM RH (%)',
        'hrrr_lpw (mm)',
        '1000mb-700mb Lapse Rate (C/km)',
        '700mb-500mb Lapse Rate (C/km)',
        'rh_1000', 'rh_975', 'rh_950', 'rh_925', 'rh_900', 'rh_875', 'rh_850',
        'rh_825', 'rh_800', 'rh_775', 'rh_750', 'rh_725', 'rh_700', 'rh_675',
        'rh_650', 'rh_625', 'rh_600', 'rh_575', 'rh_550', 'rh_525', 'rh_500',
        'sin_season',
        'cos_season'
    ]

    print("\n🔗 Generating Individual Training Datasets...")

    for cycle_name, df_list in cycle_master_dfs.items():
        if not df_list:
            print(f"   ⚠️ No data successfully compiled for {cycle_name}.")
            continue

        master_train_df = pd.concat(df_list, ignore_index=True)

        # Add shared engineering features
        day_of_year_series = master_train_df['valid_time'].dt.dayofyear
        master_train_df['sin_season'] = np.sin(2 * np.pi * day_of_year_series / 365.25)
        master_train_df['cos_season'] = np.cos(2 * np.pi * day_of_year_series / 365.25)

        # Set index and ensure datetime format
        master_train_df = master_train_df.set_index('valid_time')
        master_train_df.index = pd.to_datetime(master_train_df.index)

        # 📅 APPLY DATE THRESHOLD FILTER
        original_len = len(master_train_df)
        master_train_df = master_train_df.sort_index().loc[START_DATE:END_DATE]
        
        # 🚀 ENFORCE STRICT COLUMN ORDER
        # Filter to only columns that actually exist in the DataFrame to prevent KeyErrors
        existing_cols = [col for col in TARGET_COLUMN_ORDER if col in master_train_df.columns]
        master_train_df = master_train_df[existing_cols]

        output_filename = f"TdAI_Training_Data_{station}_{cycle_name}.csv"
        output_full_path = os.path.join(trained_models_path, output_filename)

        master_train_df.to_csv(output_full_path, index=True)
        print(f"   💾 Saved {cycle_name} Dataset: {len(master_train_df)} total samples -> {output_filename}")

print("=" * 70)
print("✨ ALL DONE! Individual training datasets successfully generated.")

