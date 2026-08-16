"""
TdAI Probabilistic Ingestion, Prediction, and Verification Pipeline
"""

import os
import io
import glob
import random
import datetime
import requests
import numpy as np
import pandas as pd
import xarray as xr
import joblib
import lightgbm

def seed_everything(seed=42):
    """Locks random states to enforce analytical reproducibility across iterations."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"✅ Random state locked at seed: {seed}")

# -------------------------------------------------------------------------
# 🗺️ STATION CONFIG - lat/lon sourced from data_download/hrrr_soundings_download.py
# so operational point-extraction matches the exact points training data was
# built from.
# -------------------------------------------------------------------------
STATIONS = {
    'CAR': (46.870490, -68.017221),
    'FVE': (47.285172, -68.307131),
    'HUL': (46.118457, -67.792894),
    'MLT': (45.647771, -68.692475),
    'GNR': (45.462979, -69.554546),
    'BGR': (44.807400, -68.828100),
}

QUANTILES = ['q10', 'q25', 'q50', 'q75', 'q90']

OUTPUT_HEADERS = [
    'valid_time', 'TdAI Run Time (UTC)', 'TdAI Status', 'NBM Dewpoint (F)',
    'TdAI_Predicted_Bias_q10', 'TdAI_Corrected_Dewpoint_q10',
    'TdAI_Predicted_Bias_q25', 'TdAI_Corrected_Dewpoint_q25',
    'TdAI_Predicted_Bias_q50', 'TdAI_Corrected_Dewpoint_q50',
    'TdAI_Predicted_Bias_q75', 'TdAI_Corrected_Dewpoint_q75',
    'TdAI_Predicted_Bias_q90', 'TdAI_Corrected_Dewpoint_q90',
    'ASOS Ground Truth Dewpoint (F)', 'Raw NBM Error (F)', 'Post TdAI Median Error (F)', 'TdAI Median Skill Score (%)'
]

# TdAI's models are only trained on March 1 - November 15 data (fire
# season) - outside that window there's no valid basis for a prediction, so
# operational runs are paused rather than extrapolating onto an unseen season.
WINTER_PAUSE_STATUS = "TdAI runs paused for the winter"

def is_winter_pause(check_date):
    """True for Nov 16 - Feb 29(28) inclusive. Checking the whole month of
    February (rather than a specific day 28/29 cutoff) naturally covers the
    leap-year boundary without special-casing it."""
    month, day = check_date.month, check_date.day
    if month in (12, 1, 2):
        return True
    if month == 11 and day >= 16:
        return True
    return False

def write_winter_pause_status(station, base_path):
    """Writes a placeholder Day1/Day2 status row (no forecast data) to a
    station's output CSV explaining that TdAI is paused for the off-season -
    reuses the same status-driven display path the dashboard already uses
    for gate-bypassed rows, so no dashboard changes are needed."""
    output_dir = os.path.join(base_path, "probabilistic_output/")
    os.makedirs(output_dir, exist_ok=True)
    output_csv_path = os.path.join(output_dir, f"TdAI_probabilistic_operational_{station}.csv")

    if os.path.exists(output_csv_path):
        combined_log_df = pd.read_csv(output_csv_path)
        for col in OUTPUT_HEADERS:
            if col not in combined_log_df.columns:
                combined_log_df[col] = np.nan
        combined_log_df['valid_time'] = combined_log_df['valid_time'].astype(str).str.strip()
    else:
        combined_log_df = pd.DataFrame(columns=OUTPUT_HEADERS)

    # Anchor to UTC, not the local machine's timezone - a script running on
    # a non-UTC machine near a UTC day boundary would otherwise write
    # placeholder rows for the wrong calendar date (see the equivalent bug
    # in main()'s HRRR/NBM date selection).
    current_time_utc = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    today = current_time_utc.date()
    target_valid_times = [
        datetime.datetime.combine(today, datetime.time(21, 0)),
        datetime.datetime.combine(today + datetime.timedelta(days=1), datetime.time(21, 0)),
    ]

    new_rows = []
    for vtime in target_valid_times:
        vtime_str = vtime.strftime('%Y-%m-%d %H:%M:%S')

        # Preserve any historical ASOS verification already recorded for
        # this valid_time (e.g. a prior in-season run) instead of blanking it.
        old_asos = np.nan
        if not combined_log_df.empty:
            existing_match = combined_log_df[combined_log_df['valid_time'] == vtime_str]
            if not existing_match.empty:
                old_asos = existing_match['ASOS Ground Truth Dewpoint (F)'].iloc[0]

        row = {
            'valid_time': vtime_str,
            'TdAI Run Time (UTC)': current_time_utc.strftime('%Y-%m-%d %H:%M UTC'),
            'TdAI Status': WINTER_PAUSE_STATUS,
            'NBM Dewpoint (F)': np.nan,
            'ASOS Ground Truth Dewpoint (F)': old_asos,
            'Raw NBM Error (F)': np.nan,
            'Post TdAI Median Error (F)': np.nan,
            'TdAI Median Skill Score (%)': np.nan,
        }
        for q in QUANTILES:
            row[f'TdAI_Predicted_Bias_{q}'] = 0.0
            row[f'TdAI_Corrected_Dewpoint_{q}'] = np.nan
        new_rows.append(row)

    new_entry_df = pd.DataFrame(new_rows)
    if not combined_log_df.empty:
        combined_log_df = combined_log_df[~combined_log_df['valid_time'].isin(new_entry_df['valid_time'])]

    combined_log_df = pd.concat([combined_log_df, new_entry_df], ignore_index=True)
    combined_log_df = combined_log_df.sort_values(by='valid_time').reset_index(drop=True)
    combined_log_df.to_csv(output_csv_path, index=False)
    print(f"❄️ K{station}: {WINTER_PAUSE_STATUS} -> {output_csv_path}")

def calculate_lpw_vectorized(df, all_levels):
    """Calculates Integrated Water Vapor (LPW) in mm across pressure levels using numpy math."""
    g = 9.80665
    rho_w = 1000.0
    q_matrix = []

    for lvl in all_levels:
        p = float(lvl)
        dpt_col = f'dpt_{lvl}'
        # August-Roche-Magnus formulation for actual vapor pressure
        e = 6.1094 * np.exp((17.625 * (df[dpt_col])) / (df[dpt_col] + 243.04))
        # Mixing ratio (w) and specific humidity (q) conversion
        w = 0.622 * e / (p - e)
        q = w / (1.0 + w)
        q_matrix.append(q.values)

    q_matrix = np.array(q_matrix)
    lpw_total = np.zeros(len(df))

    # Hydrostatic layer integration
    for i in range(len(all_levels) - 1):
        p_high = float(all_levels[i])
        p_low = float(all_levels[i+1])
        dp = (p_high - p_low) * 100.0  # Convert hPa to Pascals
        q_avg = (q_matrix[i] + q_matrix[i+1]) / 2.0
        lpw_total += (q_avg * dp) / (g * rho_w) * 1000.0

    return lpw_total

def calculate_lapse_rate_vectorized(df, p_bottom, p_top):
    """Calculates vertical temperature lapse rate (°C/km) via the Hypsometric Equation."""
    t_bottom_col = f't_{p_bottom}'
    t_top_col = f't_{p_top}'
    if t_bottom_col not in df.columns or t_top_col not in df.columns:
        return np.nan

    t_bottom = df[t_bottom_col]
    t_top = df[t_top_col]
    delta_t = t_bottom - t_top
    t_mean_k = ((t_bottom + t_top) / 2.0) + 273.15
    # Calculate geopotential layer thickness in meters
    dz_meters = (287.05 * t_mean_k / 9.80665) * np.log(float(p_bottom) / float(p_top))
    return delta_t / (dz_meters / 1000.0)

def download_hrrr_grib(date_str, run_hour='12', forecast_hour=0):
    """Downloads target HRRR GRIB2 parameters directly from NCEP NOMADS servers."""
    base_url = f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/hrrr/prod/hrrr.{date_str}/conus/"
    filename = f"hrrr.t{run_hour}z.wrfprsf{forecast_hour:02d}.grib2"
    url = base_url + filename
    print(f"Attempting to download HRRR GRIB: {url}")
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"   └── Successfully ingested temporary raster: {filename}")
        return filename
    except Exception as e:
        print(f"   ⚠️ NCEP server registry blocker on F{forecast_hour:02d}: {e}")
        return None

def get_nbm_bulletin(date_str, run_hour='13'):
    """Retrieves the NBM text blend terminal output for a specified run hour (all stations)."""
    base_url = f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/blend/prod/blend.{date_str}/{run_hour}/text/"
    url = base_url + f"blend_nbstx.t{run_hour}z"
    print(f"📡 Requesting NBM Text Feed: {url}")
    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            print(f"   └── Successfully fetched NBM text bulletin for {date_str} ({run_hour}Z).")
            return response.text
        print(f"   ⚠️ File not ready yet on server for date {date_str} (Status: {response.status_code})")
        return None
    except Exception as e:
        print(f"   ❌ NBM terminal network fault: {e}")
        return None

def extract_hrrr_point_profile(ds_filtered, lat, lon, valid_time_str, fhr):
    """Extracts the nearest-gridpoint vertical profile (500-1000 hPa) for one
    station from an already-opened HRRR isobaric dataset."""
    lon_360 = lon + 360.0 if lon < 0 else lon
    squared_distance = ((ds_filtered['latitude'] - lat) ** 2) + ((ds_filtered['longitude'] - lon_360) ** 2)
    y_idx, x_idx = np.unravel_index(np.nanargmin(squared_distance.to_numpy()), squared_distance.shape)

    ds_point = ds_filtered.isel(y=y_idx, x=x_idx)
    p_lvls = ds_point['isobaricInhPa'].to_numpy()
    mask = (p_lvls >= 500) & (p_lvls <= 1000)

    return pd.DataFrame({
        'valid_time': np.full(np.sum(mask), valid_time_str),
        'forecast_hour': np.full(np.sum(mask), fhr),
        'HRRR Pressure (hPa)': p_lvls[mask],
        'HRRR Temperature (K)': ds_point['t'].to_numpy()[mask],
        'HRRR Dewpoint (K)': ds_point['dpt'].to_numpy()[mask],
    })

def _parse_nbm_token(v):
    """Parses one NBM bulletin token to int/float. Tries int() directly
    (not str.isdigit(), which does NOT recognize a leading '-' - "-5".isdigit()
    is False, so a bare negative integer like "-5" was silently falling
    through to None instead of -5. This specifically corrupted negative
    Temperature/Dewpoint readings, which are common in this climate outside
    peak summer, into NaN downstream)."""
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return None

def parse_nbm_station_block(bulletin_text, station_id):
    """Isolates one station's block (e.g. 'KCAR') from the full NBM text
    bulletin and parses it into a DataFrame of all forecast hours (not yet
    restricted to 21Z - the 15-21Z cloud cover average needs the full block)."""
    lines = bulletin_text.split('\n')
    station_lines = []
    in_block = False
    for line in lines:
        if station_id in line and "NBM" in line:
            in_block = True
        if in_block:
            station_lines.append(line)
            if len(station_lines) > 2 and (line.strip() == "" or line.startswith('#') or "STATION" in line):
                if "STATION" in line or line.startswith('#'):
                    station_lines.pop()
                break

    if not station_lines:
        return None

    parsed_data = {}
    targets = {
        'UTC': 'UTC Hour', 'TMP': 'NBM Temperature (F)', 'DPT': 'NBM Dewpoint (F)',
        'SKY': 'NBM Cloud Cover (%)', 'WDR': 'NBM Wind Direction (tens deg)',
        'WSP': 'NBM Wind Speed (kts)', 'MHT': 'NBM Mixing Height (100s ft)',
    }
    for line in station_lines:
        tokens = line.split()
        if not tokens or tokens[0] not in targets:
            continue
        parsed_data[targets[tokens[0]]] = [_parse_nbm_token(v) for v in tokens[1:]]

    if not parsed_data:
        return None

    return pd.DataFrame(parsed_data)

def build_nbm_df(bulletin_text, station_id, successful_date_str):
    """Parses a station's NBM block into a fully-featured, all-hours
    DataFrame (valid_time, RH, wind direction/mixing height renamed) - the
    15-21Z cloud cover averaging happens on this before the 21Z-only filter
    is applied by the caller."""
    nbm_df = parse_nbm_station_block(bulletin_text, station_id)
    if nbm_df is None or 'UTC Hour' not in nbm_df.columns:
        return None

    init_date = datetime.datetime.strptime(successful_date_str, '%Y%m%d')
    valid_times = []
    curr_dt = init_date
    prev_hr = -1
    for hr in nbm_df['UTC Hour']:
        if hr is None:
            valid_times.append(pd.NaT)
            continue
        if prev_hr != -1 and hr < prev_hr:
            curr_dt += datetime.timedelta(days=1)
        valid_times.append(curr_dt.replace(hour=hr, minute=0, second=0, microsecond=0))
        prev_hr = hr

    nbm_df['valid_time'] = pd.to_datetime(valid_times)
    if 'NBM Wind Direction (tens deg)' in nbm_df.columns:
        nbm_df['NBM Wind Direction (deg)'] = nbm_df['NBM Wind Direction (tens deg)'] * 10
    if 'NBM Mixing Height (100s ft)' in nbm_df.columns:
        nbm_df['NBM Mixing Height (100s ft AGL)'] = nbm_df['NBM Mixing Height (100s ft)']

    return nbm_df

def process_station(station, lat, lon, target_run_hour, forecast_hours,
                     bulletin_text, successful_date_str, grib_files, base_path):
    """Runs the full ingest -> predict -> verify -> CSV-log pipeline for one
    station. Returns nothing; writes directly to that station's output CSV."""
    print(f"\n{'=' * 70}")
    print(f"🏢 STATION: K{station}")
    print(f"{'=' * 70}")

    output_dir = os.path.join(base_path, "probabilistic_output/")
    os.makedirs(output_dir, exist_ok=True)
    output_csv_path = os.path.join(output_dir, f"TdAI_probabilistic_operational_{station}.csv")

    # -------------------------------------------------------------------------
    # 🛰️ SECTION 1: EXTRACT THIS STATION'S HRRR PROFILE FROM THE SHARED GRIB FILES
    # -------------------------------------------------------------------------
    all_forecast_dfs = []
    for fhr, grib_file in grib_files.items():
        print(f"📊 Parsing vertical profile structures from temporary raster F{fhr:02d}...")
        with xr.open_dataset(grib_file, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa'}}) as ds_filtered:
            valid_time_str = pd.Timestamp(ds_filtered['valid_time'].values).strftime('%Y-%m-%d %H:%M:%S')
            all_forecast_dfs.append(extract_hrrr_point_profile(ds_filtered, lat, lon, valid_time_str, fhr))

    master_hrrr_profiles_df = pd.concat(all_forecast_dfs, ignore_index=True)
    t_c = master_hrrr_profiles_df['HRRR Temperature (K)'].astype(float) - 273.15
    dp_c = master_hrrr_profiles_df['HRRR Dewpoint (K)'].astype(float) - 273.15
    es = np.exp((17.625 * t_c) / (243.04 + t_c))
    e = np.exp((17.625 * dp_c) / (243.04 + dp_c))
    master_hrrr_profiles_df['HRRR RH (%)'] = round(np.clip(100 * (e / es), 0.0, 100.0), 1)

    # -------------------------------------------------------------------------
    # 📊 SECTION 2: PARSE THIS STATION'S NBM BLOCK + 15-21Z CLOUD COVER AVERAGE
    # -------------------------------------------------------------------------
    nbm_df = build_nbm_df(bulletin_text, f"K{station}", successful_date_str)
    if nbm_df is None:
        print(f"⚠️ Could not locate/parse an NBM block for K{station} in the bulletin. Skipping station.")
        return

    nbm_tc = (nbm_df['NBM Temperature (F)'] - 32) * (5.0 / 9.0)
    nbm_tdc = (nbm_df['NBM Dewpoint (F)'] - 32) * (5.0 / 9.0)
    nbm_es = np.exp((17.625 * nbm_tc) / (243.04 + nbm_tc))
    nbm_e = np.exp((17.625 * nbm_tdc) / (243.04 + nbm_tdc))
    nbm_df['NBM RH (%)'] = round(np.clip(100 * (nbm_e / nbm_es), 0.0, 100.0), 1)

    # Replace the single-hour (21Z) SKY cover with the 15Z-21Z average -
    # exact same logic as model_training/TdAI_v3.1_Training_Dataset_Compilation.py
    # so live inference sees the same feature the models were trained on.
    print("☁️ Replacing single-hour 21Z SKY cover with the 15Z-21Z average (matches training)...")
    nbm_df['valid_date'] = nbm_df['valid_time'].dt.normalize()
    sky_window = (nbm_df['valid_time'].dt.hour >= 15) & (nbm_df['valid_time'].dt.hour <= 21)
    cloud_cover_avg_by_date = nbm_df[sky_window].groupby('valid_date')['NBM Cloud Cover (%)'].mean()
    nbm_df['NBM Cloud Cover (%)'] = nbm_df['valid_date'].map(cloud_cover_avg_by_date)
    nbm_df = nbm_df.drop(columns=['valid_date'])

    print("⏰ Filtering output matrix arrays to parse 21Z peak afternoon mixing windows...")
    nbm_df = nbm_df[nbm_df['valid_time'].dt.hour == 21].copy()

    if nbm_df.empty:
        print(f"⚠️ No 21Z NBM rows available for K{station}. Skipping station.")
        return

    # -------------------------------------------------------------------------
    # 🔗 SECTION 3: MATRIX ALIGNMENT & METEOROLOGICAL EXPANSIONS
    # -------------------------------------------------------------------------
    master_hrrr_profiles_df['valid_time'] = pd.to_datetime(master_hrrr_profiles_df['valid_time'])
    valid_time_to_fhr = master_hrrr_profiles_df[['valid_time', 'forecast_hour']].drop_duplicates().set_index('valid_time')['forecast_hour']

    for var in ['HRRR Pressure (hPa)', 'HRRR Temperature (K)', 'HRRR Dewpoint (K)', 'HRRR RH (%)']:
        master_hrrr_profiles_df[var] = pd.to_numeric(master_hrrr_profiles_df[var], errors='coerce')

    hrrr_pivoted = master_hrrr_profiles_df.pivot(index='valid_time', columns='HRRR Pressure (hPa)', values=['HRRR Temperature (K)', 'HRRR Dewpoint (K)', 'HRRR RH (%)'])
    new_cols = [f"t_{int(float(l))}" if v == 'HRRR Temperature (K)' else f"dpt_{int(float(l))}" if v == 'HRRR Dewpoint (K)' else f"rh_{int(float(l))}" for v, l in hrrr_pivoted.columns]
    hrrr_pivoted.columns = new_cols
    hrrr_pivoted = hrrr_pivoted.reset_index()

    thermal_cols = [c for c in hrrr_pivoted.columns if c.startswith('t_') or c.startswith('dpt_')]
    hrrr_pivoted[thermal_cols] = hrrr_pivoted[thermal_cols] - 273.15

    all_levels = sorted([int(c.split('_')[1]) for c in hrrr_pivoted.columns if c.startswith('t_')], reverse=True)
    hrrr_pivoted['hrrr_lpw (mm)'] = calculate_lpw_vectorized(hrrr_pivoted, all_levels)
    hrrr_pivoted['1000mb-700mb Lapse Rate (C/km)'] = calculate_lapse_rate_vectorized(hrrr_pivoted, 1000, 700)
    hrrr_pivoted['700mb-500mb Lapse Rate (C/km)'] = calculate_lapse_rate_vectorized(hrrr_pivoted, 700, 500)

    master_input_df = pd.merge(nbm_df, hrrr_pivoted, on='valid_time', how='inner')
    if master_input_df.empty:
        print(f"⚠️ NBM/HRRR valid_time alignment produced zero matching rows for K{station}. Skipping station.")
        return

    master_input_df['forecast_hour'] = master_input_df['valid_time'].map(valid_time_to_fhr)

    doy = master_input_df['valid_time'].dt.dayofyear
    master_input_df['sin_season'] = np.sin(2 * np.pi * doy / 365.25)
    master_input_df['cos_season'] = np.cos(2 * np.pi * doy / 365.25)

    # -------------------------------------------------------------------------
    # 🔮 SECTION 4: PROBABILISTIC QUANTILE MULTI-PREDICTION ENGINE (PER-CYCLE MODELS)
    # -------------------------------------------------------------------------
    model_dir = os.path.join(base_path, "model_training", "trained_models", station)
    day1_fhr, day2_fhr = forecast_hours[0], forecast_hours[1]

    # The 00Z cycle (run overnight, ~03Z cron) maps to '03z_DayN' models; the
    # 12Z cycle (~15Z cron) maps to '15z_DayN' models - matches the CYCLES
    # naming in TdAI_v3.1_Training_Dataset_Compilation.py.
    cycle_prefix = '03' if target_run_hour == '00' else '15'
    fhr_labels = {day1_fhr: (f"{cycle_prefix}z_Day1", "Day 1"), day2_fhr: (f"{cycle_prefix}z_Day2", "Day 2")}
    models_by_fhr = {}

    print(f"🧠 Model selection: {target_run_hour}Z cycle → Day 1 uses F{day1_fhr:02d}, Day 2 uses F{day2_fhr:02d}")

    for fhr, (cycle_name, label) in fhr_labels.items():
        model_path = os.path.join(model_dir, f"tdai_probabilistic_model_{station}_{cycle_name}.joblib")
        features_path = os.path.join(model_dir, f"probabilistic_model_feature_schema_{station}_{cycle_name}.joblib")

        if not (os.path.exists(model_path) and os.path.exists(features_path)):
            print(f"⏭️ {label} ({cycle_name}): probabilistic ensemble or schema missing at {model_dir}. Skipping this forecast hour for K{station}.")
            continue

        print(f"   └── {label} ({cycle_name}): loading ensemble '{os.path.basename(model_path)}' + schema '{os.path.basename(features_path)}'")

        models_by_fhr[fhr] = {
            'label': label,
            'ensemble': joblib.load(model_path),
            'feature_order': joblib.load(features_path),
            'model_filename': os.path.basename(model_path),
        }

    if not models_by_fhr:
        print(f"❌ No usable probabilistic ensembles found for K{station} under {model_dir}. Skipping station entirely.")
        return

    for q in QUANTILES:
        master_input_df[f'TdAI_Predicted_Bias_{q}'] = 0.0
        master_input_df[f'TdAI_Corrected_Dewpoint_{q}'] = master_input_df['NBM Dewpoint (F)'].astype(float).round(1)

    master_input_df['TdAI Status'] = "Active"

    t_pass = master_input_df['NBM Temperature (F)'] >= 50.0
    rh_pass = master_input_df['NBM RH (%)'] <= 60.0
    sky_pass = master_input_df['NBM Cloud Cover (%)'] <= 60.0
    threshold_mask = t_pass & rh_pass & sky_pass

    for idx, row in master_input_df.iterrows():
        v_str = pd.to_datetime(row['valid_time']).strftime('%Y-%m-%d %H:%M')
        if threshold_mask[idx]:
            print(f"🔥 {v_str} matches boundary requirements. Initializing Quantile Matrix Engine...")
        else:
            reasons = []
            if not t_pass[idx]: reasons.append("T < 50 F")
            if not rh_pass[idx]: reasons.append("RH > 60 %")
            if not sky_pass[idx]: reasons.append("Sky > 60 %")

            status_text = f"{', '.join(reasons)}"
            master_input_df.at[idx, 'TdAI Status'] = status_text
            print(f"🛑 {v_str} bypassed. Criteria flag down: {status_text}")

    for fhr, cfg in models_by_fhr.items():
        fhr_mask = threshold_mask & (master_input_df['forecast_hour'] == fhr)
        passing_rows = master_input_df[fhr_mask].copy()

        if passing_rows.empty:
            print(f"⏭️ {cfg['label']} ({cfg['model_filename']}): no rows passed threshold gating, ensemble not invoked.")
            continue

        run_vtimes = ', '.join(pd.to_datetime(passing_rows['valid_time']).dt.strftime('%Y-%m-%d %H:%M UTC'))
        print(f"🚀 {cfg['label']}: running '{cfg['model_filename']}' on {len(passing_rows)} row(s) → {run_vtimes}")

        X_live = passing_rows.set_index('valid_time') if 'valid_time' in passing_rows.columns else passing_rows.copy()
        X_live = X_live[cfg['feature_order']]

        # 1. Run inference across all underlying quantile estimators first to generate the raw bias arrays
        raw_biases = {}
        for q in QUANTILES:
            raw_biases[q] = cfg['ensemble'][q].predict(X_live)
            master_input_df.loc[fhr_mask, f'TdAI_Predicted_Bias_{q}'] = np.round(raw_biases[q], 1)

        # 2. Cross-mapped alignment: a larger predicted BIAS quantile means MORE
        # drying subtracted, which means a LOWER corrected dewpoint - so the
        # bias q90 (largest error) maps to the dewpoint q10 (driest outcome),
        # and vice versa. This is intentional, not an inversion bug.
        master_input_df.loc[fhr_mask, 'TdAI_Corrected_Dewpoint_q10'] = np.round(
            master_input_df.loc[fhr_mask, 'NBM Dewpoint (F)'] - raw_biases['q90'], 1
        )
        master_input_df.loc[fhr_mask, 'TdAI_Corrected_Dewpoint_q25'] = np.round(
            master_input_df.loc[fhr_mask, 'NBM Dewpoint (F)'] - raw_biases['q75'], 1
        )
        master_input_df.loc[fhr_mask, 'TdAI_Corrected_Dewpoint_q50'] = np.round(
            master_input_df.loc[fhr_mask, 'NBM Dewpoint (F)'] - raw_biases['q50'], 1
        )
        master_input_df.loc[fhr_mask, 'TdAI_Corrected_Dewpoint_q75'] = np.round(
            master_input_df.loc[fhr_mask, 'NBM Dewpoint (F)'] - raw_biases['q25'], 1
        )
        master_input_df.loc[fhr_mask, 'TdAI_Corrected_Dewpoint_q90'] = np.round(
            master_input_df.loc[fhr_mask, 'NBM Dewpoint (F)'] - raw_biases['q10'], 1
        )

    # -------------------------------------------------------------------------
    # 📊 SECTION 5: HISTORICAL SYSTEM LATENCY SYNC & LEDGER GENERATOR
    # -------------------------------------------------------------------------
    print(f"\n📡 Writing ensemble telemetry to logging arrays for K{station}...")

    headers = OUTPUT_HEADERS

    if os.path.exists(output_csv_path):
        combined_log_df = pd.read_csv(output_csv_path)
        for col in headers:
            if col not in combined_log_df.columns:
                combined_log_df[col] = np.nan
    else:
        print("📝 Generating a fresh probabilistic verification array...")
        combined_log_df = pd.DataFrame(columns=headers)

    current_time_utc = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    new_rows_list = []

    for idx in range(len(master_input_df)):
        row_data = master_input_df.iloc[idx]
        forecast_valid_time = row_data['valid_time']

        log_row = {
            'valid_time': forecast_valid_time.strftime('%Y-%m-%d %H:%M:%S'),
            'TdAI Run Time (UTC)': current_time_utc.strftime('%Y-%m-%d %H:%M UTC'),
            'TdAI Status': row_data['TdAI Status'],
            'NBM Dewpoint (F)': row_data['NBM Dewpoint (F)'] if threshold_mask[idx] else np.nan,
            'ASOS Ground Truth Dewpoint (F)': np.nan,
            'Raw NBM Error (F)': np.nan,
            'Post TdAI Median Error (F)': np.nan,
            'TdAI Median Skill Score (%)': np.nan
        }

        for q in QUANTILES:
            log_row[f'TdAI_Predicted_Bias_{q}'] = row_data[f'TdAI_Predicted_Bias_{q}']
            log_row[f'TdAI_Corrected_Dewpoint_{q}'] = row_data[f'TdAI_Corrected_Dewpoint_{q}'] if threshold_mask[idx] else np.nan

        new_rows_list.append(log_row)

    if not new_rows_list:
        return

    new_entry_df = pd.DataFrame(new_rows_list)
    new_entry_df['valid_time'] = pd.to_datetime(new_entry_df['valid_time']).dt.strftime('%Y-%m-%d %H:%M:%S')

    if not combined_log_df.empty:
        combined_log_df['valid_time'] = pd.to_datetime(combined_log_df['valid_time'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')

        for target_vtime in new_entry_df['valid_time']:
            existing_match = combined_log_df[combined_log_df['valid_time'] == target_vtime]
            if not existing_match.empty:
                old_asos = existing_match['ASOS Ground Truth Dewpoint (F)'].iloc[0]
                if pd.notna(old_asos):
                    print(f"♻️ Retaining historical verification observations for {target_vtime}")
                    new_entry_df.loc[new_entry_df['valid_time'] == target_vtime, 'ASOS Ground Truth Dewpoint (F)'] = old_asos

                    r_nbm_err = new_entry_df.loc[new_entry_df['valid_time'] == target_vtime, 'NBM Dewpoint (F)'].values[0] - old_asos
                    p_tdai_err = new_entry_df.loc[new_entry_df['valid_time'] == target_vtime, 'TdAI_Corrected_Dewpoint_q50'].values[0] - old_asos
                    skill_score = (1.0 - (abs(p_tdai_err) / abs(r_nbm_err))) * 100 if abs(r_nbm_err) > 0 else 0.0

                    new_entry_df.loc[new_entry_df['valid_time'] == target_vtime, 'Raw NBM Error (F)'] = round(r_nbm_err, 2)
                    new_entry_df.loc[new_entry_df['valid_time'] == target_vtime, 'Post TdAI Median Error (F)'] = round(p_tdai_err, 2)
                    new_entry_df.loc[new_entry_df['valid_time'] == target_vtime, 'TdAI Median Skill Score (%)'] = round(skill_score, 1)

    target_valid_times = new_entry_df['valid_time'].tolist()
    combined_log_df = combined_log_df[~combined_log_df['valid_time'].isin(target_valid_times)]
    combined_log_df = pd.concat([combined_log_df, new_entry_df], ignore_index=True)

    combined_log_df['valid_time'] = pd.to_datetime(combined_log_df['valid_time']).dt.strftime('%Y-%m-%d %H:%M:%S')
    combined_log_df = combined_log_df.sort_values(by=['valid_time', 'ASOS Ground Truth Dewpoint (F)'], na_position='first')
    combined_log_df = combined_log_df.drop_duplicates(subset=['valid_time'], keep='last')
    combined_log_df = combined_log_df.sort_values(by='valid_time').reset_index(drop=True)
    combined_log_df_dt = pd.to_datetime(combined_log_df['valid_time'])

    # -------------------------------------------------------------------------
    # 🔄 RETROSPECTIVE VERIFICATION SUB-ENGINE (BULK DESERIALIZATION LOGIC)
    # -------------------------------------------------------------------------
    missing_mask = combined_log_df['ASOS Ground Truth Dewpoint (F)'].isna() & (combined_log_df_dt + datetime.timedelta(minutes=15) <= current_time_utc)
    missing_indices = combined_log_df[missing_mask].index

    if len(missing_indices) > 0:
        print(f"\n🔄 Running validation parsing across {len(missing_indices)} historical timestamps for K{station}...")
        missing_vtimes = pd.to_datetime(combined_log_df.loc[missing_indices, 'valid_time'])
        start_date = missing_vtimes.min() - datetime.timedelta(days=1)
        end_date = missing_vtimes.max() + datetime.timedelta(days=1)

        asos_url = (
            f"https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?"
            f"station={station}&data=dwpf"
            f"&year1={start_date.year}&month1={start_date.month}&day1={start_date.day}"
            f"&year2={end_date.year}&month2={end_date.month}&day2={end_date.day}"
            f"&tz=UTC&format=comma"
        )

        bulk_asos_df = pd.DataFrame()
        try:
            res = requests.get(asos_url, timeout=25)
            if res.status_code == 200:
                bulk_asos_df = pd.read_csv(io.StringIO(res.text), comment='#')
                if not bulk_asos_df.empty and 'dwpf' in bulk_asos_df.columns:
                    bulk_asos_df['valid_dt'] = pd.to_datetime(bulk_asos_df['valid'])
                    bulk_asos_df['rounded_dt'] = bulk_asos_df['valid_dt'].dt.round('h')
                    bulk_asos_df['rounded_valid_time_str'] = bulk_asos_df['rounded_dt'].dt.strftime('%Y-%m-%d %H:%M:%S')
        except Exception as e:
            print(f"   ❌ Network fault during bulk verification sync: {e}")

        if not bulk_asos_df.empty and 'rounded_valid_time_str' in bulk_asos_df.columns:
            for idx in missing_indices:
                v_time = pd.to_datetime(combined_log_df.loc[idx, 'valid_time'])
                target_vtime_str = v_time.strftime('%Y-%m-%d %H:%M:%S')
                v_status = str(combined_log_df.loc[idx, 'TdAI Status']).strip()

                target_obs = bulk_asos_df[bulk_asos_df['rounded_valid_time_str'] == target_vtime_str].copy()
                if not target_obs.empty:
                    target_obs['dwpf_numeric'] = pd.to_numeric(target_obs['dwpf'], errors='coerce')
                    valid_reports = target_obs.dropna(subset=['dwpf_numeric'])

                    if not valid_reports.empty:
                        # Routine + SPECI reports can both round to the same
                        # clock hour - keep the one closest to the top of the
                        # hour, matching the dedup fix already applied to the
                        # offline data_download/ASOS_download.py pipeline.
                        valid_reports = valid_reports.copy()
                        valid_reports['_minutes_from_hour'] = (valid_reports['valid_dt'] - valid_reports['rounded_dt']).abs()
                        closest_report = valid_reports.sort_values('_minutes_from_hour').iloc[0]
                        asos_gt = float(closest_report['dwpf_numeric'])
                        combined_log_df.loc[idx, 'ASOS Ground Truth Dewpoint (F)'] = asos_gt

                        if v_status == "Active":
                            nbm_dpt = float(combined_log_df.loc[idx, 'NBM Dewpoint (F)'])
                            tdai_dpt = float(combined_log_df.loc[idx, 'TdAI_Corrected_Dewpoint_q50'])

                            r_nbm_err = nbm_dpt - asos_gt
                            p_tdai_err = tdai_dpt - asos_gt
                            skill_score = (1.0 - (abs(p_tdai_err) / abs(r_nbm_err))) * 100 if abs(r_nbm_err) > 0 else 0.0

                            combined_log_df.loc[idx, 'Raw NBM Error (F)'] = round(r_nbm_err, 2)
                            combined_log_df.loc[idx, 'Post TdAI Median Error (F)'] = round(p_tdai_err, 2)
                            combined_log_df.loc[idx, 'TdAI Median Skill Score (%)'] = round(skill_score, 1)

    combined_log_df.to_csv(output_csv_path, index=False)
    print(f"💾 Storage synchronization complete for K{station} → {output_csv_path}")

def main():
    seed_everything(42)
    base_path = "./"

    # Anchor everything to UTC, not the local machine's timezone. GitHub
    # Actions runners default to TZ=UTC so this bug never showed up on the
    # cron, but a machine running this manually from a non-UTC timezone
    # (e.g. US Eastern) can be a full calendar day behind UTC for several
    # hours around each UTC midnight, which used to make date.today() return
    # the wrong day for the HRRR/NBM cycle being targeted.
    current_time_utc = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    today = current_time_utc.date()

    if is_winter_pause(today):
        print(f"❄️ {WINTER_PAUSE_STATUS} (outside the March 1 - November 15 training window). "
              f"Skipping HRRR/NBM downloads and writing status rows only.")
        for station in STATIONS:
            try:
                write_winter_pause_status(station, base_path)
            except Exception as e:
                print(f"❌ K{station} winter-pause status write failed: {e}")
        return

    # -------------------------------------------------------------------------
    # 🛰️ SECTION 1: CHOOSE AND DOWNLOAD THE MOST RECENT 12Z OR 00Z HRRR RUN
    #    (one shared download - every station's point is extracted from the
    #    same GRIB files in process_station, no need to redownload per station)
    # -------------------------------------------------------------------------
    yesterday = today - datetime.timedelta(days=1)

    if current_time_utc.hour < 14:
        target_run_hour = '00'
        forecast_hours = [21, 45]  # Day 1 Afternoon (F21) & Day 2 Afternoon (F45)
        date_strs = [today.strftime('%Y%m%d'), yesterday.strftime('%Y%m%d')]
        print(f"🌙 Overnight Cron: Extracting 00Z HRRR Cycles for Day 1 (F21) and Day 2 (F45) afternoon windows...")
    else:
        target_run_hour = '12'
        forecast_hours = [9, 33]   # Day 1 Afternoon (F09) & Day 2 Afternoon (F33)
        date_strs = [today.strftime('%Y%m%d'), yesterday.strftime('%Y%m%d')]
        print(f"☀️ Daytime Cron: Extracting 12Z HRRR Cycles for Day 1 (F09) and Day 2 (F33) afternoon windows...")

    grib_files = {}
    for ds_date in date_strs:
        success = True
        temp_files = {}
        for fhr in forecast_hours:
            local_file = download_hrrr_grib(ds_date, run_hour=target_run_hour, forecast_hour=fhr)
            if local_file:
                temp_files[fhr] = local_file
            else:
                success = False
                for f in temp_files.values():
                    if os.path.exists(f): os.remove(f)
                break
        if success:
            grib_files = temp_files
            break

    if not grib_files:
        raise RuntimeError(f"Could not retrieve complete synchronous {target_run_hour}z HRRR frames from server registry.")

    # -------------------------------------------------------------------------
    # 📊 SECTION 2: DYNAMIC DOWNLOAD OF NBM (01Z or 13Z) - ONE SHARED BULLETIN
    #    (contains every station's block; each is parsed out in process_station)
    # -------------------------------------------------------------------------
    if current_time_utc.hour < 14:
        nbm_run_hour = '01'
        print("🌙 Overnight Cron: Targeting the 01Z NBM Text Bulletin...")
    else:
        nbm_run_hour = '13'
        print("☀️ Daytime Cron: Targeting the 13Z NBM Text Bulletin...")

    bulletin_text = None
    successful_date_str = None
    for date_str in date_strs:
        bulletin_text = get_nbm_bulletin(date_str, run_hour=nbm_run_hour)
        if bulletin_text:
            successful_date_str = date_str
            break

    if not bulletin_text:
        raise RuntimeError(f"NBM operational terminal bulletin stream ({nbm_run_hour}Z) unreachable.")

    # -------------------------------------------------------------------------
    # 🔁 SECTION 3: RUN THE FULL PIPELINE FOR EACH STATION
    # -------------------------------------------------------------------------
    try:
        for station, (lat, lon) in STATIONS.items():
            try:
                process_station(
                    station=station, lat=lat, lon=lon,
                    target_run_hour=target_run_hour, forecast_hours=forecast_hours,
                    bulletin_text=bulletin_text, successful_date_str=successful_date_str,
                    grib_files=grib_files, base_path=base_path,
                )
            except Exception as e:
                print(f"❌ K{station} pipeline failed: {e}")
                continue
    finally:
        # Clean up the shared GRIB files/index sidecars now that every
        # station has finished extracting its point from them.
        for grib_file in grib_files.values():
            if os.path.exists(grib_file):
                os.remove(grib_file)
            for idx_file in glob.glob(f"{grib_file}*.idx"):
                os.remove(idx_file)

if __name__ == "__main__":
    main()
