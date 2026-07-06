#!/usr/bin/env python3
"""
TdAI Probabilistic Ingestion, Prediction, and Verification Pipeline
Automated Convective Boundary Layer Post-Processing Ensembles Module
"""

import os
import io
import re
import time
import random
import datetime
import requests
import numpy as np
import pandas as pd
import xarray as xr
import joblib

def seed_everything(seed=42):
    """Locks random states to enforce analytical reproducibility across iterations."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"✅ Random state locked at seed: {seed}")

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
    """Retrieves the NBM text blend terminal output for a specified run hour."""
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

def main():
    seed_everything(42)
    # Target KCAR geographical coordinates
    kcar_lat, kcar_lon = 46.870478, -68.017225
    
    # Cloud repository localized mapping
    base_path = "./"
    output_csv_path = os.path.join(base_path, "operational_probabilistic_log.csv")

    # -------------------------------------------------------------------------
    # 🛰️ SECTION 1: CHOOSE AND DOWNLOAD THE MOST RECENT 12Z OR 00Z HRRR RUN
    # -------------------------------------------------------------------------
    current_time_utc = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    
    today = datetime.date.today()
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
    all_forecast_dfs = []

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

    for fhr, grib_file in grib_files.items():
        print(f"📊 Parsing vertical profile structures from temporary raster F{fhr:02d}...")
        with xr.open_dataset(grib_file, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa'}}) as ds_filtered:
            valid_time_str = pd.Timestamp(ds_filtered['valid_time'].values).strftime('%Y-%m-%d %H:%M:%S')
            
            kcar_lon_360 = kcar_lon + 360.0 if kcar_lon < 0 else kcar_lon
            squared_distance = ((ds_filtered['latitude'] - kcar_lat)**2) + ((ds_filtered['longitude'] - kcar_lon_360)**2)
            y_idx, x_idx = np.unravel_index(np.nanargmin(squared_distance.to_numpy()), squared_distance.shape)
            
            ds_point = ds_filtered.isel(y=y_idx, x=x_idx)
            p_lvls = ds_point['isobaricInhPa'].to_numpy()
            
            mask = (p_lvls >= 500) & (p_lvls <= 1000)
            
            all_forecast_dfs.append(pd.DataFrame({
                'valid_time': np.full(np.sum(mask), valid_time_str),
                'HRRR Pressure (hPa)': p_lvls[mask],
                'HRRR Temperature (K)': ds_point['t'].to_numpy()[mask],
                'HRRR Dewpoint (K)': ds_point['dpt'].to_numpy()[mask]
            }))
        if os.path.exists(grib_file): os.remove(grib_file)

    master_hrrr_profiles_df = pd.concat(all_forecast_dfs, ignore_index=True)
    t_c = master_hrrr_profiles_df['HRRR Temperature (K)'].astype(float) - 273.15
    dp_c = master_hrrr_profiles_df['HRRR Dewpoint (K)'].astype(float) - 273.15
    es = np.exp((17.625 * t_c) / (243.04 + t_c))
    e = np.exp((17.625 * dp_c) / (243.04 + dp_c))
    master_hrrr_profiles_df['HRRR RH (%)'] = round(np.clip(100 * (e / es), 0.0, 100.0), 1)

    # -------------------------------------------------------------------------
    # 📊 SECTION 2: DYNAMIC DOWNLOAD OF NBM (01Z or 13Z)
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

    lines = bulletin_text.split('\n')
    kcar_lines = []
    in_block = False
    for line in lines:
        if "KCAR" in line and "NBM" in line: in_block = True
        if in_block:
            kcar_lines.append(line)
            if len(kcar_lines) > 2 and (line.strip() == "" or line.startswith('#') or "STATION" in line):
                if "STATION" in line or line.startswith('#'): kcar_lines.pop()
                break

    parsed_data = {}
    targets = {'UTC': 'UTC Hour', 'TMP': 'NBM Temperature (F)', 'DPT': 'NBM Dewpoint (F)', 'SKY': 'NBM Cloud Cover (%)', 'WDR': 'NBM Wind Direction (tens deg)', 'WSP': 'NBM Wind Speed (kts)', 'MHT': 'NBM Mixing Height (100s ft)'}
    for line in kcar_lines:
        tokens = line.split()
        if not tokens or tokens[0] not in targets: continue
        parsed_data[targets[tokens[0]]] = [int(v) if v.isdigit() else float(v) if '.' in v else None for v in tokens[1:]]

    nbm_df = pd.DataFrame(parsed_data)
    init_date = datetime.datetime.strptime(successful_date_str, '%Y%m%d')
    valid_times = []
    curr_dt = init_date
    prev_hr = -1
    for hr in nbm_df['UTC Hour']:
        if hr is None:
            valid_times.append(pd.NaT)
            continue
        if prev_hr != -1 and hr < prev_hr: curr_dt += datetime.timedelta(days=1)
        valid_times.append(curr_dt.replace(hour=hr, minute=0, second=0, microsecond=0))
        prev_hr = hr

    nbm_df['valid_time'] = pd.to_datetime(valid_times)
    if 'NBM Wind Direction (tens deg)' in nbm_df.columns: nbm_df['NBM Wind Direction (deg)'] = nbm_df['NBM Wind Direction (tens deg)'] * 10
    if 'NBM Mixing Height (100s ft)' in nbm_df.columns: nbm_df['NBM Mixing Height (100s ft AGL)'] = nbm_df['NBM Mixing Height (100s ft)']
    
    nbm_tc = (nbm_df['NBM Temperature (F)'] - 32) * (5.0 / 9.0)
    nbm_tdc = (nbm_df['NBM Dewpoint (F)'] - 32) * (5.0 / 9.0)
    nbm_es = np.exp((17.625 * nbm_tc) / (243.04 + nbm_tc))
    nbm_e = np.exp((17.625 * nbm_tdc) / (243.04 + nbm_tdc))
    nbm_df['NBM RH (%)'] = round(np.clip(100 * (nbm_e / nbm_es), 0.0, 100.0), 1)
    
    print("⏰ Filtering output matrix arrays to parse 21Z peak afternoon mixing windows...")
    nbm_df = nbm_df[nbm_df['valid_time'].dt.hour == 21].copy()

    # -------------------------------------------------------------------------
    # 🔗 SECTION 3: MATRIX ALIGNMENT & METEOROLOGICAL EXPANSIONS
    # -------------------------------------------------------------------------
    nbm_df['valid_time'] = pd.to_datetime(nbm_df['valid_time'])
    master_hrrr_profiles_df['valid_time'] = pd.to_datetime(master_hrrr_profiles_df['valid_time'])
    
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
    doy = master_input_df['valid_time'].dt.dayofyear
    master_input_df['sin_season'] = np.sin(2 * np.pi * doy / 365.25)
    master_input_df['cos_season'] = np.cos(2 * np.pi * doy / 365.25)

    # -------------------------------------------------------------------------
    # 🔮 SECTION 4: PROBABILISTIC QUANTILE MULTI-PREDICTION ENGINE
    # -------------------------------------------------------------------------
    model_import_path = os.path.join(base_path, "tdai_probabilistic_models.joblib")
    features_import_path = os.path.join(base_path, "probabilistic_model_feature_schema.joblib")
    
    if not (os.path.exists(model_import_path) and os.path.exists(features_import_path)):
        raise FileNotFoundError("❌ Probabilistic ensemble assets or schema configuration missing from repository root.")
        
    prob_ensemble = joblib.load(model_import_path)
    trained_feature_order = joblib.load(features_import_path)

    # Initialize data structures for all quantiles
    quantiles = ['q10', 'q25', 'q50', 'q75', 'q90']
    for q in quantiles:
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

    passing_rows = master_input_df[threshold_mask].copy()

    if not passing_rows.empty:
        X_live = passing_rows.set_index('valid_time') if 'valid_time' in passing_rows.columns else passing_rows.copy()
        # Enforce exact structural feature alignment sequence
        X_live = X_live[trained_feature_order]

        # Loop through all ensemble models inside our dictionary structure
        for q in quantiles:
            bias_predictions = prob_ensemble[q].predict(X_live)
            master_input_df.loc[threshold_mask, f'TdAI_Predicted_Bias_{q}'] = np.round(bias_predictions, 1)
            master_input_df.loc[threshold_mask, f'TdAI_Corrected_Dewpoint_{q}'] = np.round(
                master_input_df.loc[threshold_mask, 'NBM Dewpoint (F)'] - master_input_df.loc[threshold_mask, f'TdAI_Predicted_Bias_{q}'], 1
            )

    # -------------------------------------------------------------------------
    # 📊 SECTION 5: HISTORICAL SYSTEM LATENCY SYNC & LEDGER GENERATOR
    # -------------------------------------------------------------------------
    print("\n📡 Writing ensemble telemetry to logging arrays...")
    
    headers = [
        'valid_time', 'TdAI Run Time (UTC)', 'TdAI Status', 'NBM Temperature (F)', 'NBM RH (%)', 'NBM Dewpoint (F)',
        'TdAI_Predicted_Bias_q10', 'TdAI_Corrected_Dewpoint_q10',
        'TdAI_Predicted_Bias_q25', 'TdAI_Corrected_Dewpoint_q25',
        'TdAI_Predicted_Bias_q50', 'TdAI_Corrected_Dewpoint_q50',
        'TdAI_Predicted_Bias_q75', 'TdAI_Corrected_Dewpoint_q75',
        'TdAI_Predicted_Bias_q90', 'TdAI_Corrected_Dewpoint_q90',
        'ASOS Ground Truth Dewpoint (F)', 'Raw NBM Error (F)', 'Post TdAI Median Error (F)', 'TdAI Median Skill Score (%)'
    ]
    
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
            'NBM Temperature (F)': row_data['NBM Temperature (F)'] if threshold_mask[idx] else np.nan,
            'NBM RH (%)': row_data['NBM RH (%)'] if threshold_mask[idx] else np.nan,
            'NBM Dewpoint (F)': row_data['NBM Dewpoint (F)'] if threshold_mask[idx] else np.nan,
            'ASOS Ground Truth Dewpoint (F)': np.nan, 
            'Raw NBM Error (F)': np.nan, 
            'Post TdAI Median Error (F)': np.nan, 
            'TdAI Median Skill Score (%)': np.nan
        }
        
        # Log data spectrum across all quantiles 
        for q in quantiles:
            log_row[f'TdAI_Predicted_Bias_{q}'] = row_data[f'TdAI_Predicted_Bias_{q}']
            log_row[f'TdAI_Corrected_Dewpoint_{q}'] = row_data[f'TdAI_Corrected_Dewpoint_{q}'] if threshold_mask[idx] else np.nan
            
        new_rows_list.append(log_row)

    if new_rows_list:
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
                        
                        # Score verification strictly based on the 50th percentile (Median) 
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
            print(f"\n🔄 Running validation parsing across {len(missing_indices)} historical timestamps...")
            missing_vtimes = pd.to_datetime(combined_log_df.loc[missing_indices, 'valid_time'])
            start_date = missing_vtimes.min() - datetime.timedelta(days=1)
            end_date = missing_vtimes.max() + datetime.timedelta(days=1)
            
            asos_url = (
                f"https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?"
                f"station=CAR&data=dwpf"
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
                        bulk_asos_df['rounded_valid_time_str'] = bulk_asos_df['valid_dt'].dt.round('h').dt.strftime('%Y-%m-%d %H:%M:%S')
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
                            asos_gt = float(valid_reports['dwpf_numeric'].iloc[0])
                            combined_log_df.loc[idx, 'ASOS Ground Truth Dewpoint (F)'] = asos_gt
                            
                            if v_status == "Active":
                                nbm_dew = float(combined_log_df.loc[idx, 'NBM Dewpoint (F)'])
                                tdai_dew = float(combined_log_df.loc[idx, 'TdAI_Corrected_Dewpoint_q50'])
                                
                                r_nbm_err = nbm_dew - asos_gt
                                p_tdai_err = tdai_dew - asos_gt
                                skill_score = (1.0 - (abs(p_tdai_err) / abs(r_nbm_err))) * 100 if abs(r_nbm_err) > 0 else 0.0
                                
                                combined_log_df.loc[idx, 'Raw NBM Error (F)'] = round(r_nbm_err, 2)
                                combined_log_df.loc[idx, 'Post TdAI Median Error (F)'] = round(p_tdai_err, 2)
                                combined_log_df.loc[idx, 'TdAI Median Skill Score (%)'] = round(skill_score, 1)

        combined_log_df.to_csv(output_csv_path, index=False)
        print("💾 Storage synchronization complete.")

if __name__ == "__main__":
    main()
