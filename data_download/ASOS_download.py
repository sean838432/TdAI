"""
This script grabs dewpoint data from specified ASOS stations and saves the data to CSV files

Info on CGI parameters is available at:

    https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?help
"""

import requests
import os
import time
import io
import pandas as pd

################################## INPUTS ####################################
# List of station identifiers (IEM searches all networks automatically)
STATIONS = ['CAR', 'BGR', 'BHB', 'HUL', 'GNR', 'MLT', 'FVE']
DATA_VARS = ['dwpf', 'valid']
BASE_URL = 'https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py'

# Date range for the bulk request
START_YEAR = 2020
END_YEAR = 2026

# Seasonal Filtering (March 1 to Nov 15)
START_MD = (3, 1)
END_MD = (11, 15)

SAVE_DIR = "/home/sean834/TdAI/data_download/ASOS_data/"
##############################################################################

for station in STATIONS:
    # ⚡ 'network' parameter omitted — IEM defaults to searching all networks!
    params = {
        'station': station,
        'data': DATA_VARS,
        'year1': START_YEAR, 'month1': 1, 'day1': 1,
        'year2': END_YEAR, 'month2': 12, 'day2': 31,
        'format': 'onlycomma',
        'tz': 'UTC',
        'missing': 'M',
        'report_type': 3, # METAR reports only
    }

    try:
        print(f"🚀 Fetching bulk data for {station} ({START_YEAR}-{END_YEAR})...")
        response = requests.get(BASE_URL, params=params, timeout=60)
        response.raise_for_status()

        # Load into Pandas for fast seasonal filtering
        df = pd.read_csv(
            io.StringIO(response.text),
            comment='#',
            skiprows=1,
            names=['Station', 'valid_time', 'ASOS Dewpoint (F)'],
            na_values='M'
        )

        initial_count = len(df)
        df = df.dropna(subset=['ASOS Dewpoint (F)'])
        dropped_count = initial_count - len(df)

        if dropped_count > 0:
            print(f"🧹 Cleaned up {dropped_count} missing ('M') observations.")

        if df.empty:
            print(f"⚠️ No data remaining for {station} after cleaning.")
            continue

        # Convert to datetime with explicit format
        df['valid_time'] = pd.to_datetime(df['valid_time'], format='%Y-%m-%d %H:%M')

        # --- THE SEASONAL FILTER ---
        df['month_day'] = df['valid_time'].apply(lambda x: (x.month, x.day))
        seasonal_mask = (df['month_day'] >= START_MD) & (df['month_day'] <= END_MD)
        df_filtered = df[seasonal_mask].drop(columns=['month_day'])

        # --- DEDUPLICATE TO ONE OBSERVATION PER HOUR ---
        # METAR issues routine reports plus occasional SPECI (special)
        # reports, so more than one raw observation can round to the same
        # clock hour downstream. Keep only the observation closest to the
        # top of each hour - training data later merges on the rounded
        # hour, and duplicate hours there produce two rows with identical
        # NBM/HRRR predictors but conflicting ASOS-derived targets.
        hour_bucket = df_filtered['valid_time'].dt.round('h')
        minutes_from_hour = (df_filtered['valid_time'] - hour_bucket).abs()
        pre_dedup_count = len(df_filtered)
        df_filtered = (
            df_filtered.assign(_hour_bucket=hour_bucket, _minutes_from_hour=minutes_from_hour)
            .sort_values('_minutes_from_hour')
            .drop_duplicates(subset='_hour_bucket', keep='first')
            .drop(columns=['_hour_bucket', '_minutes_from_hour'])
            .sort_values('valid_time')
        )
        deduped_count = pre_dedup_count - len(df_filtered)
        if deduped_count > 0:
            print(f"🧹 Removed {deduped_count} duplicate-hour observations (kept the one closest to the top of each hour).")

        # Save individual file
        filename = f"K{station}_{START_YEAR}_to_{END_YEAR}_asos.csv"
        save_path = os.path.join(SAVE_DIR, filename)

        df_filtered.to_csv(save_path, index=False)
        print(f"✅ Saved {len(df_filtered)} obs for K{station} to {filename}")

        # Wait to be polite to the server before moving to next station
        print("⏳ Waiting 10 seconds before next station...")
        time.sleep(10)

    except Exception as e:
        print(f"❌ Error for {station}: {e}")
        time.sleep(30) # Longer wait on error

print("\n✨ All station downloads complete!")