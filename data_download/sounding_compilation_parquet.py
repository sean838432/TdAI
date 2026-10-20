# =========================================================================
# 🚀 COMPILE CSV FILES INTO PER-RUN PARQUET FILES
# =========================================================================

import pandas as pd
from pathlib import Path

STATION_IDS = ["KBTV", "KSLK", "KRUT", "KVSF", "KPBG", "KMPV"]  # Add all the stations you want to compile
soundings_dir = Path(r"/home/sean834/TdAI/data_download/HRRR_forecast_soundings")

for STATION_ID in STATION_IDS:
    station_dir = soundings_dir / STATION_ID
    print(f"\n────────────────── Processing Station: {STATION_ID} ──────────────────")

    if not station_dir.is_dir():
        print(f"❌ Station folder not found: {station_dir}. Skipping.")
        continue

    # Each subfolder of station_dir is a run/forecast-hour combo, e.g. 12z_f09, 12z_f36
    run_dirs = sorted(d for d in station_dir.iterdir() if d.is_dir())

    if not run_dirs:
        print(f"❌ No run folders found under {station_dir}.")
        continue

    for run_dir in run_dirs:
        run_label = run_dir.name
        print(f"\n📦 Starting Parquet compilation for {STATION_ID}/{run_label}...")

        # Get a list of all successfully downloaded CSV files for this run
        csv_files = list(run_dir.glob("*.csv"))

        if not csv_files:
            print("❌ No CSV files found to compile.")
            continue

        print(f"🔍 Found {len(csv_files)} CSV files. Merging...")

        compiled_dfs = []
        for index, file_path in enumerate(csv_files):
            try:
                # Read the CSV profile
                temp_df = pd.read_csv(file_path)

                # Explicitly parse the datetimes so parquet handles them natively
                temp_df['valid_time'] = pd.to_datetime(temp_df['valid_time'])
                temp_df['init_time'] = pd.to_datetime(temp_df['init_time'])

                compiled_dfs.append(temp_df)

                # Print compilation progress periodically
                if (index + 1) % 500 == 0:
                    print(f"   Processed {index + 1}/{len(csv_files)} files...")

            except Exception as e:
                print(f"⚠️ Error reading {file_path.name} during merge: {e}")
                continue

        if not compiled_dfs:
            print("❌ Compilation failed: No valid data frames could be aggregated.")
            continue

        # Combine all profiles for this run into one dataframe
        print("🔗 Concatenating dataframes...")
        run_df = pd.concat(compiled_dfs, ignore_index=True)

        # Sort chronologically by valid_time and pressure height for machine learning sanity
        # (Assuming 'isobaricInhPa' exists in the columns)
        sort_cols = [c for c in ['valid_time', 'isobaricInhPa'] if c in run_df.columns]
        if sort_cols:
            run_df = run_df.sort_values(by=sort_cols).reset_index(drop=True)

        # Define output destination: one parquet file per run, inside the station folder
        parquet_filename = f"{STATION_ID}_{run_label}_Soundings.parquet"
        parquet_path = station_dir / parquet_filename

        print(f"💾 Writing to Parquet: {parquet_path}")
        # Use snappy compression (standard, fast, highly compressed for numbers)
        run_df.to_parquet(parquet_path, engine='pyarrow', compression='snappy', index=False)

        print(f"✨ SUCCESS! Compiled dataset contains {len(run_df)} total level-rows.")
        print(f"📊 Dimensions: {run_df.shape}")
