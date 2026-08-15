import pandas as pd
from herbie import Herbie
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import warnings
from pathlib import Path
import xarray as xr
import gc

warnings.filterwarnings("ignore", message="This pattern is interpreted as a regular expression")

# --- HRRR RUN CONFIG ---
# Loops over all 4 operational cycles used by TdAI (matches the CYCLES config
# in model_training/TdAI_v3.1_Deterministic_TRAINING.py): init hour, forecast
# lead hour, and the run label used in filenames/logs.
CYCLES = [
    (12, 9, '12z_f09'),
    (12, 33, '12z_f33'),
    (0, 21, '00z_f21'),
    (0, 45, '00z_f45'),
]

# --- STATIONS ---
# STATION_ID = "KCAR"
# STATION_LAT, STATION_LON = 46.870490, -68.017221
# STATION_ID = "KBGR"
# STATION_LAT, STATION_LON = 44.8074, -68.8281
# STATION_ID = "KBHB"
# STATION_LAT, STATION_LON = 44.452318, -68.370777
# STATION_ID = "KGNR"
# STATION_LAT, STATION_LON = 45.462979, -69.554546
# STATION_ID = "KMLT"
# STATION_LAT, STATION_LON = 45.647771, -68.692475
# STATION_ID = "KHUL"
# STATION_LAT, STATION_LON = 46.118457, -67.792894
STATION_ID = "KFVE"
STATION_LAT, STATION_LON = 47.285172, -68.307131


# Station coords df
station_coords = pd.DataFrame({"longitude": [STATION_LON], "latitude": [STATION_LAT]})

# Base output dir - per-cycle subfolder is created inside download_hrrr_hour,
# since the cycle now varies per task instead of being fixed for the whole run.
base_output_dir = Path(r"/home/sean834/TdAI/data_download/HRRR_forecast_soundings") / STATION_ID

def download_hrrr_hour(args):
    """
    args: (run_date, init_hour, fxx, run_label)
    run_date: The calendar day of the HRRR run.
    Downloads the init_hour run and the fxx forecast hour from it.
    """
    run_date, init_hour, fxx, run_label = args
    init_time = run_date.replace(hour=init_hour, minute=0, second=0, microsecond=0)
    lead_time = fxx
    valid_time = init_time + pd.Timedelta(hours=lead_time)

    output_dir = base_output_dir / run_label
    output_dir.mkdir(parents=True, exist_ok=True)

    file_timestamp = init_time.strftime("%Y%m%d")
    save_path = output_dir / f"hrrr_{run_label}_{STATION_ID.lower()}_{file_timestamp}.csv"

    if save_path.exists():
        return # Skip

    try:
        # --- INITIALIZE HERBIE WITH FIXED INIT HOUR AND FORECAST HOUR ---
        H = Herbie(init_time, model="hrrr", product="prs", fxx=lead_time, verbose=False, priority=['aws', 'google'])
        
        # Pull the sounding (Variables and logic remain same)
        ds = H.xarray(":(TMP|DPT|UGRD|VGRD):(1013|1000|975|950|925|900|875|850|825|800|775|750|725|700|675|650|625|600|575|550|525|500) mb")
        
        # --- ROBUST DATA EXTRACTION ---
        if isinstance(ds, list):
            df_list = []
            for d in ds:
                p = d.herbie.pick_points(station_coords).load().to_dataframe().reset_index()
                if 'isobaricInhPa' in p.columns:
                    p = p.set_index('isobaricInhPa')
                df_list.append(p)
            df = pd.concat(df_list, axis=1)
            df = df.loc[:, ~df.columns.duplicated()].reset_index()
        else:
            ds_point = ds.herbie.pick_points(station_coords).load()
            df = ds_point.to_dataframe().reset_index()
        
        if 'longitude' in df.columns:
            df['longitude'] = df['longitude'].apply(lambda x: x - 360 if x > 180 else x)
        
        # Store metadata
        df['valid_time'] = valid_time
        df['init_time'] = init_time
        df['lead_time'] = lead_time

        df.to_csv(save_path, index=False)

        print(f"✅ Saved {run_label} Forecast: {file_timestamp}")

    except Exception as e:
        print(f"❌ Error {run_label} {file_timestamp}: {str(e)[:50]}")
        
    finally:
        # --- THE MEMORY CLEANUP ---
        # 1. Close the Xarray Dataset (this releases the file handle)
        if ds is not None:
            if isinstance(ds, list):
                for d in ds:
                    d.close()
            else:
                ds.close()
        
        # 2. Delete large objects from local scope
        if 'df' in locals(): del df
        if 'ds' in locals(): del ds
        if 'ds_point' in locals(): del ds_point
        
        # 3. Force Python to empty the trash
        gc.collect()



if __name__ == "__main__":
    print(f"🚀 Downloading {STATION_ID} across {len(CYCLES)} cycles: {[c[2] for c in CYCLES]}")

    all_tasks = []
    for year in range(2020, 2027):
        days = pd.date_range(start=f"{year}-03-01", end=f"{year}-11-15", freq='D')
        for init_hour, fxx, run_label in CYCLES:
            for d in days:
                all_tasks.append((d, init_hour, fxx, run_label))

    print(f"🚀 Total files to process: {len(all_tasks)}")


    # Process the data in parallel and in chunks (to save memory)
    chunk_size = 100
    for i in range(0, len(all_tasks), chunk_size):
        chunk = all_tasks[i : i + chunk_size]
        print(f"📦 Batch {i//chunk_size + 1}/{(len(all_tasks)-1)//chunk_size + 1}...")

        # Use ProcessPool instead of ThreadPool. ThreadPool crashes EVERYTHING!!!
        # Set max_workers to 4 or 6 (don't max out your CPU)
        with ProcessPoolExecutor(max_workers=8) as executor:
            executor.map(download_hrrr_hour, chunk)

    print("🏁 Done!")




























# if __name__ == "__main__":
#     all_dates = []
#     for year in range(2020, 2026): 
#         days = pd.date_range(start=f"{year}-04-01", end=f"{year}-11-15", freq='D')
#         for day in days:
#             for hour in range(15, 24):
#                 all_dates.append(day.replace(hour=hour))

#     print(f"🚀 Total files to process: {len(all_dates)}")
    
#     # --- STABILITY FIX: PROCESS IN CHUNKS ---
#     # This prevents the computer from trying to "plan" 12,000 tasks at once.
#     chunk_size = 50 # Process 50 files at a time
#     for i in range(0, len(all_dates), chunk_size):
#         chunk = all_dates[i : i + chunk_size]
#         print(f"📦 Processing Batch {i//chunk_size + 1} ({chunk[0].strftime('%Y-%m')})...")
        
#         with ThreadPoolExecutor(max_workers=2) as executor:
#             # map is more memory-efficient than submit for large lists
#             executor.map(download_hrrr_hour, chunk)
            
#     print("🏁 All Batches Complete!")










# if __name__ == "__main__":
#     all_dates = []
#     for year in range(2020, 2026): # 2020 to 2025
#         days = pd.date_range(start=f"{year}-04-01", end=f"{year}-11-15", freq='D')
#         for day in days:
#             for hour in range(15, 24):
#                 all_dates.append(day.replace(hour=hour))

#     print(f"🚀 Total files to process: {len(all_dates)}")
#     if len(all_dates) > 0:
#         print(f"First task: {all_dates[0]}")
    
#     # PARALLEL PROCESSING: max_workers is the number of files to download in parallel
#     with ThreadPoolExecutor(max_workers=10) as executor:
#         # This creates a dictionary of {future_object: date}
#         future_to_date = {executor.submit(download_hrrr_hour, dt): dt for dt in all_dates}
        
#         # This pulls the results as they finish, regardless of order
#         for future in as_completed(future_to_date):
#             try:
#                 future.result() # This triggers the prints/errors inside your function
#             except Exception as e:
#                 print(f"Critial Thread Error: {e}")
        

# import os
# import pandas as pd
# from herbie import Herbie
# import warnings
# from pathlib import Path
# import xarray as xr
# import gc  # Garbage Collector

# # Force Xarray to NOT use Dask (Dask can cause ghost memory bloat in loops)
# xr.set_options(use_new_combine_kwarg_defaults=True)
# warnings.filterwarnings("ignore")

# # --- Configuration ---
# STATION_ID = "KBGR"
# STATION_LAT, STATION_LON = 44.8074, -68.8281
# station_coords = pd.DataFrame({"longitude": [STATION_LON], "latitude": [STATION_LAT]})

# # ⚠️ RECOMMENDATION: Use a simple local path first
# output_dir = Path(r"C:\TdAI_Data") / STATION_ID 
# output_dir.mkdir(parents=True, exist_ok=True)

# def download_hrrr_hour(dt):
#     file_timestamp = dt.strftime("%Y%m%d_%H%M")
#     save_path = output_dir / f"hrrr_{STATION_ID.lower()}_{file_timestamp}.csv"
    
#     if save_path.exists():
#         return # Silent skip for speed

#     try:
#         # 1. Initialize Herbie
#         H = Herbie(dt, model="hrrr", product="prs", fxx=0, verbose=False, priority=['aws', 'google'])
        
#         # 2. Open Xarray with 'load=True' to force it into RAM immediately 
#         # then close the connection to the GRIB file.
#         search = ":(TMP|DPT|UGRD|VGRD):(1013|1000|975|950|925|900|875|850|825|800|775|750|725|700|675|650|625|600|575|550|525|500) mb"
        
#         with H.xarray(search) as ds:
#             ds_point = ds.herbie.pick_points(station_coords).load() # .load() is key
#             df = ds_point.to_dataframe().reset_index()
        
#         # 3. Clean and Save
#         if 'longitude' in df.columns:
#             df['longitude'] = df['longitude'].apply(lambda x: x - 360 if x > 180 else x)
        
#         df['valid_time'] = dt
#         df.to_csv(save_path, index=False)
#         print(f"✅ Saved: {file_timestamp}")

#         # 4. EXPLICIT CLEANUP (Crucial for Spyder)
#         del df, ds_point
#         gc.collect() 

#     except Exception as e:
#         print(f"❌ Error {file_timestamp}: {str(e)[:50]}")

# if __name__ == "__main__":
#     all_dates = []
#     for year in range(2020, 2026): 
#         days = pd.date_range(start=f"{year}-04-01", end=f"{year}-11-15", freq='D')
#         for day in days:
#             for hour in range(15, 24):
#                 all_dates.append(day.replace(hour=hour))

#     print(f"🚀 Total files: {len(all_dates)}")
    
#     # 📉 STABILITY TWEAK: 
#     # Try running purely SEQUENTIALLY first to see if it still crashes.
#     # If this works for 100 files, we can re-enable parallel.
#     for i, dt in enumerate(all_dates):
#         download_hrrr_hour(dt)
        
#         # Periodic status update
#         if i % 50 == 0:
#             print(f"--- Processed {i} / {len(all_dates)} ---")







