"""
Full historical download of HRRR VVEL (omega, 21 isobaric
levels matching the existing T/Td/u/v archive), SOILW (soil moisture, 4
near-surface depths), MSLMA (mean sea level pressure - synoptic proxy), and
500mb geopotential height (synoptic proxy), for all 4 training stations
across every operational cycle (12z_f09, 12z_f33, 00z_f21, 00z_f45), 2021-2026.

"""

import pandas as pd
from herbie import Herbie
from concurrent.futures import ProcessPoolExecutor
import warnings
from pathlib import Path
import gc

warnings.filterwarnings("ignore", message="This pattern is interpreted as a regular expression")

################################## INPUTS ####################################
STATIONS = {
    'KCAR': (46.870490, -68.017221),
    'KBGR': (44.8074, -68.8281),
    'KGNR': (45.462979, -69.554546),
    'KMLT': (45.647771, -68.692475),
    'KHUL': (46.118457, -67.792894),
    'KBHB': (44.452318, -68.370777),
    'KFVE': (47.285172, -68.307131),
}

# Matches the 4 operational cycles in TdAI_v3.1_Deterministic_OPERATIONAL.py's
# CYCLES config (init hour, forecast lead hour, run label used in filenames).
CYCLES = [
    (12, 9, '12z_f09'),
    (12, 33, '12z_f33'),
    (0, 21, '00z_f21'),
    (0, 45, '00z_f45'),
]

# Matches START_DATE/END_DATE in TdAI_v3.1_Deterministic_OPERATIONAL.py (HRRRv4-only range).
YEARS = range(2021, 2027)

PRESSURE_LEVELS = [500, 525, 550, 575, 600, 625, 650, 675, 700, 725, 750, 775,
                   800, 825, 850, 875, 900, 925, 950, 975, 1000]
SOIL_DEPTHS = ['0.01', '0.04', '0.1', '0.3']  # meters below ground

output_dir = Path("/home/sean834/TdAI/data_download/HRRR_vvel_soilw_mslma_hgt")
output_dir.mkdir(parents=True, exist_ok=True)
################################################################################


def download_row(args):
    station_id, lat, lon, run_date, init_hour, fxx, run_label = args
    station_coords = pd.DataFrame({"longitude": [lon], "latitude": [lat]})

    init_time = run_date.replace(hour=init_hour, minute=0, second=0, microsecond=0)
    valid_time = init_time + pd.Timedelta(hours=fxx)
    file_timestamp = init_time.strftime("%Y%m%d")

    station_cycle_dir = output_dir / f"{station_id}_{run_label}"
    station_cycle_dir.mkdir(parents=True, exist_ok=True)
    save_path = station_cycle_dir / f"{station_id}_{run_label}_{file_timestamp}.csv"

    if save_path.exists():
        return

    try:
        H = Herbie(init_time, model="hrrr", product="prs", fxx=fxx, verbose=False, priority=['aws', 'google'])

        row = {'station': station_id, 'valid_time': valid_time, 'init_time': init_time}

        # --- VVEL at isobaric levels ---
        levels_str = "|".join(str(p) for p in PRESSURE_LEVELS)
        ds_vvel = H.xarray(f":VVEL:({levels_str}) mb")
        df_vvel = ds_vvel.herbie.pick_points(station_coords).load().to_dataframe().reset_index()
        for _, r in df_vvel.iterrows():
            row[f"vvel_{int(r['isobaricInhPa'])}"] = r['w']

        # --- SOILW at near-surface depths ---
        depths_pattern = "|".join(d.replace('.', r'\.') for d in SOIL_DEPTHS)
        ds_soil = H.xarray(f":SOILW:({depths_pattern})-(?:{depths_pattern}) m below ground")
        df_soil = ds_soil.herbie.pick_points(station_coords).load().to_dataframe().reset_index()
        for _, r in df_soil.iterrows():
            row[f"soilw_{r['depthBelowLandLayer']}"] = r['soilw']

        # --- MSLMA (synoptic proxy: surface high vs low) ---
        ds_mslp = H.xarray(":MSLMA:mean sea level")
        df_mslp = ds_mslp.herbie.pick_points(station_coords).load().to_dataframe().reset_index()
        row['mslma'] = df_mslp['mslma'].iloc[0] if 'mslma' in df_mslp.columns else df_mslp.iloc[0, -1]

        # --- 500mb height (synoptic proxy: ridge vs trough) ---
        ds_hgt = H.xarray(":HGT:500 mb")
        df_hgt = ds_hgt.herbie.pick_points(station_coords).load().to_dataframe().reset_index()
        row['hgt_500'] = df_hgt['gh'].iloc[0] if 'gh' in df_hgt.columns else df_hgt.iloc[0, -1]

        pd.DataFrame([row]).to_csv(save_path, index=False)
        print(f"✅ {station_id} {run_label} {file_timestamp}")

    except Exception as e:
        print(f"❌ {station_id} {run_label} {file_timestamp}: {str(e)[:80]}")

    finally:
        gc.collect()


if __name__ == "__main__":
    tasks = []
    for year in YEARS:
        dates = pd.date_range(start=f"{year}-03-01", end=f"{year}-11-15", freq='D')
        for station_id, (lat, lon) in STATIONS.items():
            for init_hour, fxx, run_label in CYCLES:
                for d in dates:
                    tasks.append((station_id, lat, lon, d, init_hour, fxx, run_label))

    print(f"🚀 Full download: {len(tasks)} station-cycle-days "
          f"({len(STATIONS)} stations x {len(CYCLES)} cycles x ~{len(dates)} days/year x {len(list(YEARS))} years)")

    chunk_size = 100
    for i in range(0, len(tasks), chunk_size):
        chunk = tasks[i:i + chunk_size]
        print(f"📦 Batch {i // chunk_size + 1}/{(len(tasks) - 1) // chunk_size + 1}...")
        with ProcessPoolExecutor(max_workers=8) as executor:
            list(executor.map(download_row, chunk))

    print("🏁 Full download complete!")
