"""

This script loads in NBM station data over a specified time range
from the the NOAA NBM GRIB2 PDS S3 AWS bucket and saves the data to individual
files based on the NBM run (e.g. 20251115_12z.csv).

The NBM data can be found at: https://noaa-nbm-grib2-pds.s3.amazonaws.com/index.html

Raw per-cycle bulletins (blend_nbstx) are ~30 MB and contain every NBM
station in the country. Rather than saving that whole file, the downloader
streams the S3 object and keeps only the block(s) belonging to STATIONS,
closing the connection as soon as every requested station has been found.
Each station's raw files are written to their own subfolder under
NBM_Raw_Data_Files (e.g. NBM_Raw_Data_Files/KCAR/), tiny and containing
only the raw, unmodified bulletin text for that station.

"""

import os
import re
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import boto3
from botocore import UNSIGNED
from botocore.client import Config
from botocore.exceptions import ClientError
import pandas as pd


################################## INPUTS ####################################
STATIONS = ['KCAR', 'KHUL', 'KMLT', 'KGNR', 'KBGR', 'KBHB', 'KFVE']
BUCKET_NAME = 'noaa-nbm-grib2-pds'
PRODUCT = 'blend'
SUBDIRECTORY = 'text'
CYCLE_HOUR = '13'
LOCAL_DIR = "/home/sean834/TdAI/data_download/NBM_data"
RAW_DATA_DIR = os.path.join(LOCAL_DIR, "NBM_Raw_Data_Files")
MAX_WORKERS = 16

YEARS = [2020, 2021, 2022, 2023, 2024, 2025, 2026]
START_MD = (2, 28)
END_MD = (11, 15)
##############################################################################


def extract_station_blocks(body, station_ids):
    """Stream an NBM text bulletin body and pull out only the raw blocks
    for the requested station ids, stopping the read as soon as every
    station has been found (so we don't pay to transfer the rest of the
    ~30 MB file)."""
    wanted = set(station_ids)
    found = {}
    current_id = None
    current_lines = []

    try:
        for raw_line in body.iter_lines():
            line = raw_line.decode('utf-8', errors='replace')

            if not line.strip():
                if current_id is not None:
                    found[current_id] = current_lines
                    current_id, current_lines = None, []
                    if wanted.issubset(found.keys()):
                        break
                continue

            if current_id is None:
                token = line.split()[0]
                if token in wanted:
                    current_id = token
                    current_lines = [line]
                # else: skip lines belonging to stations we don't need
            else:
                current_lines.append(line)
    finally:
        body.close()

    return found


def station_file_path(raw_dir, cycle_date, cycle_hour, station_id):
    return os.path.join(raw_dir, station_id, f"{cycle_date}_blend_nbstx_{station_id}.t{cycle_hour}z")


def download_nbm_data(s3, bucket_name, product, cycle_date, cycle_hour, subdirectory, raw_dir, station_ids):
    # Each station gets its own small raw file, so adding a new station to
    # STATIONS later only fetches the stations that aren't on disk yet -
    # stations already downloaded in a prior run are never re-fetched.
    missing = [sid for sid in station_ids if not os.path.exists(station_file_path(raw_dir, cycle_date, cycle_hour, sid))]
    if not missing:
        return True

    s3_key = f'{product}.{cycle_date}/{cycle_hour}/{subdirectory}/blend_nbstx.t{cycle_hour}z'

    try:
        obj = s3.get_object(Bucket=bucket_name, Key=s3_key)
    except ClientError as e:
        if e.response.get('Error', {}).get('Code') in ('NoSuchKey', '404'):
            return False
        print(f"Download error for {cycle_date}: {e}")
        return False

    blocks = extract_station_blocks(obj['Body'], missing)
    if not blocks:
        return False

    for sid, lines in blocks.items():
        local_path = station_file_path(raw_dir, cycle_date, cycle_hour, sid)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        tmp_path = local_path + '.tmp'
        with open(tmp_path, 'w') as f:
            f.write('\n'.join(lines))
            f.write('\n\n')
        os.replace(tmp_path, local_path)  # atomic so a crash mid-write can't leave a fake "already downloaded" file
        print(f"Saved raw file: {local_path}")

    return True


def process_data(cycle_date, cycle_hour, raw_dir, station_id):
    file_path = station_file_path(raw_dir, cycle_date, cycle_hour, station_id)
    target_found = False
    nbm_run_time = None
    utc, tmp, dpt, sky, mix, wsp, wdr = [], [], [], [], [], [], []

    if not os.path.exists(file_path):
        return None, [], [], [], [], [], [], []

    try:
        with open(file_path, 'r') as file:
            for line in file:
                if station_id in line:
                    target_found = True
                    date_match = re.search(r'(\d{1,2}/\d{1,2}/\d{4})', line)
                    time_match = re.search(r'(\d{4}) UTC', line)
                    if date_match and time_match:
                        date_obj = datetime.strptime(date_match.group(1), '%m/%d/%Y')
                        nbm_run_time = f"{date_obj.strftime('%Y%m%d')}{time_match.group(1)[:2]}"

                if target_found:
                    line_fixed = line.replace('-', ' -')  # The code struggles with negative numbers in the line with no spacing so we add a space
                    parts = line_fixed.strip().split()
                    if not parts: continue
                    tag = parts[0]
                    vals = parts[1:]

                    if 'UTC' in tag: utc = vals
                    elif 'DPT' in tag: dpt = vals
                    elif 'TMP' in tag: tmp = vals
                    elif 'SKY' in tag: sky = vals
                    elif 'WSP' in tag: wsp = vals
                    elif 'WDR' in tag: wdr = vals
                    elif 'MHT' in tag:
                        mix = vals
                        target_found = False
                        break

    except Exception as e:
        print(f"Error reading {station_id} in {cycle_date}: {e}")

    return nbm_run_time, utc, tmp, dpt, sky, mix, wsp, wdr


##############################################################################
#                             Main script execution                          #
##############################################################################

os.makedirs(RAW_DATA_DIR, exist_ok=True)

# Create Date List
all_dates = []
for year in YEARS:
    curr = datetime(year, START_MD[0], START_MD[1])
    stop = datetime(year, END_MD[0], END_MD[1])
    while curr <= stop:
        all_dates.append(curr.strftime('%Y%m%d'))
        curr += timedelta(days=1)

station_data_store = {station: [] for station in STATIONS}

print(f"Processing {len(STATIONS)} stations over {len(all_dates)} days...")

# Download every cycle in parallel. Dates whose raw file already exists on
# disk are skipped instantly inside download_nbm_data before any S3 call.
s3_client = boto3.client(
    's3',
    config=Config(signature_version=UNSIGNED, max_pool_connections=MAX_WORKERS),
)

results = {}
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
    futures = {
        pool.submit(
            download_nbm_data, s3_client, BUCKET_NAME, PRODUCT, cycle_date,
            CYCLE_HOUR, SUBDIRECTORY, RAW_DATA_DIR, STATIONS,
        ): cycle_date
        for cycle_date in all_dates
    }
    for future in as_completed(futures):
        cycle_date = futures[future]
        try:
            results[cycle_date] = future.result()
        except Exception as e:
            print(f"Failed {cycle_date}: {e}")
            results[cycle_date] = False

for cycle_date in all_dates:
    if not results.get(cycle_date):
        continue

    for sid in STATIONS:
        nbm_init_str, utc_l, tmp_1, dpt_l, sky_l, mix_l, wsp_l, wdr_l = process_data(cycle_date, CYCLE_HOUR, RAW_DATA_DIR, sid)

        if not nbm_init_str or not (len(utc_l) == len(tmp_1) == len(dpt_l) == len(sky_l) == len(mix_l) == len(wsp_l) == len(wdr_l)):
            continue

        try:
            nbm_init_time = datetime.strptime(nbm_init_str, '%Y%m%d%H')

            # Keep 15z, 18z, and 21z forecasts for both day 1 (same date as
            # initialization) and day 2 (24 hours later).
            day1_start = datetime.combine(nbm_init_time.date(), datetime.min.time())
            day2_start = day1_start + timedelta(days=1)
            target_times = {
                day + timedelta(hours=hr)
                for day in (day1_start, day2_start)
                for hr in (15, 18, 21)
            }

            last_hr, days_added = -1, 0

            for fcst_hr, t, d, s, m, ws, wd in zip(utc_l, tmp_1, dpt_l, sky_l, mix_l, wsp_l, wdr_l):
                hr_int = int(fcst_hr)
                if last_hr != -1 and hr_int < last_hr: days_added += 1

                v_time = datetime.combine(nbm_init_time.date(), datetime.min.time()) + \
                         timedelta(days=days_added) + timedelta(hours=hr_int)

                if last_hr == -1 and hr_int < nbm_init_time.hour:
                    v_time += timedelta(days=1)
                    days_added += 1

                last_hr = hr_int

                if v_time in target_times:
                    try:
                        station_data_store[sid].append({
                            'NBM Initialization Time (UTC)': nbm_init_time,
                            'NBM Forecast Valid (UTC)': v_time,
                            'NBM Temperature (F)': float(t),
                            'NBM Dewpoint (F)': float(d),
                            'NBM Cloud Cover (%)': float(s),
                            'NBM Mixing Height (100s ft AGL)': float(m),
                            'NBM Wind Speed (kts)': float(ws),
                            'NBM Wind Direction (deg)': float(wd)
                        })
                    except ValueError: continue
        except Exception:
            continue

# STEP 5: SAVE INDIVIDUAL FILES
for sid in STATIONS:
    if station_data_store[sid]:
        df = pd.DataFrame(station_data_store[sid])
        df = df.drop_duplicates(subset=['NBM Initialization Time (UTC)', 'NBM Forecast Valid (UTC)'])
        output_path = os.path.join(LOCAL_DIR, f"NBM_Master_Data_{sid}_{CYCLE_HOUR}z.csv")
        df.to_csv(output_path, index=False)
        print(f"{sid} saved: {len(df)} rows -> {output_path}")
    else:
        print(f"No data for {sid}")
