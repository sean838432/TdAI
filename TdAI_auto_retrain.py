"""
TdAI Monthly Auto-Retrain Pipeline

Intended to run via an external cron (Google Cloud Scheduler hitting this
repo's workflow_dispatch endpoint) on the 1st of each month. It:

    1) Reads each station/cycle's existing training dataset CSV to find the
       last calendar day already compiled.
    2) Walks forward one calendar day at a time (from that day through
       yesterday), downloading that day's HRRR soundings (Herbie/AWS+Google),
       NBM text bulletins (S3 archive), for ALL 6 stations at once per
       network call, computing the engineered feature row for each
       station/cycle, then discarding the raw HRRR/NBM data before moving to
       the next day. ASOS ground truth is bulk-fetched once per station up
       front (cheap, one request covers the whole backfill range).
    3) Appends any genuinely new rows (deduped against what's already on
       disk) to model_training_AUTO/training_dataset/ - a SEPARATE sandbox
       directory from the live model_training_STATIC/training_dataset/, seeded
       from it on first use (copied once, never overwritten afterward) so
       every later run keeps building on the AUTO copy instead of the live
       one. The live directory is never written to by this script.
    4) Enforces a trailing SLIDING_WINDOW_YEARS-year (default 6) rolling
       window on that same AUTO dataset - e.g. once a run has backfilled
       through March 2027, everything from before March 2021 is dropped on
       that same run, so the dataset's span stays constant over time
       instead of growing forever.
    5) Retrains all 24 deterministic AND all 24 probabilistic station/cycle
       models on the resulting AUTO dataset and deploys them to
       model_training_AUTO/trained_models/ - again a sandbox, never the
       live model_training_STATIC/trained_models/ - after backing up the previous
       AUTO models to model_training_AUTO/trained_models_backup/, so a bad
       retrain can be rolled back by restoring that folder. Promoting
       AUTO's output into the live model_training_STATIC/ directories is a
       separate, deliberate step left for a human to do (or a future
       script) - this pipeline never touches production data or
       production models on its own.

This mirrors, feature-for-feature, model_training_STATIC/TdAI_v3.1_Training_Dataset_
Compilation.py (compile step), model_training_STATIC/TdAI_v3.1_Deterministic_
TRAINING.py, and model_training_STATIC/TdAI_v3.1_Probabilistic_TRAINING.py (both
train steps, PRODUCTION_MODE=True) - see those files for the authoritative
feature/training definitions this script is kept in sync with. Both model
families train on the exact same TdAI_Training_Data_{station}_{cycle}.csv
files - only the model architecture differs. Processing one day at a time
(rather than pre-downloading the whole backfill range) is deliberate: a full
month of raw HRRR+NBM held simultaneously would be close to a GitHub Actions
runner's usable disk.

NOTE ON NBM PARSING: this script deliberately reuses data_download/NBM_
download.py's older `line.replace('-', ' -')` negative-number workaround
instead of the operational script's corrected `_parse_nbm_token` tokenizer,
so newly-appended rows stay consistent with the ~5,500 existing historical
rows that were produced by the older method. If that's ever revisited, it
should be revisited for both the historical archive and this script together.
"""

import os
import io
import re
import gc
import time
import shutil
import random
import warnings
import datetime
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
import boto3
from botocore import UNSIGNED
from botocore.client import Config
from botocore.exceptions import ClientError
from herbie import Herbie
from sklearn.ensemble import HistGradientBoostingRegressor
from lightgbm import LGBMRegressor
import joblib

warnings.filterwarnings("ignore", message="This pattern is interpreted as a regular expression")


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


seed_everything(42)

################################## INPUTS ####################################
# TEST MODE: set True to recompute a specific date range (any range - past
# dates already in the training dataset work great for validation) and
# PRINT a side-by-side comparison against whatever's already on disk,
# instead of running the real backfill/retrain. Nothing is written to disk
# in this mode - it's read-only against the live training dataset and pure
# network/compute otherwise. Dates are inclusive, 'YYYY-MM-DD', UTC.
TEST_MODE = True
TEST_START_DATE = '2026-06-15'
TEST_END_DATE = '2026-06-15'

# WORKFLOW TEST SETUP: set True to rebuild model_training_AUTO/ from
# scratch as a realistic whole-pipeline test rig - each station/cycle's
# REAL, full-size live training CSV is copied over with just its single
# most recent valid_time row dropped, so the very next real run (TEST_MODE
# back to False) sees itself as exactly one cycle behind and does one
# small, real backfill day, then a real sliding-window pass (meaningful
# here, since the full 6 years of real data is present - a tiny synthetic
# dataset can't exercise that), then a real backup + full deterministic +
# probabilistic retrain at near-production dataset size. When this is True
# the script ONLY does this setup and exits - it does not itself run the
# backfill/retrain. Never touches the live model_training_STATIC/ directory;
# rebuilds model_training_AUTO/ from scratch each time so repeated test
# runs behave identically. Safe to re-run as many times as you want.
SETUP_WORKFLOW_TEST = False

STATIONS = ['CAR', 'FVE', 'HUL', 'MLT', 'GNR', 'BGR']

STATION_COORDS = {
    'CAR': (46.870490, -68.017221),
    'FVE': (47.285172, -68.307131),
    'HUL': (46.118457, -67.792894),
    'MLT': (45.647771, -68.692475),
    'GNR': (45.462979, -69.554546),
    'BGR': (44.8074, -68.8281),
}

# Matches model_training_STATIC/TdAI_v3.1_Training_Dataset_Compilation.py's CYCLES
CYCLES = [
    {'name': '15z_Day1', 'hrrr_init_hour': 12, 'hrrr_fxx': 9, 'nbm_cycle_hour': '13', 'target_day_offset': 0},
    {'name': '15z_Day2', 'hrrr_init_hour': 12, 'hrrr_fxx': 33, 'nbm_cycle_hour': '13', 'target_day_offset': 1},
    {'name': '03z_Day1', 'hrrr_init_hour': 0, 'hrrr_fxx': 21, 'nbm_cycle_hour': '01', 'target_day_offset': 0},
    {'name': '03z_Day2', 'hrrr_init_hour': 0, 'hrrr_fxx': 45, 'nbm_cycle_hour': '01', 'target_day_offset': 1},
]
CYCLE_NAMES = [c['name'] for c in CYCLES]

# Matches model_training_STATIC/TdAI_v3.1_Probabilistic_TRAINING.py
TARGET_QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]

# Rolling training window: after each backfill, rows older than this many
# years before "today" are dropped from model_training_AUTO/training_dataset/
# - e.g. once this script has backfilled through March 2027, rows from
# before March 2021 get dropped on that same run, so the dataset always
# spans a trailing SLIDING_WINDOW_YEARS-year window instead of growing
# unbounded.
SLIDING_WINDOW_YEARS = 6

# HRRR isobaric levels required for every engineered feature - matches
# rh_1000...rh_500 in TARGET_COLUMN_ORDER exactly. A day whose sounding is
# missing any of these levels is dropped for that station/cycle (mirrors the
# original compile script's dropna()).
REQUIRED_LEVELS = [1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 725,
                   700, 675, 650, 625, 600, 575, 550, 525, 500]
_LEVELS_PATTERN = "(" + "|".join(str(l) for l in REQUIRED_LEVELS) + ") mb"
# TMP and DPT are fetched as two SEPARATE Herbie searches rather than one
# combined ":(TMP|DPT):..." string - verified live that the combined search
# silently truncates to only 18 of the 21 requested levels (cfgrib's
# hypercube merge drops 1000/975/950mb with no error/warning when TMP+DPT
# are merged together), while each variable fetched alone reliably returns
# all 21 levels.
TMP_SEARCH_STRING = ":TMP:" + _LEVELS_PATTERN
DPT_SEARCH_STRING = ":DPT:" + _LEVELS_PATTERN

NBM_BUCKET = 'noaa-nbm-grib2-pds'

# Live source directory - the original, manually-curated training data
# this script reads (read-only, one-time seed) but never writes to.
LIVE_DIR = "model_training_STATIC"

# Sandbox output root - this script writes every training-dataset/model
# artifact it produces here instead of touching LIVE_DIR directly. See
# module docstring.
AUTO_DIR = "model_training_AUTO"

TARGET_COLUMN_ORDER = [
    'Target Error (F)', 'NBM Temperature (F)', 'NBM Cloud Cover (%)',
    'NBM Mixing Height (100s ft AGL)', 'NBM Wind Speed (kts)', 'NBM Wind Direction (deg)',
    'NBM RH (%)', 'hrrr_lpw (mm)', '1000mb-700mb Lapse Rate (C/km)', '700mb-500mb Lapse Rate (C/km)',
] + [f'rh_{lvl}' for lvl in REQUIRED_LEVELS] + ['sin_season', 'cos_season']
###############################################################################


####################################################################
#                                                                  #
#            SHARED FEATURE-ENGINEERING MATH (VERBATIM)            #
#      Copied from TdAI_v3.1_Training_Dataset_Compilation.py       #
#          so the formulas can never silently drift apart          #
#                                                                  #
####################################################################

def calculate_lpw_vectorized(df, all_levels):
    g = 9.80665
    rho_w = 1000.0
    q_matrix = []
    for lvl in all_levels:
        p = float(lvl)
        dpt_col = f'dpt_{lvl}'
        e = 6.1094 * np.exp((17.625 * (df[dpt_col])) / (df[dpt_col] + 243.04))
        w = 0.622 * e / (p - e)
        q = w / (1.0 + w)
        q_matrix.append(q.values)
    q_matrix = np.array(q_matrix)
    lpw_total = np.zeros(len(df))
    for i in range(len(all_levels) - 1):
        p_high = float(all_levels[i])
        p_low = float(all_levels[i + 1])
        dp = (p_high - p_low) * 100.0
        q_avg = (q_matrix[i] + q_matrix[i + 1]) / 2.0
        lpw_total += (q_avg * dp) / (g * rho_w) * 1000.0
    return lpw_total


def calculate_lapse_rate_vectorized(df, p_bottom, p_top):
    t_bottom_col = f't_{p_bottom}'
    t_top_col = f't_{p_top}'
    if t_bottom_col not in df.columns or t_top_col not in df.columns:
        return np.nan
    t_bottom = df[t_bottom_col]
    t_top = df[t_top_col]
    delta_t = t_bottom - t_top
    t_mean_k = ((t_bottom + t_top) / 2.0) + 273.15
    dz_meters = (287.05 * t_mean_k / 9.80665) * np.log(float(p_bottom) / float(p_top))
    dz_km = dz_meters / 1000.0
    return delta_t / dz_km


####################################################################
#                                                                  #
#                        ASOS GROUND TRUTH                         #
#                                                                  #
####################################################################

def fetch_bulk_asos(station, start_date, end_date):
    """One bulk request per station covering the whole backfill range.
    Mirrors data_download/ASOS_download.py's request shape and its
    closest-to-the-hour dedup fix exactly. Returns {rounded Timestamp:
    dewpoint (F)}."""
    params = {
        'station': station, 'data': ['dwpf', 'valid'],
        'year1': start_date.year, 'month1': start_date.month, 'day1': start_date.day,
        'year2': end_date.year, 'month2': end_date.month, 'day2': end_date.day,
        'format': 'onlycomma', 'tz': 'UTC', 'missing': 'M', 'report_type': 3,
    }

    # The IEM mesonet endpoint occasionally throws a transient 503 (observed
    # live - not a rate limit tripped by us specifically, since it can hit
    # the 1st/2nd request of a run just as easily as the 6th) - a short
    # retry with backoff clears it almost every time.
    for attempt in range(3):
        try:
            res = requests.get('https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py', params=params, timeout=60)
            if res.status_code != 200:
                print(f"   ⚠️ Bulk ASOS request for K{station} returned status {res.status_code} (attempt {attempt + 1}/3).")
                time.sleep(5 * (attempt + 1))
                continue
            df = pd.read_csv(io.StringIO(res.text), comment='#', skiprows=1,
                              names=['Station', 'valid_time', 'ASOS Dewpoint (F)'], na_values='M')
            df = df.dropna(subset=['ASOS Dewpoint (F)'])
            if df.empty:
                return {}
            df['valid_time'] = pd.to_datetime(df['valid_time'], format='%Y-%m-%d %H:%M')
            hour_bucket = df['valid_time'].dt.round('h')
            minutes_from_hour = (df['valid_time'] - hour_bucket).abs()
            df = (df.assign(_hb=hour_bucket, _mfh=minutes_from_hour)
                    .sort_values('_mfh')
                    .drop_duplicates(subset='_hb', keep='first'))
            return dict(zip(df['_hb'], df['ASOS Dewpoint (F)']))
        except Exception as e:
            print(f"   ❌ Bulk ASOS fetch failed for K{station} (attempt {attempt + 1}/3): {e}")
            time.sleep(5 * (attempt + 1))

    print(f"   ❌ Bulk ASOS request for K{station} failed after 3 attempts - will retry on the next run.")
    return {}


####################################################################
#                                                                  #
#                          HRRR (PER DAY)                          #
#                                                                  #
####################################################################

def _fetch_variable_at_points(H, search_string, station_coords, varname):
    """Fetches one variable subset to REQUIRED_LEVELS at all station
    points in a single GRIB2 byte-range fetch. Returns a DataFrame with
    columns [point_stid, isobaricInhPa, varname] (Kelvin), or None on
    failure."""
    ds = None
    try:
        ds = H.xarray(search_string)
        if isinstance(ds, list):
            frames = []
            for d in ds:
                p = d.herbie.pick_points(station_coords).load().swap_dims({"point": "point_stid"}).to_dataframe().reset_index()
                if varname in p.columns and 'isobaricInhPa' in p.columns:
                    frames.append(p[['point_stid', 'isobaricInhPa', varname]])
            if not frames:
                return None
            return pd.concat(frames, ignore_index=True)
        else:
            p = ds.herbie.pick_points(station_coords).load().swap_dims({"point": "point_stid"}).to_dataframe().reset_index()
            if varname not in p.columns or 'isobaricInhPa' not in p.columns:
                return None
            return p[['point_stid', 'isobaricInhPa', varname]]
    finally:
        if ds is not None:
            if isinstance(ds, list):
                for d in ds:
                    d.close()
            else:
                ds.close()


def download_hrrr_day(run_date, init_hour, fxx, station_coords):
    """Fetches one HRRR run/lead-hour, pulling all 6 stations out of two
    GRIB2 subset fetches (TMP, then DPT - see _fetch_variable_at_points)
    via pick_points, merged on (station, level). Returns {station:
    {'valid_time':..., 'levels': DataFrame indexed by isobaricInhPa with
    columns t/dpt in Kelvin}}."""
    init_time = run_date.replace(hour=init_hour, minute=0, second=0, microsecond=0)
    valid_time = init_time + pd.Timedelta(hours=fxx)
    try:
        H = Herbie(init_time, model="hrrr", product="prs", fxx=fxx, verbose=False, priority=['aws', 'google'])

        tmp_df = _fetch_variable_at_points(H, TMP_SEARCH_STRING, station_coords, 't')
        dpt_df = _fetch_variable_at_points(H, DPT_SEARCH_STRING, station_coords, 'dpt')
        if tmp_df is None or dpt_df is None:
            return {}

        df = pd.merge(tmp_df, dpt_df, on=['point_stid', 'isobaricInhPa'], how='inner')

        out = {}
        for station in station_coords['stid']:
            sdf = df[df['point_stid'] == station][['isobaricInhPa', 't', 'dpt']].dropna()
            if sdf.empty:
                continue
            out[station] = {'valid_time': valid_time, 'levels': sdf.set_index('isobaricInhPa')[['t', 'dpt']]}
        return out

    except Exception as e:
        print(f"   ❌ HRRR {init_time.strftime('%Y%m%d')} {init_hour:02d}z f{fxx:02d}: {str(e)[:80]}")
        return {}
    finally:
        gc.collect()


####################################################################
#                                                                  #
#                           NBM (PER DAY)                          #
#                                                                  #
####################################################################

def extract_station_blocks(body, station_ids):
    """Verbatim from data_download/NBM_download.py: streams the S3 bulletin
    body, keeping only the raw blocks for the requested stations and
    stopping as soon as every one has been found."""
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
            else:
                current_lines.append(line)
    finally:
        body.close()
    return found


def parse_nbm_block(lines, station_id):
    """Parses one already-extracted station block into its raw forecast
    arrays. Deliberately mirrors NBM_download.py's process_data() -
    including its older negative-number workaround - rather than the
    operational script's newer tokenizer, for consistency with existing
    historical training rows (see module docstring)."""
    target_found = False
    nbm_run_time = None
    utc, tmp, dpt, sky, mix, wsp, wdr = [], [], [], [], [], [], []

    for line in lines:
        if station_id in line:
            target_found = True
            date_match = re.search(r'(\d{1,2}/\d{1,2}/\d{4})', line)
            time_match = re.search(r'(\d{4}) UTC', line)
            if date_match and time_match:
                date_obj = datetime.datetime.strptime(date_match.group(1), '%m/%d/%Y')
                nbm_run_time = f"{date_obj.strftime('%Y%m%d')}{time_match.group(1)[:2]}"

        if target_found:
            line_fixed = line.replace('-', ' -')
            parts = line_fixed.strip().split()
            if not parts:
                continue
            tag = parts[0]
            vals = parts[1:]
            if 'UTC' in tag:
                utc = vals
            elif 'DPT' in tag:
                dpt = vals
            elif 'TMP' in tag:
                tmp = vals
            elif 'SKY' in tag:
                sky = vals
            elif 'WSP' in tag:
                wsp = vals
            elif 'WDR' in tag:
                wdr = vals
            elif 'MHT' in tag:
                mix = vals
                target_found = False
                break

    if not nbm_run_time or not (len(utc) == len(tmp) == len(dpt) == len(sky) == len(mix) == len(wsp) == len(wdr)):
        return None
    return nbm_run_time, utc, tmp, dpt, sky, mix, wsp, wdr


def download_nbm_day(cycle_date_str, cycle_hour, stations, s3_client):
    """Streams one NBM text bulletin from the S3 archive and parses out all
    6 stations in a single network call. Returns {station: parsed tuple}."""
    s3_key = f'blend.{cycle_date_str}/{cycle_hour}/text/blend_nbstx.t{cycle_hour}z'
    try:
        obj = s3_client.get_object(Bucket=NBM_BUCKET, Key=s3_key)
    except ClientError as e:
        code = e.response.get('Error', {}).get('Code')
        if code not in ('NoSuchKey', '404'):
            print(f"   ⚠️ NBM S3 error for {cycle_date_str} {cycle_hour}z: {e}")
        return {}
    except Exception as e:
        print(f"   ⚠️ NBM S3 fetch failed for {cycle_date_str} {cycle_hour}z: {e}")
        return {}

    station_ids = [f"K{s}" for s in stations]
    blocks = extract_station_blocks(obj['Body'], station_ids)

    result = {}
    for station in stations:
        sid = f"K{station}"
        if sid not in blocks:
            continue
        parsed = parse_nbm_block(blocks[sid], sid)
        if parsed is not None:
            result[station] = parsed
    return result


def nbm_rows_by_valid_time(parsed):
    """Reconstructs absolute valid-times for every forecast hour in a
    parsed NBM block - the same day-rollover logic as NBM_download.py's
    main loop."""
    nbm_run_time, utc_l, tmp_l, dpt_l, sky_l, mix_l, wsp_l, wdr_l = parsed
    nbm_init_time = datetime.datetime.strptime(nbm_run_time, '%Y%m%d%H')
    day_start = datetime.datetime.combine(nbm_init_time.date(), datetime.time())

    rows = {}
    last_hr, days_added = -1, 0
    for fcst_hr, t, d, s, m, ws, wd in zip(utc_l, tmp_l, dpt_l, sky_l, mix_l, wsp_l, wdr_l):
        hr_int = int(fcst_hr)
        if last_hr != -1 and hr_int < last_hr:
            days_added += 1
        v_time = day_start + timedelta(days=days_added) + timedelta(hours=hr_int)
        if last_hr == -1 and hr_int < nbm_init_time.hour:
            v_time += timedelta(days=1)
            days_added += 1
        last_hr = hr_int
        try:
            rows[v_time] = {
                'NBM Temperature (F)': float(t),
                'NBM Dewpoint (F)': float(d),
                'NBM Cloud Cover (%)': float(s),
                'NBM Mixing Height (100s ft AGL)': float(m),
                'NBM Wind Speed (kts)': float(ws),
                'NBM Wind Direction (deg)': float(wd),
            }
        except ValueError:
            continue
    return rows


####################################################################
#                                                                  #
#                      BACKFILL ORCHESTRATION                      #
#                                                                  #
####################################################################

def get_next_needed_init_day(station, cycle_name, offset, training_dataset_path):
    """The next init day this station/cycle still needs, based on the
    latest valid_time already saved (valid_time = init_day + offset)."""
    path = os.path.join(training_dataset_path, f"TdAI_Training_Data_{station}_{cycle_name}.csv")
    if not os.path.exists(path):
        print(f"⚠️ K{station} {cycle_name}: no existing training dataset file - skipping this cycle in the backfill.")
        return None
    try:
        df = pd.read_csv(path, usecols=['valid_time'])
    except Exception as e:
        print(f"⚠️ K{station} {cycle_name}: could not read existing dataset ({e}) - skipping this cycle in the backfill.")
        return None
    if df.empty:
        return None
    last_valid = pd.to_datetime(df['valid_time']).max()
    last_init_day = (last_valid - pd.Timedelta(days=offset)).normalize()
    return (last_init_day + pd.Timedelta(days=1)).date()


def build_feature_row(station, cycle, hrrr_result, nbm_result, bulk_asos, target_valid_dt):
    if station not in hrrr_result or station not in nbm_result:
        return None

    nbm_rows = nbm_rows_by_valid_time(nbm_result[station])
    target_day_norm = datetime.datetime.combine(target_valid_dt.date(), datetime.time())
    window_vtimes = [target_day_norm + timedelta(hours=h) for h in (15, 18, 21)]
    window_rows = [nbm_rows[v] for v in window_vtimes if v in nbm_rows]
    row_21z = nbm_rows.get(target_day_norm + timedelta(hours=21))
    if row_21z is None or not window_rows:
        return None

    cloud_cover_avg = sum(r['NBM Cloud Cover (%)'] for r in window_rows) / len(window_rows)

    asos_gt = bulk_asos.get(station, {}).get(pd.Timestamp(target_valid_dt))
    if asos_gt is None:
        return None

    levels_df = hrrr_result[station]['levels']
    present_levels = set(int(l) for l in levels_df.index)
    if not all(lvl in present_levels for lvl in REQUIRED_LEVELS):
        return None

    wide = {}
    for lvl in REQUIRED_LEVELS:
        t_k, dpt_k = levels_df.loc[lvl, ['t', 'dpt']]
        wide[f't_{lvl}'] = float(t_k) - 273.15
        wide[f'dpt_{lvl}'] = float(dpt_k) - 273.15
    wide_df = pd.DataFrame([wide])

    lpw = calculate_lpw_vectorized(wide_df, REQUIRED_LEVELS)[0]

    lapse_1000_700 = calculate_lapse_rate_vectorized(wide_df, 1000, 700)
    lapse_1000_700 = float(lapse_1000_700.iloc[0]) if hasattr(lapse_1000_700, 'iloc') else np.nan
    lapse_700_500 = calculate_lapse_rate_vectorized(wide_df, 700, 500)
    lapse_700_500 = float(lapse_700_500.iloc[0]) if hasattr(lapse_700_500, 'iloc') else np.nan

    rh_features = {}
    for lvl in REQUIRED_LEVELS:
        t_val, dpt_val = wide_df[f't_{lvl}'].iloc[0], wide_df[f'dpt_{lvl}'].iloc[0]
        es = np.exp((17.625 * t_val) / (243.04 + t_val))
        e = np.exp((17.625 * dpt_val) / (243.04 + dpt_val))
        rh_features[f'rh_{lvl}'] = float(np.clip(100 * (e / es), 0.0, 100.0))

    nbm_tc = (row_21z['NBM Temperature (F)'] - 32) * (5.0 / 9.0)
    nbm_tdc = (row_21z['NBM Dewpoint (F)'] - 32) * (5.0 / 9.0)
    nbm_es = np.exp((17.625 * nbm_tc) / (243.04 + nbm_tc))
    nbm_e = np.exp((17.625 * nbm_tdc) / (243.04 + nbm_tdc))
    nbm_rh = float(np.clip(100 * (nbm_e / nbm_es), 0.0, 100.0))

    day_of_year = target_valid_dt.timetuple().tm_yday

    feature_row = {
        'valid_time': target_valid_dt,
        'Target Error (F)': row_21z['NBM Dewpoint (F)'] - asos_gt,
        'NBM Temperature (F)': row_21z['NBM Temperature (F)'],
        'NBM Cloud Cover (%)': cloud_cover_avg,
        'NBM Mixing Height (100s ft AGL)': row_21z['NBM Mixing Height (100s ft AGL)'],
        'NBM Wind Speed (kts)': row_21z['NBM Wind Speed (kts)'],
        'NBM Wind Direction (deg)': row_21z['NBM Wind Direction (deg)'],
        'NBM RH (%)': nbm_rh,
        'hrrr_lpw (mm)': lpw,
        '1000mb-700mb Lapse Rate (C/km)': lapse_1000_700,
        '700mb-500mb Lapse Rate (C/km)': lapse_700_500,
        **rh_features,
        'sin_season': np.sin(2 * np.pi * day_of_year / 365.25),
        'cos_season': np.cos(2 * np.pi * day_of_year / 365.25),
    }

    if any(pd.isna(v) for k, v in feature_row.items() if k != 'valid_time'):
        return None
    return feature_row


def seed_auto_dataset_dir(base_path):
    """One-time seed: copies each station/cycle's live training dataset
    CSV into model_training_AUTO/training_dataset/ the first time this
    pipeline runs against it. Never overwrites a file already there, so
    once seeded, all later runs build purely on the AUTO copy - the live
    model_training_STATIC/training_dataset/ is only ever read, never written."""
    live_dataset_path = os.path.join(base_path, LIVE_DIR, "training_dataset/")
    auto_dataset_path = os.path.join(base_path, AUTO_DIR, "training_dataset/")
    os.makedirs(auto_dataset_path, exist_ok=True)

    for station in STATIONS:
        for cycle in CYCLES:
            filename = f"TdAI_Training_Data_{station}_{cycle['name']}.csv"
            auto_file = os.path.join(auto_dataset_path, filename)
            if os.path.exists(auto_file):
                continue
            live_file = os.path.join(live_dataset_path, filename)
            if os.path.exists(live_file):
                shutil.copy2(live_file, auto_file)
                print(f"🌱 Seeded {AUTO_DIR}/training_dataset/{filename} from the live dataset.")

    return auto_dataset_path


def run_backfill(base_path):
    training_dataset_path = seed_auto_dataset_dir(base_path)

    today_utc = datetime.datetime.now(datetime.timezone.utc).date()
    end_date = today_utc - timedelta(days=1)

    next_needed = {}
    for station in STATIONS:
        for cycle in CYCLES:
            next_needed[(station, cycle['name'])] = get_next_needed_init_day(
                station, cycle['name'], cycle['target_day_offset'], training_dataset_path
            )

    valid_starts = [d for d in next_needed.values() if d is not None]
    if not valid_starts:
        print("❌ No existing training dataset files found for any station/cycle - aborting backfill "
              "(run the initial data_download/ scripts and model_training_STATIC/TdAI_v3.1_Training_Dataset_Compilation.py first).")
        return

    start_date = min(valid_starts)

    if start_date > end_date:
        print(f"✅ Training data already up to date through {end_date}. Skipping backfill.")
        return

    total_days = (end_date - start_date).days + 1
    print(f"🚀 Backfilling from {start_date} through {end_date} ({total_days} calendar day(s))...")

    print("\n📡 Pre-fetching bulk ASOS ground truth for all stations...")
    bulk_asos = {}
    asos_fetch_start = start_date - timedelta(days=1)
    asos_fetch_end = end_date + timedelta(days=2)
    for station in STATIONS:
        bulk_asos[station] = fetch_bulk_asos(station, asos_fetch_start, asos_fetch_end)
        print(f"   K{station}: {len(bulk_asos[station])} hourly observations cached.")
        time.sleep(2)

    station_coords = pd.DataFrame({
        'stid': STATIONS,
        'longitude': [STATION_COORDS[s][1] for s in STATIONS],
        'latitude': [STATION_COORDS[s][0] for s in STATIONS],
    })

    s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    current_day = start_date
    day_num = 0
    while current_day <= end_date:
        day_num += 1
        run_date = pd.Timestamp(current_day)
        print(f"\n📅 [{day_num}/{total_days}] Init day {current_day}...")

        needed_cycles = [
            cycle for cycle in CYCLES
            if any(current_day >= next_needed[(s, cycle['name'])] for s in STATIONS if next_needed[(s, cycle['name'])] is not None)
        ]

        # The 4 HRRR cycles are independent GRIB2 fetches (separate Herbie
        # objects, separate local cache files), so fetching them concurrently
        # cuts a day's wall-clock HRRR time roughly 4x - real measured
        # latency during testing showed a single day's fetches (4 cycles x
        # TMP+DPT each) taking several minutes run serially, which would
        # threaten GitHub Actions' 6-hour job limit over a multi-week
        # backfill.
        hrrr_cache = {}
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {
                pool.submit(download_hrrr_day, run_date, cycle['hrrr_init_hour'], cycle['hrrr_fxx'], station_coords): cycle['name']
                for cycle in needed_cycles
            }
            for future in as_completed(futures):
                hrrr_cache[futures[future]] = future.result()

        nbm_cache = {}
        for cycle_hour in ('13', '01'):
            relevant_cycles = [c for c in CYCLES if c['nbm_cycle_hour'] == cycle_hour]
            if any(current_day >= next_needed[(s, c['name'])] for s in STATIONS for c in relevant_cycles
                   if next_needed[(s, c['name'])] is not None):
                nbm_cache[cycle_hour] = download_nbm_day(current_day.strftime('%Y%m%d'), cycle_hour, STATIONS, s3_client)

        day_rows = {(station, cycle['name']): [] for station in STATIONS for cycle in CYCLES}
        for cycle in CYCLES:
            c_name = cycle['name']
            hrrr_result = hrrr_cache.get(c_name, {})
            nbm_result = nbm_cache.get(cycle['nbm_cycle_hour'], {})
            target_date = current_day if cycle['target_day_offset'] == 0 else current_day + timedelta(days=1)
            target_valid_dt = datetime.datetime.combine(target_date, datetime.time(21, 0))

            for station in STATIONS:
                key = (station, c_name)
                needed_day = next_needed[key]
                if needed_day is None or current_day < needed_day:
                    continue

                row = build_feature_row(station, cycle, hrrr_result, nbm_result, bulk_asos, target_valid_dt)
                if row is not None:
                    day_rows[key].append(row)

        # Flush to disk after every calendar day rather than buffering the
        # whole backfill in memory - if the job gets interrupted (timeout,
        # crash) partway through a long backfill, everything already
        # written stays saved and next month's run picks up right where
        # this one stopped instead of losing the whole range.
        flush_new_rows(training_dataset_path, day_rows)

        del hrrr_cache, nbm_cache, day_rows
        gc.collect()
        current_day += timedelta(days=1)


def flush_new_rows(training_dataset_path, new_rows):
    for (station, cycle_name), rows in new_rows.items():
        if not rows:
            continue
        dataset_full_path = os.path.join(training_dataset_path, f"TdAI_Training_Data_{station}_{cycle_name}.csv")

        new_df = pd.DataFrame(rows).set_index('valid_time')
        new_df.index = pd.to_datetime(new_df.index)

        existing_df = pd.read_csv(dataset_full_path, index_col='valid_time')
        existing_df.index = pd.to_datetime(existing_df.index)

        new_df = new_df[~new_df.index.isin(existing_df.index)]
        if new_df.empty:
            continue

        existing_cols = [c for c in existing_df.columns]
        combined_df = pd.concat([existing_df, new_df[existing_cols]]).sort_index()
        combined_df.to_csv(dataset_full_path, index=True)
        print(f"   💾 K{station} {cycle_name}: appended {len(new_df)} new row(s) -> {len(combined_df)} total.")


def enforce_sliding_window(training_dataset_path):
    """Drops rows older than SLIDING_WINDOW_YEARS before today from every
    station/cycle training CSV, so the dataset stays a trailing rolling
    window instead of growing unbounded - e.g. a run in March 2027 drops
    everything from before March 2021. Runs every time after backfilling,
    so the cutoff advances by the same amount each month as new data is
    added, keeping the window width constant over time."""
    cutoff_date = pd.Timestamp(datetime.datetime.now(datetime.timezone.utc).date()) - pd.DateOffset(years=SLIDING_WINDOW_YEARS)
    print(f"\n🪟 Enforcing {SLIDING_WINDOW_YEARS}-year sliding window (dropping rows before {cutoff_date.date()})...")

    for station in STATIONS:
        for cycle in CYCLES:
            dataset_full_path = os.path.join(training_dataset_path, f"TdAI_Training_Data_{station}_{cycle['name']}.csv")
            if not os.path.exists(dataset_full_path):
                continue

            df = pd.read_csv(dataset_full_path, index_col='valid_time')
            df.index = pd.to_datetime(df.index)

            keep_mask = df.index >= cutoff_date
            if keep_mask.all():
                continue

            dropped = int((~keep_mask).sum())
            df = df[keep_mask].sort_index()
            df.to_csv(dataset_full_path, index=True)
            print(f"   🗑️  K{station} {cycle['name']}: dropped {dropped} row(s) older than {cutoff_date.date()} -> {len(df)} remaining.")


####################################################################
#                                                                  #
#                       RETRAIN & REDEPLOY                         #
#                                                                  #
####################################################################

def strip_non_bl_busts(X_train, y_train):
    """Verbatim from TdAI_v3.1_Deterministic_TRAINING.py - removes moist
    bust cases not driven by BL dynamics from the training data only."""
    near_surface_rh_cols = ['rh_800', 'rh_825', 'rh_850', 'rh_875', 'rh_900', 'rh_925', 'rh_950', 'rh_975', 'rh_1000']
    near_surface_rh_mean = X_train[near_surface_rh_cols].mean(axis=1)
    mask = (y_train >= 3.0) & (near_surface_rh_mean > 80.0)
    if mask.any():
        print(f"      🧹 Stripping {mask.sum()} non-BL-driven moist bust rows from training data")
    return X_train[~mask], y_train[~mask]


def backup_auto_models(base_path):
    """Minimal safety net: snapshot the previous AUTO-sandbox models before
    either retrain pass overwrites anything, so a bad retrain (e.g. a bad
    month of upstream data, or a bug here) can be rolled back by restoring
    this folder. Called ONCE, before both the deterministic and
    probabilistic retrain passes, so the snapshot reflects the state before
    either one touched anything. Neither trained_models/ nor
    trained_models_backup/ here is the live model_training_STATIC/trained_models/
    - promoting a retrain into production is a separate, deliberate step."""
    live_models_path = os.path.join(base_path, AUTO_DIR, "trained_models/")
    backup_models_path = os.path.join(base_path, AUTO_DIR, "trained_models_backup/")

    if os.path.isdir(live_models_path):
        if os.path.isdir(backup_models_path):
            shutil.rmtree(backup_models_path)
        shutil.copytree(live_models_path, backup_models_path)
        print(f"🛟 Backed up previous {AUTO_DIR} models -> {backup_models_path}")


def train_and_deploy_deterministic(base_path):
    """Mirrors model_training_STATIC/TdAI_v3.1_Deterministic_TRAINING.py,
    PRODUCTION_MODE=True."""
    training_dataset_path = os.path.join(base_path, AUTO_DIR, "training_dataset/")
    live_models_path = os.path.join(base_path, AUTO_DIR, "trained_models/")

    for station in STATIONS:
        print(f"\n────────────────── Retraining K{station} (deterministic) ──────────────────")
        for c_name in CYCLE_NAMES:
            dataset_file = f"TdAI_Training_Data_{station}_{c_name}.csv"
            dataset_full_path = os.path.join(training_dataset_path, dataset_file)

            if not os.path.exists(dataset_full_path):
                print(f"   ⚠️ Dataset file missing: {dataset_file}. Skipping cycle.")
                continue

            df = pd.read_csv(dataset_full_path)
            if 'valid_time' in df.columns:
                df = df.set_index('valid_time')
            df.index = pd.to_datetime(df.index)

            X_train = df.drop(columns=['Target Error (F)'], errors='ignore')
            y_train = df['Target Error (F)']

            X_train, y_train = strip_non_bl_busts(X_train, y_train)

            gb_model = HistGradientBoostingRegressor(
                max_iter=400, learning_rate=0.03, max_depth=4,
                max_features=0.5, loss='absolute_error', random_state=42,
            )
            sample_weights = np.where(y_train >= 5.0, 5.0, np.where(y_train >= 3.0, 2.0, 1.0))
            gb_model.fit(X_train, y_train, sample_weight=sample_weights)

            station_models_output_path = os.path.join(live_models_path, f"{station}/")
            os.makedirs(station_models_output_path, exist_ok=True)
            model_save_path = os.path.join(station_models_output_path, f"tdai_deterministic_model_{station}_{c_name}.joblib")
            schema_save_path = os.path.join(station_models_output_path, f"deterministic_model_feature_schema_{station}_{c_name}.joblib")
            joblib.dump(gb_model, model_save_path)
            joblib.dump(X_train.columns.tolist(), schema_save_path)
            print(f"   💾 Retrained & deployed {c_name} ({len(X_train)} samples)")


def train_and_deploy_probabilistic(base_path):
    """Mirrors model_training_STATIC/TdAI_v3.1_Probabilistic_TRAINING.py,
    PRODUCTION_MODE=True (no holdout, so no calibration-check block - that
    only runs in that script's development/validation mode). Reads the
    identical TdAI_Training_Data_{station}_{cycle}.csv files as the
    deterministic pass - the deterministic and probabilistic models are
    two different architectures trained on the exact same dataset."""
    training_dataset_path = os.path.join(base_path, AUTO_DIR, "training_dataset/")
    live_models_path = os.path.join(base_path, AUTO_DIR, "trained_models/")

    for station in STATIONS:
        print(f"\n────────────────── Retraining K{station} (probabilistic) ──────────────────")
        for c_name in CYCLE_NAMES:
            dataset_file = f"TdAI_Training_Data_{station}_{c_name}.csv"
            dataset_full_path = os.path.join(training_dataset_path, dataset_file)

            if not os.path.exists(dataset_full_path):
                print(f"   ⚠️ Dataset file missing: {dataset_file}. Skipping cycle.")
                continue

            df = pd.read_csv(dataset_full_path)
            if 'valid_time' in df.columns:
                df = df.set_index('valid_time')
            df.index = pd.to_datetime(df.index)

            X_train = df.drop(columns=['Target Error (F)'], errors='ignore')
            y_train = df['Target Error (F)']

            X_train, y_train = strip_non_bl_busts(X_train, y_train)

            sample_weights = np.where(y_train >= 5.0, 5.0, np.where(y_train >= 3.0, 2.0, 1.0))

            probabilistic_models = {}
            for q in TARGET_QUANTILES:
                active_lambda = 8.0 if q == 0.50 else 2.0
                model = LGBMRegressor(
                    objective='regression_l1' if q == 0.50 else 'quantile',
                    alpha=q,
                    n_estimators=80,
                    learning_rate=0.04,
                    min_child_samples=30,
                    max_depth=4,
                    reg_lambda=active_lambda,
                    random_state=42,
                    colsample_bytree=0.65,
                    verbose=-1,
                )
                model.fit(X_train, y_train, sample_weight=sample_weights)
                probabilistic_models[f"q{int(q*100)}"] = model

            station_model_output_path = os.path.join(live_models_path, f"{station}/")
            os.makedirs(station_model_output_path, exist_ok=True)
            model_path = os.path.join(station_model_output_path, f"tdai_probabilistic_model_{station}_{c_name}.joblib")
            schema_path = os.path.join(station_model_output_path, f"probabilistic_model_feature_schema_{station}_{c_name}.joblib")

            joblib.dump(probabilistic_models, model_path, compress=3)
            joblib.dump(list(X_train.columns), schema_path)
            print(f"   💾 Retrained & deployed {c_name} ({len(X_train)} samples, {len(TARGET_QUANTILES)} quantiles)")


####################################################################
#                                                                  #
#                             TEST MODE                            #
#                                                                  #
####################################################################

def _print_row_comparison(station, cycle_name, target_valid_dt, existing_row, new_row):
    print(f"\n   ── K{station} {cycle_name} @ {target_valid_dt} ──")

    if new_row is None:
        print("      ❌ Recompute FAILED (missing HRRR/NBM/ASOS data for this valid time).")
        return
    if existing_row is None:
        print("      ℹ️  No existing row for this valid_time - nothing to compare against. Recomputed values:")
        for col in TARGET_COLUMN_ORDER:
            print(f"         {col}: {new_row.get(col)}")
        return

    any_mismatch = False
    for col in TARGET_COLUMN_ORDER:
        old_val = existing_row.get(col)
        new_val = new_row.get(col)
        try:
            diff = float(new_val) - float(old_val)
        except (TypeError, ValueError):
            diff = None
        mismatch = diff is None or abs(diff) > 0.05
        any_mismatch = any_mismatch or mismatch
        flag = "⚠️ " if mismatch else "   "
        print(f"      {flag}{col:35s} existing={old_val!s:>12}  recomputed={new_val!s:>12}  diff={diff}")

    print("      ✅ MATCH" if not any_mismatch else "      ⚠️  MISMATCH(ES) ABOVE")


def run_test_mode(base_path, test_start_str, test_end_str):
    """Recomputes every station/cycle for the given INIT-day range and
    prints a side-by-side diff against whatever's already saved in the
    live training dataset for the resulting valid_time(s) - read-only,
    writes nothing to disk. Great for sanity-checking this script's
    feature engineering against a date you already trust."""
    print("🧪 TEST MODE - recomputing and comparing against the live training dataset (nothing will be written)")
    print("=" * 70)

    live_dataset_path = os.path.join(base_path, LIVE_DIR, "training_dataset/")
    test_start = datetime.date.fromisoformat(test_start_str)
    test_end = datetime.date.fromisoformat(test_end_str)

    existing = {}
    for station in STATIONS:
        for cycle in CYCLES:
            path = os.path.join(live_dataset_path, f"TdAI_Training_Data_{station}_{cycle['name']}.csv")
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path, index_col='valid_time')
            df.index = pd.to_datetime(df.index)
            existing[(station, cycle['name'])] = df

    print(f"\n📡 Pre-fetching bulk ASOS ground truth for all stations ({test_start} to {test_end})...")
    bulk_asos = {}
    for station in STATIONS:
        bulk_asos[station] = fetch_bulk_asos(station, test_start - timedelta(days=1), test_end + timedelta(days=2))
        print(f"   K{station}: {len(bulk_asos[station])} hourly observations cached.")
        time.sleep(2)

    station_coords = pd.DataFrame({
        'stid': STATIONS,
        'longitude': [STATION_COORDS[s][1] for s in STATIONS],
        'latitude': [STATION_COORDS[s][0] for s in STATIONS],
    })
    s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    current_day = test_start
    while current_day <= test_end:
        run_date = pd.Timestamp(current_day)
        print(f"\n📅 Test init day {current_day}...")

        hrrr_cache = {}
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {
                pool.submit(download_hrrr_day, run_date, cycle['hrrr_init_hour'], cycle['hrrr_fxx'], station_coords): cycle['name']
                for cycle in CYCLES
            }
            for future in as_completed(futures):
                hrrr_cache[futures[future]] = future.result()

        nbm_cache = {}
        for cycle_hour in ('13', '01'):
            nbm_cache[cycle_hour] = download_nbm_day(current_day.strftime('%Y%m%d'), cycle_hour, STATIONS, s3_client)

        for cycle in CYCLES:
            c_name = cycle['name']
            hrrr_result = hrrr_cache.get(c_name, {})
            nbm_result = nbm_cache.get(cycle['nbm_cycle_hour'], {})
            target_date = current_day if cycle['target_day_offset'] == 0 else current_day + timedelta(days=1)
            target_valid_dt = datetime.datetime.combine(target_date, datetime.time(21, 0))

            for station in STATIONS:
                new_row = build_feature_row(station, cycle, hrrr_result, nbm_result, bulk_asos, target_valid_dt)

                existing_row = None
                existing_df = existing.get((station, c_name))
                if existing_df is not None and pd.Timestamp(target_valid_dt) in existing_df.index:
                    existing_row = existing_df.loc[pd.Timestamp(target_valid_dt)]

                _print_row_comparison(station, c_name, target_valid_dt, existing_row, new_row)

        del hrrr_cache, nbm_cache
        gc.collect()
        current_day += timedelta(days=1)

    print("\n✨ TEST MODE COMPLETE! (nothing was written to disk)")


def setup_workflow_test(base_path):
    """Rebuilds model_training_AUTO/ from scratch as a realistic
    whole-pipeline test rig - see the SETUP_WORKFLOW_TEST comment in the
    INPUTS block. Reads the real live training data (read-only) and writes
    only under model_training_AUTO/.

    Appends ONE synthetic row to the end of each real, full-size CSV,
    positioned so the very next real run sees itself as exactly one cycle
    behind - regardless of how far behind that file's REAL last date
    already is. This matters because some stations' live data (e.g. one
    with a backlog from before this pipeline existed) can already be
    weeks behind, and a 'quick' test must not accidentally trigger a real
    multi-hour catch-up backfill. The synthetic row's valid_time is always
    later than every real row (so get_next_needed_init_day's max() picks
    it up), and its date is chosen so the day the real pipeline goes on to
    compute next is a genuinely NEW valid_time - never the same one, so
    there's no collision/dedup with the synthetic row itself. Its other
    feature values are copied from the real last row (season features
    recomputed for the synthetic date) - a placeholder, not a real
    observation, harmless as one row among a thousand+ for a mechanics
    test but not meant to be trusted as real training data."""
    print("🧪 Setting up a whole-workflow test environment...")
    print("=" * 70)

    live_dataset_path = os.path.join(base_path, LIVE_DIR, "training_dataset/")
    auto_root = os.path.join(base_path, AUTO_DIR)
    auto_dataset_path = os.path.join(auto_root, "training_dataset/")
    auto_models_path = os.path.join(auto_root, "trained_models/")
    auto_backup_path = os.path.join(auto_root, "trained_models_backup/")

    # SAFETY GUARD #1 (primary): refuse to wipe model_training_AUTO/
    # trained_models/ if it already has ANY files in it. Real deployed
    # models are the single most expensive thing this function's cleanup
    # step deletes - they take real retraining time to regenerate and
    # aren't reconstructible from a training-dataset row-count comparison
    # alone, so their mere existence is reason enough to refuse.
    if os.path.isdir(auto_models_path) and any(os.scandir(auto_models_path)):
        print(f"🛑 REFUSING to run: {AUTO_DIR}/trained_models/ already has real deployed models in it. "
              f"Rebuilding this test rig would delete them, and this setup was only ever safe to run "
              f"before {AUTO_DIR}/ held real production output. Back up {AUTO_DIR}/ manually first if "
              f"you really want to rebuild it from scratch.")
        return

    # SAFETY GUARD #2: refuse if the training dataset itself already has
    # more rows than the live LIVE_DIR copy for any file, meaning a real
    # backfill has already run against it (even if trained_models/ was
    # somehow cleared separately).
    if os.path.isdir(auto_dataset_path):
        for station in STATIONS:
            for cycle in CYCLES:
                filename = f"TdAI_Training_Data_{station}_{cycle['name']}.csv"
                auto_file = os.path.join(auto_dataset_path, filename)
                live_file = os.path.join(live_dataset_path, filename)
                if not (os.path.exists(auto_file) and os.path.exists(live_file)):
                    continue
                auto_rows = sum(1 for _ in open(auto_file)) - 1
                live_rows = sum(1 for _ in open(live_file)) - 1
                if auto_rows > live_rows:
                    print(f"🛑 REFUSING to run: {AUTO_DIR}/training_dataset/{filename} already has "
                          f"{auto_rows} rows - more than the {live_rows} in {LIVE_DIR}/, meaning real "
                          f"backfill progress has already happened here. Rebuilding this test rig would "
                          f"destroy it. Back up {AUTO_DIR}/ manually first if you really want to rebuild "
                          f"it from scratch, then remove that backup's own copy before re-running this.")
                    return

    # Clean slate every time, so repeated test runs behave identically
    # instead of accumulating state (extra backfilled days, stale models)
    # from a previous test.
    for path in (auto_dataset_path, auto_models_path, auto_backup_path):
        if os.path.isdir(path):
            shutil.rmtree(path)
    os.makedirs(auto_dataset_path, exist_ok=True)

    today_utc = datetime.datetime.now(datetime.timezone.utc).date()

    for station in STATIONS:
        for cycle in CYCLES:
            c_name = cycle['name']
            filename = f"TdAI_Training_Data_{station}_{c_name}.csv"
            live_file = os.path.join(live_dataset_path, filename)
            if not os.path.exists(live_file):
                print(f"   ⚠️ Missing live file: {filename}. Skipping.")
                continue

            df = pd.read_csv(live_file)
            if df.empty:
                print(f"   ⚠️ {filename} is empty. Skipping.")
                continue

            # Day1 (offset 0): synthetic day = today-2 -> next needed init
            # day = today-1 -> loop computes a NEW valid_time at today-1.
            # Day2 (offset 1): synthetic day = today-1 -> next needed init
            # day = today-1 -> loop computes a NEW valid_time at today
            # (offset applied on top of today-1). Neither case's newly
            # computed valid_time equals the synthetic one just appended.
            synthetic_day = today_utc - timedelta(days=(2 if cycle['target_day_offset'] == 0 else 1))
            synthetic_valid_dt = datetime.datetime.combine(synthetic_day, datetime.time(21, 0))

            synthetic_row = df.iloc[-1].copy()
            synthetic_row['valid_time'] = synthetic_valid_dt.strftime('%Y-%m-%d %H:%M:%S')
            day_of_year = synthetic_valid_dt.timetuple().tm_yday
            if 'sin_season' in synthetic_row.index:
                synthetic_row['sin_season'] = np.sin(2 * np.pi * day_of_year / 365.25)
            if 'cos_season' in synthetic_row.index:
                synthetic_row['cos_season'] = np.cos(2 * np.pi * day_of_year / 365.25)

            appended_df = pd.concat([df, synthetic_row.to_frame().T], ignore_index=True)
            appended_df.to_csv(os.path.join(auto_dataset_path, filename), index=False)
            print(f"   ➕ {filename}: {len(df)} -> {len(appended_df)} rows "
                  f"(appended synthetic valid_time {synthetic_valid_dt} - placeholder features, not a real observation)")

    print(f"\n✅ Test environment ready under {AUTO_DIR}/.")
    print("   Now set SETUP_WORKFLOW_TEST = False and TEST_MODE = False and run this script again -")
    print("   it will see itself as exactly one cycle behind, backfill that one real day, run the")
    print("   sliding window, back up, and retrain deterministic + probabilistic models on the")
    print(f"   near-full real dataset - entirely inside {AUTO_DIR}/, never touching model_training_STATIC/.")


def main():
    base_path = "./"

    if SETUP_WORKFLOW_TEST:
        setup_workflow_test(base_path)
        return

    if TEST_MODE:
        run_test_mode(base_path, TEST_START_DATE, TEST_END_DATE)
        return

    print("🔁 INITIALIZING TdAI MONTHLY AUTO-RETRAIN PIPELINE")
    print("=" * 70)

    run_backfill(base_path)

    auto_dataset_path = os.path.join(base_path, AUTO_DIR, "training_dataset/")
    enforce_sliding_window(auto_dataset_path)

    print(f"\n🏋️ Retraining and deploying models to {AUTO_DIR}/trained_models/...")
    backup_auto_models(base_path)
    train_and_deploy_deterministic(base_path)
    train_and_deploy_probabilistic(base_path)

    print("\n✨ AUTO-RETRAIN PIPELINE COMPLETE!")


if __name__ == "__main__":
    main()
