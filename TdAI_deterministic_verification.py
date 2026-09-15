"""
TdAI Deterministic Verification-Only Pipeline

Runs completely independently of forecast generation - no HRRR download, no
NBM bulletin parsing, no NDFD fetch, no model inference, and no import from
TdAI_deterministic_operational.py. It only catches up each station's output
ledger on ASOS Ground Truth for any row whose valid_time has already
passed, computing Raw NBM Error, Post TdAI Error, TdAI Skill Score, NDFD
Error, and NDFD Skill Score for "Active" rows.

Runs on its own schedule (~2130Z, shortly after the 21Z valid time) via
TdAI_verify.yml, separately from the 0245Z/1445Z forecast-generating cron -
so a day's verification happens right after it's actually observable,
rather than waiting for the next forecast run (which could be several
hours later, and after this refactor no longer does any verification of
its own at all) to happen to touch that row.
"""

import os
import io
import time
import datetime
import requests
import numpy as np
import pandas as pd

################################## INPUTS ####################################
STATIONS = ['CAR', 'FVE', 'HUL', 'MLT', 'GNR', 'BGR']

# Mirrors TdAI_deterministic_operational.py's OUTPUT_HEADERS - kept as a
# separate copy here (rather than imported) so this script has zero
# dependency on the forecast-generation script. Keep the two in sync if the
# ledger schema ever changes on either side.
OUTPUT_HEADERS = [
    'valid_time', 'TdAI Run Time (UTC)', 'TdAI Status', 'NBM Temperature (F)',
    'NBM Dewpoint (F)', 'NBM Max Wind Gust 15-21Z (kts)', 'NDFD Dewpoint (F)',
    'TdAI Predicted Bias (F)', 'TdAI Corrected Dewpoint (F)', 'TdAI Top Drivers',
    'ASOS Ground Truth Dewpoint (F)', 'Raw NBM Error (F)', 'Post TdAI Error (F)', 'TdAI Skill Score (%)',
    'NDFD Error (F)', 'NDFD Skill Score (%)'
]
###############################################################################


def verify_pending_rows(station, base_path):
    """Scans a station's existing output ledger for any row whose valid_time
    has passed (+ a 15-minute grace period) but still lacks ASOS Ground
    Truth, and backfills it via a bulk ASOS pull - computing Raw NBM Error,
    Post TdAI Error, TdAI Skill Score, NDFD Error, and NDFD Skill Score for
    "Active" rows."""
    output_dir = os.path.join(base_path, "deterministic_output/")
    output_csv_path = os.path.join(output_dir, f"TdAI_deterministic_operational_{station}.csv")

    if not os.path.exists(output_csv_path):
        print(f"⚠️ K{station}: no output ledger found yet. Nothing to verify.")
        return

    combined_log_df = pd.read_csv(output_csv_path)
    for col in OUTPUT_HEADERS:
        if col not in combined_log_df.columns:
            combined_log_df[col] = np.nan

    current_time_utc = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    combined_log_df['valid_time'] = pd.to_datetime(combined_log_df['valid_time'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
    combined_log_df_dt = pd.to_datetime(combined_log_df['valid_time'])

    missing_mask = combined_log_df['ASOS Ground Truth Dewpoint (F)'].isna() & (combined_log_df_dt + datetime.timedelta(minutes=15) <= current_time_utc)
    missing_indices = combined_log_df[missing_mask].index

    # Rows that are already ASOS-verified (and have a Raw NBM Error to use
    # as the skill-score denominator) but are still missing NDFD Error/Skill
    # Score - e.g. a row verified before NDFD Dewpoint existed, with NDFD
    # data added to the ledger afterward. No network call needed for these;
    # everything required is already sitting in the row.
    ndfd_backfill_mask = (
        combined_log_df['ASOS Ground Truth Dewpoint (F)'].notna()
        & combined_log_df['NDFD Dewpoint (F)'].notna()
        & combined_log_df['Raw NBM Error (F)'].notna()
        & combined_log_df['NDFD Skill Score (%)'].isna()
    )
    ndfd_backfill_indices = combined_log_df[ndfd_backfill_mask].index

    if len(missing_indices) == 0 and len(ndfd_backfill_indices) == 0:
        print(f"✅ K{station}: no rows currently awaiting verification.")
        return

    if len(missing_indices) > 0:
        print(f"\n🔄 Found {len(missing_indices)} historical rows awaiting real-time verification for K{station}...")
        missing_vtimes = pd.to_datetime(combined_log_df.loc[missing_indices, 'valid_time'])
        start_date = missing_vtimes.min() - datetime.timedelta(days=1)
        end_date = missing_vtimes.max() + datetime.timedelta(days=1)

        print(f"📡 Pooling bulk ASOS data matrix from server registry: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")

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
                    print("   ✅ Bulk observation database compiled and cached locally in workflow memory.")
                else:
                    # This used to fail completely silently - a 200 response
                    # with an empty body or missing 'dwpf' column (a real,
                    # observed transient IEM mesonet hiccup) left no trace at
                    # all in the log, making a skipped station look identical
                    # to "nothing was missing to begin with."
                    print(f"   ⚠️ Bulk ASOS response for K{station} was empty or missing expected columns "
                          f"(got {len(bulk_asos_df)} rows) - will retry on the next run.")
            else:
                print(f"   ⚠️ Bulk ASOS request for K{station} returned status {res.status_code} - will retry on the next run.")
        except Exception as e:
            print(f"   ❌ Network latency during bulk dataset retrieval: {e}")

        if not bulk_asos_df.empty and 'rounded_valid_time_str' in bulk_asos_df.columns:
            for idx in missing_indices:
                v_time = pd.to_datetime(combined_log_df.loc[idx, 'valid_time'])
                target_vtime_str = v_time.strftime('%Y-%m-%d %H:%M:%S')
                v_status = str(combined_log_df.loc[idx, 'TdAI Status']).strip()

                print(f"   └── Processing validation row for: {v_time.strftime('%Y-%m-%d %H:%M UTC')} [Status: {v_status}]")
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
                            tdai_dpt = float(combined_log_df.loc[idx, 'TdAI Corrected Dewpoint (F)'])

                            r_nbm_err = nbm_dpt - asos_gt
                            p_tdai_err = tdai_dpt - asos_gt
                            skill_score = (1.0 - (abs(p_tdai_err) / abs(r_nbm_err))) * 100 if abs(r_nbm_err) > 0 else 0.0

                            combined_log_df.loc[idx, 'Raw NBM Error (F)'] = round(r_nbm_err, 2)
                            combined_log_df.loc[idx, 'Post TdAI Error (F)'] = round(p_tdai_err, 2)
                            combined_log_df.loc[idx, 'TdAI Skill Score (%)'] = round(skill_score, 1)

                            ndfd_dpt_raw = combined_log_df.loc[idx, 'NDFD Dewpoint (F)']
                            if pd.notna(ndfd_dpt_raw):
                                ndfd_err = float(ndfd_dpt_raw) - asos_gt
                                ndfd_skill_score = (1.0 - (abs(ndfd_err) / abs(r_nbm_err))) * 100 if abs(r_nbm_err) > 0 else 0.0
                                combined_log_df.loc[idx, 'NDFD Error (F)'] = round(ndfd_err, 2)
                                combined_log_df.loc[idx, 'NDFD Skill Score (%)'] = round(ndfd_skill_score, 1)

                            print(f"        ✅ Active Row Validated! ASOS: {asos_gt}F | TdAI Skill: {round(skill_score, 1)}%")
                        else:
                            print(f"        ... Bypassed Row Validated! Observed ASOS Td: {asos_gt}F (Calculations omitted).")

    if len(ndfd_backfill_indices) > 0:
        print(f"\n🔁 Backfilling NDFD Error/Skill Score for {len(ndfd_backfill_indices)} already-verified row(s) at K{station}...")
        for idx in ndfd_backfill_indices:
            asos_gt = float(combined_log_df.loc[idx, 'ASOS Ground Truth Dewpoint (F)'])
            r_nbm_err = float(combined_log_df.loc[idx, 'Raw NBM Error (F)'])
            ndfd_dpt = float(combined_log_df.loc[idx, 'NDFD Dewpoint (F)'])

            ndfd_err = ndfd_dpt - asos_gt
            ndfd_skill_score = (1.0 - (abs(ndfd_err) / abs(r_nbm_err))) * 100 if abs(r_nbm_err) > 0 else 0.0
            combined_log_df.loc[idx, 'NDFD Error (F)'] = round(ndfd_err, 2)
            combined_log_df.loc[idx, 'NDFD Skill Score (%)'] = round(ndfd_skill_score, 1)
            print(f"   └── K{station} {combined_log_df.loc[idx, 'valid_time']}: NDFD Error={round(ndfd_err, 2)}F, NDFD Skill={round(ndfd_skill_score, 1)}%")

    combined_log_df.to_csv(output_csv_path, index=False)
    print(f"💾 Verification sync complete for K{station} → {output_csv_path}")


def main():
    base_path = "./"

    print("🔍 INITIALIZING TdAI DETERMINISTIC VERIFICATION-ONLY PASS")
    print("=" * 70)

    for station in STATIONS:
        try:
            verify_pending_rows(station, base_path)
        except Exception as e:
            print(f"❌ K{station} verification pass failed: {e}")
            continue

        # A real run's logs showed the IEM mesonet bulk-ASOS endpoint
        # silently returning empty/non-CSV responses starting with the 3rd
        # of 6 back-to-back station requests (the first two always
        # succeeded, every one after consistently failed) - almost
        # certainly a rate limit/throttle tripped by hitting the same
        # service six times with zero delay. A short pause between stations
        # avoids that.
        time.sleep(2)

    print("\n✨ VERIFICATION COMPLETE!")

if __name__ == "__main__":
    main()
