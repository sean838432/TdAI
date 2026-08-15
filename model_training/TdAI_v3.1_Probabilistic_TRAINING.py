'''
This script trains a Gradient Boosted Decision Tree (GBDT) algorithm to predict 21z NBM Td error at 21z at the 10th, 25th, 50th, 75th, and 90th percentiles.
It is trained on 6 years of HRRR, NBM, and ASOS data from 2021-2026

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
        HRRR RH at all levels (%)
        Time of year

    OUTCOME VARIABLE:
        Td error from the 01z/13z NBM forecast

    WEIGHTING SCHEME:
        Td error 3-4 F: Weight of 2
        Td error >= 5 F: Weight of 5

'''

import numpy as np
import pandas as pd
import random
import os
from lightgbm import LGBMRegressor
import joblib


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


####################################################################
#                                                                  #
#                           TRAIN TdAI MODELS                      #
#                                                                  #
####################################################################


# True  -> Trains on ALL available data (2021-2026) for live deployment
# False -> Leaves out HOLDOUT_YEAR strictly as an independent validation test bench.
PRODUCTION_MODE = True
HOLDOUT_YEAR = 2025
STATIONS = ['CAR', 'FVE', 'HUL', 'MLT', 'GNR', 'BGR']

# Define paths
base_path = "/home/sean834/TdAI/"
training_dataset_path = os.path.join(base_path, "model_training/training_dataset/")
models_output_path = os.path.join(base_path, "model_training/trained_models/")

# Define the 4 target operational cycles
CYCLE_NAMES = ['03z_Day1', '03z_Day2', '15z_Day1', '15z_Day2']

# Define the target quantiles for probabilistic regression
target_quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]

# Collects one calibration summary line per station/cycle, printed together at the end
calibration_summary_lines = []
# Collects per-cycle (coverage_50, coverage_80) tuples keyed by station, for the station-average summary
calibration_by_station = {}

print("🚀 INITIALIZING TdAI TRAINING")
print("=" * 70)

for station in STATIONS:
    print(f"\n=========================================================================")
    print(f"🏋️ INITIATING TdAI PROBABILISTIC TRAINING FOR STATION: {station}")
    print(f"=========================================================================")

    for cycle in CYCLE_NAMES:
        print(f"\n=========================================================================")
        print(f"🔄 INITIATING PROBABILISTIC TRAINING FOR {station} CYCLE: {cycle}")
        print(f"=========================================================================")

        # --- 0. LOAD THE SPECIFIC CYCLE DATASET ---
        data_path = os.path.join(training_dataset_path, f"TdAI_Training_Data_{station}_{cycle}.csv")
        if not os.path.exists(data_path):
            print(f"⚠️ Missing training data for {station}_{cycle}. Skipping...")
            continue

        master_train_df = pd.read_csv(data_path)

        # --- 1. PREPARE THE INDEX ---
        if 'valid_time' in master_train_df.columns:
            master_train_df = master_train_df.set_index('valid_time')

        # Guarantee the index is a parsed DatetimeIndex for perfect year slices
        master_train_df.index = pd.to_datetime(master_train_df.index)

        # --- 2. PREPARE X AND Y ---
        X = master_train_df.drop(columns=['Target Error (F)'], errors='ignore')
        y = master_train_df['Target Error (F)']

        # --- 3. DATA SPLIT SEGMENTATION ---
        if PRODUCTION_MODE:
            print("PRODUCTION MODE ACTIVATED: Training ensemble on 100% of available historical dataset...")
            X_train = X.copy()
            y_train = y.copy()
            X_test = pd.DataFrame()
            y_test = pd.Series(dtype=float)
            print(f"   └── Total Training Matrix: {len(X_train)} samples")
        else:
            print(f"DEVELOPMENT VALIDATION MODE: Isolating {HOLDOUT_YEAR} convective archive...")
            years = X.index.year
            is_test_year = (years == HOLDOUT_YEAR)
            is_train_year = ~is_test_year

            X_train = X[is_train_year]
            y_train = y[is_train_year]
            X_test = X[is_test_year]
            y_test = y[is_test_year]

            print(f"   ├── Training set size: {len(X_train)} samples")
            print(f"   └── Test/Validation set size (Strictly {HOLDOUT_YEAR}): {len(X_test)} samples")

        # --- 3b. STRIP NON-BL-DRIVEN MOIST BUSTS FROM TRAINING DATA ONLY ---
        near_surface_rh_cols = ['rh_800', 'rh_825', 'rh_850', 'rh_875', 'rh_900', 'rh_925', 'rh_950', 'rh_975', 'rh_1000']
        near_surface_rh_mean = X_train[near_surface_rh_cols].mean(axis=1)
        non_bl_bust_mask = (y_train >= 3.0) & (near_surface_rh_mean > 80.0)

        if non_bl_bust_mask.any():
            print(f"   🧹 Stripping {non_bl_bust_mask.sum()} non-BL-driven moist bust rows from training "
                  f"(NBM error >= 3F & mean 800-1000mb RH > 80%)")
            X_train = X_train[~non_bl_bust_mask]
            y_train = y_train[~non_bl_bust_mask]

        # Define sample weights (weight the moist bust days higher)
        sample_weights = np.where(y_train >= 5.0, 5.0, np.where(y_train >= 3.0, 2.0, 1.0))

        probabilistic_models = {}

        print("Initializing Probabilistic Quantile Regression Training...")

        for q in target_quantiles:
            print(f"   ├── Training GBDT Core Weights for Quantile: {q*100:.0f}th Percentile...")

            active_lambda = 8.0 if q == 0.50 else 2.0

            # Initialize the regressor using Quantile Loss
            model = LGBMRegressor(
                objective='regression_l1' if q == 0.50 else 'quantile', # LightGBM prefers L1 for 50th, quantile for others
                alpha=q,                  # This tells the model which exact percentile to lock onto
                n_estimators=80,          # Number of trees
                learning_rate=0.04,       # How much weight given to each new tree (generally want it small)
                min_child_samples=30,     # Before a prediction rule is created N samples must match that rule
                max_depth=4,              # How complex the trees can get. Low values force the model to focus on big, broad rules
                reg_lambda=active_lambda, # Penalizes the model for giving massive, outsized prediction weights to any single leaf choice.
                random_state=42,          # Ensures model reproducibility
                colsample_bytree=0.65,    # The percentage of predictors from which the model can choose its splits.
                verbose=-1                # Suppress warnings
            )

            # Train the model normally using your identical feature schema
            model.fit(X_train, y_train, sample_weight=sample_weights)

            # Store the trained weights inside a container dictionary matrix specific to this cycle
            probabilistic_models[f"q{int(q*100)}"] = model

        # --- 4b. CALIBRATION CHECK: DO THE PREDICTED INTERVALS COVER THE HOLDOUT YEAR AT THE EXPECTED RATE? ---
        if not PRODUCTION_MODE and len(X_test) > 1:
            q10_pred = probabilistic_models['q10'].predict(X_test)
            q25_pred = probabilistic_models['q25'].predict(X_test)
            q75_pred = probabilistic_models['q75'].predict(X_test)
            q90_pred = probabilistic_models['q90'].predict(X_test)

            y_test_vals = y_test.values

            coverage_50 = np.mean((y_test_vals >= q25_pred) & (y_test_vals <= q75_pred)) * 100.0
            coverage_80 = np.mean((y_test_vals >= q10_pred) & (y_test_vals <= q90_pred)) * 100.0

            calibration_summary_lines.append(
                f"   {station}_{cycle} (n={len(X_test)}): 25th-75th = {coverage_50:.1f}% (expect 50%)  |  10th-90th = {coverage_80:.1f}% (expect 80%)"
            )
            calibration_by_station.setdefault(station, []).append((coverage_50, coverage_80))

        # =============================================================================================
        # 💾 SAVE PIPELINE: Export the Unified Probabilistic Model Dictionary (only in production mode)
        # =============================================================================================
        # 1. Isolate the exact list of string column names used for training
        trained_features = list(X_train.columns)

        # Set the save path for the trained models for this station
        station_model_output_path = os.path.join(models_output_path, f"{station}/")
        os.makedirs(station_model_output_path, exist_ok=True)

        model_path = os.path.join(station_model_output_path, f"tdai_probabilistic_model_{station}_{cycle}.joblib")
        schema_path = os.path.join(station_model_output_path, f"probabilistic_model_feature_schema_{station}_{cycle}.joblib")

        try:
            # Save the 5-model group bundle for this specific cycle
            joblib.dump(probabilistic_models, model_path, compress=3)
            print(f"   ├── ✅ Probabilistic Ensemble exported to: '{os.path.basename(model_path)}'")

            # Save the column ordering list
            joblib.dump(trained_features, schema_path)
            print(f"   └── ✅ Feature Schema list exported to:      '{os.path.basename(schema_path)}'")

        except Exception as e:
            print(f"   ❌ Error exporting operational artifacts for {station}_{cycle}: {e}")

if calibration_summary_lines:
    print(f"\n📐 {HOLDOUT_YEAR} CALIBRATION SUMMARY (all stations/cycles)")
    print("=" * 70)
    for line in calibration_summary_lines:
        print(line)

    print(f"\n📐 {HOLDOUT_YEAR} CALIBRATION SUMMARY (station averages across cycles)")
    print("=" * 70)
    for station, coverages in calibration_by_station.items():
        avg_50 = np.mean([c50 for c50, c80 in coverages])
        avg_80 = np.mean([c80 for c50, c80 in coverages])
        print(f"   {station} (n_cycles={len(coverages)}): 25th-75th avg = {avg_50:.1f}% (expect 50%)  |  10th-90th avg = {avg_80:.1f}% (expect 80%)")

print("\n✨ ALL CYCLES COMPLETE! All probabilistic models trained and exported.")
