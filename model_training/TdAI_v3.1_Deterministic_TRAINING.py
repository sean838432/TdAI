'''
This script trains a Gradient Boosted Decision Tree (GBDT) algorithm to predict NBM Td error at 21z.
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
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor
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

print("🚀 INITIALIZING TdAI TRAINING")
print("=" * 70)

for station in STATIONS:
    print("\n\n")
    print("=" * 59)
    print(f"\n────────────────── Training TdAI on K{station} ──────────────────")
    print("")
    print("=" * 59)
    for c_name in CYCLE_NAMES:
        dataset_file = f"TdAI_Training_Data_{station}_{c_name}.csv"
        dataset_full_path = os.path.join(training_dataset_path, dataset_file)

        print(f"\n────────────────── Training Cycle Model: {c_name} ──────────────────")

        if not os.path.exists(dataset_full_path):
            print(f"⚠️ Dataset file missing: {dataset_file}. Skipping cycle.")
            continue

        # --- 1. LOAD & PREPARE DATASET ---
        df = pd.read_csv(dataset_full_path)

        if 'valid_time' in df.columns:
            df = df.set_index('valid_time')

        df.index = pd.to_datetime(df.index)

        # Separate X (Features) and y (Target Error)
        X = df.drop(columns=['Target Error (F)'], errors='ignore')
        y = df['Target Error (F)']

        # --- 2. DATA SPLIT SEGMENTATION ---
        years = X.index.year

        if PRODUCTION_MODE:
            print("🚀 PRODUCTION MODE ACTIVATED: Training model on 100% of training dataset...")
            X_train = X.copy()
            y_train = y.copy()
            X_test = pd.DataFrame()
            y_test = pd.Series(dtype=float)

            print(f"   📊 Training Matrix (Strictly 2021-2026): {len(X_train)} samples")
        else:
            print(f"🛡️ DEVELOPMENT VALIDATION MODE: Isolating {HOLDOUT_YEAR} convective archive...")
            is_test_year = (years == HOLDOUT_YEAR)
            is_train_year = (~is_test_year)

            X_train = X[is_train_year]
            y_train = y[is_train_year]
            X_test = X[is_test_year]
            y_test = y[is_test_year]

            print(f"   📊 Data Split Complete:")
            print(f"      • Training Set (All years except {HOLDOUT_YEAR})    : {len(X_train)} samples")
            print(f"      • Validation Set (Strictly {HOLDOUT_YEAR})          : {len(X_test)} samples")

        # --- 2b. STRIP NON-BL-DRIVEN MOIST BUSTS FROM TRAINING DATA ONLY ---
        near_surface_rh_cols = ['rh_800', 'rh_825', 'rh_850', 'rh_875', 'rh_900', 'rh_925', 'rh_950', 'rh_975', 'rh_1000']
        near_surface_rh_mean = X_train[near_surface_rh_cols].mean(axis=1)
        non_bl_bust_mask = (y_train >= 3.0) & (near_surface_rh_mean > 80.0)

        if non_bl_bust_mask.any():
            print(f"   🧹 Stripping {non_bl_bust_mask.sum()} non-BL-driven moist bust rows from training "
                  f"(NBM error >= 3F & mean 800-1000mb RH > 80%)")
            X_train = X_train[~non_bl_bust_mask]
            y_train = y_train[~non_bl_bust_mask]

        # --- 3. MODEL INITIALIZATION & WEIGHTING ---
        gb_model = HistGradientBoostingRegressor(
            max_iter=400,
            learning_rate=0.03,
            max_depth=4,
            max_features=0.5,
            loss='absolute_error',
            random_state=42
        )

        # Weighting scheme (higher weights for larger Td errors)
        sample_weights = np.where(y_train >= 5.0, 5.0, np.where(y_train >= 3.0, 2.0, 1.0))

        # --- 4. TRAIN MODEL ---
        print(f"   🏋️ Training TdAI for {c_name}...")
        gb_model.fit(X_train, y_train, sample_weight=sample_weights)

        # --- 5. MODEL TESTING (if not in production mode) ---
        if not PRODUCTION_MODE and len(X_test) > 1:

            # Define the operational T/Sky/RH gate for the test dataset (to mimic operational TdAI)
            operational_gate = (
                (X_test['NBM RH (%)'] <= 60.0) &
                (X_test['NBM Temperature (F)'] >= 50.0) &
                (X_test['NBM Cloud Cover (%)'] <= 60.0)
            )

            # Apply the operational gate to the training dataset predictors and target values (answers).
            # TdAI gets a chance to predict on every gated row, same as it would
            X_test_filtered = X_test[operational_gate]
            y_test_filtered = y_test[operational_gate]

            # --- 5a. SET UP THE SCORING TARGETS FOR TdAI ---

            # Dry Busts are scored as 0 (TdAI's own MAE/R2, NOT the NBM baseline)
            y_test_dry_0 = np.where(y_test_filtered < 0.0, 0.0, y_test_filtered)

            # NBM Moist busts only (NBM-ASOS Td error > 0) - same boolean mask reused
            # below for both y_test and y_pred so they stay aligned row-for-row.
            is_moist_bust = (y_test_filtered > 0.0).values
            y_test_moist_busts_only = y_test_filtered[is_moist_bust]

            # --- 5b. TdAI MODEL PREDICTIONS ---

            # Make predictions on the witheld test dataset (with the operational T/Sky/RH gate applied)
            y_pred = gb_model.predict(X_test_filtered)

            # We set any negative TdAI prediction to 0, just as it runs operationally
            y_pred = np.where(y_pred < 0, 0.0, y_pred)

            # We also isolate the TdAI predictions that correspond to NBM moist busts only (NBM-ASOS error > 0)
            y_pred_moist_busts_only = y_pred[is_moist_bust]

            # --- 5c. MODEL SKILL SCORE ---

            # *Main* skill score (TdAI scored with dry busts zeroed; NBM baseline
            # stays the true, un-zeroed error - see the two variables below)
            tdai_mae_dry_0 = mean_absolute_error(y_test_dry_0, y_pred)
            nbm_mae_true = y_test_filtered.abs().mean()
            skill_score = (1.0 - tdai_mae_dry_0 / nbm_mae_true) * 100.0

            # *Moist busts only* skill score (evaluates TdAI only on days the NBM-ASOS error > 0)
            tdai_mae_moist_busts_only = mean_absolute_error(y_test_moist_busts_only, y_pred_moist_busts_only)
            nbm_mae_moist_busts_only = y_test_moist_busts_only.abs().mean()
            skill_score_moist_busts_only = (1.0 - tdai_mae_moist_busts_only / nbm_mae_moist_busts_only) * 100.0

            # --- 5d. MODEL SQUARED PEARSON CORRELATION COEFFICIENT (R2) ---

            # *Main* R2 (includes zeroed out dry busts)
            if np.std(y_test_dry_0) > 0 and np.std(y_pred) > 0:
                r2_dry_0 = np.corrcoef(y_test_dry_0, y_pred)[0, 1] ** 2
            else:
                r2_dry_0 = float('nan')

            # *Moist busts only* R2 (evaluates TdAI only on days the NBM-ASOS error > 0)
            if y_test_moist_busts_only.std() > 0 and y_pred_moist_busts_only.std() > 0:
                r2_moist_busts_only = np.corrcoef(y_test_moist_busts_only, y_pred_moist_busts_only)[0, 1] ** 2
            else:
                r2_moist_busts_only = float('nan')

            # Print out metrics
            print(f"   {HOLDOUT_YEAR} Validation Results (n={len(X_test_filtered)}, moist-bust n={len(y_test_moist_busts_only)}):")
            print(f"      • Skill Score - Zeroed Dry Busts     : {skill_score:.2f}")
            print(f"      • Skill Score - NBM Moist Busts Only : {skill_score_moist_busts_only:.2f}")
            print(f"      • R² - Zeroed Dry Busts               : {r2_dry_0:.2f}")
            print(f"      • R² - NBM Moist Busts Only           : {r2_moist_busts_only:.2f}")

        # --- 6. EXPORT INDIVIDUAL MODEL WEIGHTS AND FEATURE SCHEMAS ---
        # Set the save path for the trained models for this station
        station_models_output_path = os.path.join(models_output_path, f"{station}/")
        os.makedirs(station_models_output_path, exist_ok=True)

        model_save_path = os.path.join(station_models_output_path, f"tdai_deterministic_model_{station}_{c_name}.joblib")
        schema_save_path = os.path.join(station_models_output_path, f"deterministic_model_feature_schema_{station}_{c_name}.joblib")

        joblib.dump(gb_model, model_save_path)
        joblib.dump(X_train.columns.tolist(), schema_save_path)

        print(f"   💾 Model .joblib Files Saved")

print("✨ ALL MODEL TRAINING COMPLETE!")

