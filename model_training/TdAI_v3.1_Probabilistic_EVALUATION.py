"""
TdAI v3.1 Probabilistic - POST-TRAINING EVALUATION

Loads the already-trained per-station-per-cycle quantile model bundles
produced by TdAI_v3.1_Probabilistic_TRAINING.py (no retraining happens here)
and generates:
  - Confidence interval band plots, one panel per station (all cycles pooled)
  - PIT histogram and aggregate CRPS-style score, one panel per station (all cycles pooled)

Uses the same operational-gate + moist-bust-only methodology as
TdAI_v3.1_Deterministic_EVALUATION.py.
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

################################## INPUTS ####################################
base_path = "/home/sean834/TdAI/"
training_dataset_path = os.path.join(base_path, "model_training/training_dataset/")

# This script scores models against HOLDOUT_YEAR as a strictly-unseen test
# set, so it must ONLY ever load from trained_models_EVALUATION/ - the folder
# TdAI_v3.1_Probabilistic_TRAINING.py writes to when PRODUCTION_MODE=False
# (HOLDOUT_YEAR excluded from training). The live trained_models/ folder holds
# PRODUCTION_MODE=True models trained on ALL years including HOLDOUT_YEAR, so
# evaluating against those would silently score a model on data it already
# saw during training.
models_output_path = os.path.join(base_path, "model_training/trained_models_EVALUATION/")

STATIONS = ['FVE', 'CAR', 'HUL', 'MLT', 'GNR', 'BGR']
CYCLE_NAMES = ['03z_Day1', '03z_Day2', '15z_Day1', '15z_Day2']

HOLDOUT_YEAR = 2025

do_scatter_plot = True
do_ci_band_plot = True
do_gated_coverage_check = True
do_ungated_coverage_check = True
do_pit_and_crps_check = True

N_CI_BAND_BINS = 10
###############################################################################


def load_gated_moist_bust_probabilistic(station, c_name):
    """Loads a station/cycle's compiled dataset + trained quantile model
    bundle, isolates HOLDOUT_YEAR, and returns (X_moist, y_moist, models_dict)
    after applying the operational gate + moist-bust-only filter. Returns
    None if unavailable."""
    dataset_full_path = os.path.join(training_dataset_path, f"TdAI_Training_Data_{station}_{c_name}.csv")
    model_full_path = os.path.join(models_output_path, station, f"tdai_probabilistic_model_{station}_{c_name}.joblib")
    schema_full_path = os.path.join(models_output_path, station, f"probabilistic_model_feature_schema_{station}_{c_name}.joblib")

    if not (os.path.exists(dataset_full_path) and os.path.exists(model_full_path)):
        print(f"⚠️ Missing dataset or model for K{station} {c_name}. Skipping.")
        return None

    df = pd.read_csv(dataset_full_path)
    if 'valid_time' in df.columns:
        df = df.set_index('valid_time')
    df.index = pd.to_datetime(df.index)

    X = df.drop(columns=['Target Error (F)'], errors='ignore')
    y = df['Target Error (F)']
    X_test, y_test = X[X.index.year == HOLDOUT_YEAR], y[X.index.year == HOLDOUT_YEAR]

    if X_test.empty:
        print(f"⚠️ No {HOLDOUT_YEAR} validation samples for K{station} {c_name}. Skipping.")
        return None

    operational_gate = (
        (X_test['NBM RH (%)'] <= 60.0) &
        (X_test['NBM Temperature (F)'] >= 50.0) &
        (X_test['NBM Cloud Cover (%)'] <= 60.0)
    )
    X_gated, y_gated = X_test[operational_gate], y_test[operational_gate]

    is_moist_bust = (y_gated > 0.0).values
    X_moist, y_moist = X_gated[is_moist_bust], y_gated[is_moist_bust]

    if X_moist.empty:
        print(f"⚠️ Zero moist-bust + gated {HOLDOUT_YEAR} samples for K{station} {c_name}. Skipping.")
        return None

    models_dict = joblib.load(model_full_path)

    if os.path.exists(schema_full_path):
        trained_features = joblib.load(schema_full_path)
        X_moist = X_moist[trained_features]

    return X_moist, y_moist, models_dict


def load_gated_probabilistic(station, c_name):
    """Same as load_gated_moist_bust_probabilistic but applies ONLY the
    operational gate - no moist-bust-only filter. Used to check calibration
    against the population the quantile models were actually trained on,
    isolating whether coverage patterns seen on the moist-bust-only subset
    are real miscalibration vs. a selection artifact from truncating to
    NBM error > 0. Returns None if unavailable."""
    dataset_full_path = os.path.join(training_dataset_path, f"TdAI_Training_Data_{station}_{c_name}.csv")
    model_full_path = os.path.join(models_output_path, station, f"tdai_probabilistic_model_{station}_{c_name}.joblib")
    schema_full_path = os.path.join(models_output_path, station, f"probabilistic_model_feature_schema_{station}_{c_name}.joblib")

    if not (os.path.exists(dataset_full_path) and os.path.exists(model_full_path)):
        print(f"⚠️ Missing dataset or model for K{station} {c_name}. Skipping.")
        return None

    df = pd.read_csv(dataset_full_path)
    if 'valid_time' in df.columns:
        df = df.set_index('valid_time')
    df.index = pd.to_datetime(df.index)

    X = df.drop(columns=['Target Error (F)'], errors='ignore')
    y = df['Target Error (F)']
    X_test, y_test = X[X.index.year == HOLDOUT_YEAR], y[X.index.year == HOLDOUT_YEAR]

    if X_test.empty:
        print(f"⚠️ No {HOLDOUT_YEAR} validation samples for K{station} {c_name}. Skipping.")
        return None

    operational_gate = (
        (X_test['NBM RH (%)'] <= 60.0) &
        (X_test['NBM Temperature (F)'] >= 50.0) &
        (X_test['NBM Cloud Cover (%)'] <= 60.0)
    )
    X_gated, y_gated = X_test[operational_gate], y_test[operational_gate]

    if X_gated.empty:
        print(f"⚠️ Zero gated {HOLDOUT_YEAR} samples for K{station} {c_name}. Skipping.")
        return None

    models_dict = joblib.load(model_full_path)

    if os.path.exists(schema_full_path):
        trained_features = joblib.load(schema_full_path)
        X_gated = X_gated[trained_features]

    return X_gated, y_gated, models_dict


def load_ungated_probabilistic(station, c_name):
    """Same as load_gated_probabilistic but applies NO filter at all - the
    full HOLDOUT_YEAR test set, matching the population the quantile models
    were actually trained on (training never applies the operational gate
    either). This is the fairest train/test consistency check: does the
    model generalize to a fresh sample of the SAME population it learned
    from. It is a different question from the gated-only check (which asks
    whether the model is calibrated under the specific conditions it's
    actually deployed under) - report both, neither replaces the other.
    Returns None if unavailable."""
    dataset_full_path = os.path.join(training_dataset_path, f"TdAI_Training_Data_{station}_{c_name}.csv")
    model_full_path = os.path.join(models_output_path, station, f"tdai_probabilistic_model_{station}_{c_name}.joblib")
    schema_full_path = os.path.join(models_output_path, station, f"probabilistic_model_feature_schema_{station}_{c_name}.joblib")

    if not (os.path.exists(dataset_full_path) and os.path.exists(model_full_path)):
        print(f"⚠️ Missing dataset or model for K{station} {c_name}. Skipping.")
        return None

    df = pd.read_csv(dataset_full_path)
    if 'valid_time' in df.columns:
        df = df.set_index('valid_time')
    df.index = pd.to_datetime(df.index)

    X = df.drop(columns=['Target Error (F)'], errors='ignore')
    y = df['Target Error (F)']
    X_test, y_test = X[X.index.year == HOLDOUT_YEAR], y[X.index.year == HOLDOUT_YEAR]

    if X_test.empty:
        print(f"⚠️ No {HOLDOUT_YEAR} validation samples for K{station} {c_name}. Skipping.")
        return None

    models_dict = joblib.load(model_full_path)

    if os.path.exists(schema_full_path):
        trained_features = joblib.load(schema_full_path)
        X_test = X_test[trained_features]

    return X_test, y_test, models_dict


####################################################################
#                                                                  #
#                TdAI PROBABILISTIC PERFORMANCE EVALUATION         #
#                        (Median Scatter Plot)                     #
#                                                                  #
####################################################################

if do_scatter_plot:
    scatter_output_path = os.path.join(base_path, f"model_training/{HOLDOUT_YEAR}_evaluation_OFFICIAL/")
    os.makedirs(scatter_output_path, exist_ok=True)

    print("\n" + "=" * 70)
    print(f" GENERATING ACTUAL VS. TdAI MEDIAN-PREDICTED SCATTER PLOTS ({HOLDOUT_YEAR}, All Cycles Pooled)")
    print("=" * 70)

    station_pooled = {}

    for station in STATIONS:
        actual_list, q50_list = [], []

        for c_name in CYCLE_NAMES:
            loaded = load_gated_moist_bust_probabilistic(station, c_name)
            if loaded is None:
                continue
            X_moist, y_moist, models_dict = loaded

            q50_pred = models_dict['q50'].predict(X_moist)

            actual_list.append(y_moist.values)
            q50_list.append(q50_pred)

        if not actual_list:
            print(f"⚠️ No {HOLDOUT_YEAR} moist-bust + gated samples across any cycle for K{station}. Skipping.")
            continue

        station_pooled[station] = (np.concatenate(actual_list), np.concatenate(q50_list))

    if not station_pooled:
        print("⚠️ No stations had usable data for the scatter plot.")
    else:
        # Skill within a single NBM-error magnitude bin (see
        # TdAI_v3.1_Deterministic_EVALUATION.py for why R2 is intentionally
        # excluded per-bin - range restriction mechanically compresses it)
        def bin_skill(actual_vals, predicted_vals, lo, hi):
            mask = (actual_vals >= lo) & (actual_vals < hi)
            if mask.sum() < 2:
                return float('nan')
            a, p = actual_vals[mask], predicted_vals[mask]
            nbm_mae = np.abs(a).mean()
            tdai_mae = mean_absolute_error(a, p)
            return (1.0 - tdai_mae / nbm_mae) * 100.0 if nbm_mae > 0 else float('nan')

        magnitude_bins = [(0, 3, '0-3F'), (3, 5, '3-5F'), (5, 8, '5-8F'), (8, np.inf, '8+F')]

        n_stations = len(station_pooled)
        n_cols = 2
        n_rows = int(np.ceil(n_stations / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5.6 * n_rows), squeeze=False)
        axes_flat = axes.flatten()

        for ax, (station, (actual, q50)) in zip(axes_flat, station_pooled.items()):
            if actual.std() > 0 and q50.std() > 0:
                r_squared = np.corrcoef(actual, q50)[0, 1] ** 2
            else:
                r_squared = float('nan')

            tdai_mae = mean_absolute_error(actual, q50)
            nbm_mae = np.abs(actual).mean()
            skill_score = (1.0 - tdai_mae / nbm_mae) * 100.0 if nbm_mae > 0 else float('nan')

            ax.scatter(actual, q50, s=18, alpha=0.5, color='dodgerblue', edgecolor='none')

            lo = min(actual.min(), q50.min()) - 1
            hi = max(actual.max(), q50.max()) + 1
            ax.plot([lo, hi], [lo, hi], color='black', linestyle='--', linewidth=1.2, label='Perfect Prediction (1:1)')

            if actual.std() > 0:
                slope, intercept = np.polyfit(actual, q50, 1)
                fit_x = np.array([lo, hi])
                ax.plot(fit_x, slope * fit_x + intercept, color='crimson', linewidth=1.4,
                        label=f'Median Best Fit (slope={slope:.2f})')

            ax.axhline(0, color='gray', linewidth=0.8)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)

            ax.set_xlabel('Actual NBM Error (°F)', fontsize=10)
            ax.set_ylabel('TdAI Predicted Error (°F)', fontsize=10)

            bin_strs = []
            for bin_lo, bin_hi, label in magnitude_bins:
                bin_skill_val = bin_skill(actual, q50, bin_lo, bin_hi)
                bin_strs.append(f"{label}: Skill={bin_skill_val:+.0f}%")
            breakdown_block = "   ".join(bin_strs)

            ax.set_title(f'K{station} (n={len(actual)}, R²={r_squared:.2f}, Skill={skill_score:+.1f}%)\n{breakdown_block}',
                         fontsize=9, fontweight='bold')
            ax.grid(linestyle='--', alpha=0.4)
            ax.legend(fontsize=8, loc='upper left')

        for ax in axes_flat[len(station_pooled):]:
            ax.axis('off')

        fig.suptitle(f'TdAI Median (q50) Prediction vs. Actual NBM Error by Station (All Cycles Pooled, {HOLDOUT_YEAR}, Moist Bust Days Only)',
                     fontsize=14, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.99])

        scatter_plot_path = os.path.join(scatter_output_path, "probabilistic_median_scatter_by_station.png")
        fig.savefig(scatter_plot_path, dpi=150)
        plt.close(fig)
        print(f"🖼️  Saved -> {scatter_plot_path}")

    print("\n✨ SCATTER PLOT GENERATION COMPLETE!")
    


####################################################################
#                                                                  #
#                TdAI PROBABILISTIC PERFORMANCE EVALUATION         #
#              (Confidence Interval Band Plot, Per Station)        #
#                                                                  #
####################################################################

if do_ci_band_plot:
    ci_output_path = os.path.join(base_path, f"model_training/{HOLDOUT_YEAR}_evaluation_OFFICIAL/")
    os.makedirs(ci_output_path, exist_ok=True)

    print("\n" + "=" * 70)
    print(f" GENERATING CONFIDENCE INTERVAL BAND PLOTS ({HOLDOUT_YEAR}, All Cycles Pooled)")
    print("=" * 70)

    station_pooled_quantiles = {}

    for station in STATIONS:
        actual_list, q10_list, q25_list, q50_list, q75_list, q90_list = [], [], [], [], [], []

        for c_name in CYCLE_NAMES:
            loaded = load_gated_moist_bust_probabilistic(station, c_name)
            if loaded is None:
                continue
            X_moist, y_moist, models_dict = loaded

            actual_list.append(y_moist.values)
            q10_list.append(models_dict['q10'].predict(X_moist))
            q25_list.append(models_dict['q25'].predict(X_moist))
            q50_list.append(models_dict['q50'].predict(X_moist))
            q75_list.append(models_dict['q75'].predict(X_moist))
            q90_list.append(models_dict['q90'].predict(X_moist))

        if not actual_list:
            print(f"⚠️ No {HOLDOUT_YEAR} moist-bust + gated samples across any cycle for K{station}. Skipping.")
            continue

        station_pooled_quantiles[station] = (
            np.concatenate(actual_list), np.concatenate(q10_list), np.concatenate(q25_list),
            np.concatenate(q50_list), np.concatenate(q75_list), np.concatenate(q90_list),
        )

    if not station_pooled_quantiles:
        print("⚠️ No stations had usable data for the confidence interval band plot.")
    else:
        n_stations = len(station_pooled_quantiles)
        n_cols = 2
        n_rows = int(np.ceil(n_stations / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(9 * n_cols, 4.8 * n_rows), squeeze=False)
        axes_flat = axes.flatten()

        for ax, (station, (actual, q10, q25, q50, q75, q90)) in zip(axes_flat, station_pooled_quantiles.items()):
            # Sort by predicted median so the bands read as a smooth,
            # increasing-magnitude sweep instead of a scrambled time series
            sort_idx = np.argsort(q50)
            actual_s, q10_s, q25_s, q50_s, q75_s, q90_s = (
                actual[sort_idx], q10[sort_idx], q25[sort_idx], q50[sort_idx], q75[sort_idx], q90[sort_idx]
            )
            x = q50_s

            coverage_50 = np.mean((actual >= q25) & (actual <= q75)) * 100.0
            coverage_80 = np.mean((actual >= q10) & (actual <= q90)) * 100.0

            outside_80_ci = (actual_s < q10_s) | (actual_s > q90_s)
            point_colors = np.where(outside_80_ci, 'red', 'black')
            n_above_90th = int(np.sum(actual_s > q90_s))

            ax.fill_between(x, q10_s, q90_s, color='dodgerblue', alpha=0.15,
                             label=f'80% CI (10th-90th, coverage={coverage_80:.0f}%)')
            ax.fill_between(x, q25_s, q75_s, color='dodgerblue', alpha=0.30,
                             label=f'50% CI (25th-75th, coverage={coverage_50:.0f}%)')

            # Reference: if TdAI's median were perfectly honest, the actual
            # outcome would track this diagonal - NOT the predicted value
            # plotted against itself (which is just a tautological straight
            # line and shows nothing about real calibration)
            diag_lo, diag_hi = max(0.0, x.min()), x.max()
            ax.plot([diag_lo, diag_hi], [diag_lo, diag_hi], color='gray', linestyle='--', linewidth=1.2,
                    label='Perfect Calibration (1:1)')

            ax.scatter(x, actual_s, s=10, c=point_colors, alpha=0.6, zorder=5, label='Actual NBM Error (red = outside 80% CI)')

            ax.set_xlim(left=0.0)
            ax.set_xlabel('TdAI Predicted Median (q50) Error (°F)', fontsize=10)
            ax.set_ylabel('Dewpoint Error (°F)', fontsize=10)
            ax.set_title(f'K{station} (n={len(actual)}, missed above 90th percentile={n_above_90th})', fontsize=10, fontweight='bold')
            ax.grid(linestyle='--', alpha=0.4)
            ax.legend(fontsize=7.5, loc='upper left')

        for ax in axes_flat[len(station_pooled_quantiles):]:
            ax.axis('off')

        fig.suptitle(f'TdAI Predicted Confidence Intervals vs. Actual NBM Error by Station (All Cycles Pooled, {HOLDOUT_YEAR}, Moist Bust Days Only)',
                     fontsize=13, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        ci_plot_path = os.path.join(ci_output_path, "probabilistic_ci_bands_by_station.png")
        fig.savefig(ci_plot_path, dpi=150)
        plt.close(fig)
        print(f"🖼️  Saved -> {ci_plot_path}")

    print("\n✨ CONFIDENCE INTERVAL BAND PLOT GENERATION COMPLETE!")



####################################################################
#                                                                  #
#                TdAI PROBABILISTIC PERFORMANCE EVALUATION         #
#         (Calibration Check, Operational Gate Only - No Moist     #
#                  Bust Filter, All Cycles Pooled)                 #
#                                                                  #
####################################################################

if do_gated_coverage_check:
    print("\n" + "=" * 70)
    print(f" GATED-ONLY CALIBRATION CHECK ({HOLDOUT_YEAR}, Operational Gate Only, No Moist-Bust Filter)")
    print(" (Isolates real miscalibration from the selection artifact of restricting to NBM error > 0)")
    print("=" * 70)

    gated_station_results = {}
    all_actual, all_q10, all_q25, all_q75, all_q90 = [], [], [], [], []

    for station in STATIONS:
        actual_list, q10_list, q25_list, q75_list, q90_list = [], [], [], [], []

        for c_name in CYCLE_NAMES:
            loaded = load_gated_probabilistic(station, c_name)
            if loaded is None:
                continue
            X_gated, y_gated, models_dict = loaded

            actual_list.append(y_gated.values)
            q10_list.append(models_dict['q10'].predict(X_gated))
            q25_list.append(models_dict['q25'].predict(X_gated))
            q75_list.append(models_dict['q75'].predict(X_gated))
            q90_list.append(models_dict['q90'].predict(X_gated))

        if not actual_list:
            continue

        actual = np.concatenate(actual_list)
        q10 = np.concatenate(q10_list)
        q25 = np.concatenate(q25_list)
        q75 = np.concatenate(q75_list)
        q90 = np.concatenate(q90_list)

        coverage_50 = np.mean((actual >= q25) & (actual <= q75)) * 100.0
        coverage_80 = np.mean((actual >= q10) & (actual <= q90)) * 100.0
        pct_above_90 = np.mean(actual > q90) * 100.0
        pct_below_10 = np.mean(actual < q10) * 100.0

        gated_station_results[station] = (len(actual), coverage_50, coverage_80, pct_above_90, pct_below_10)

        all_actual.append(actual)
        all_q10.append(q10)
        all_q25.append(q25)
        all_q75.append(q75)
        all_q90.append(q90)

    if not gated_station_results:
        print("⚠️ No stations had usable data for the gated-only coverage check.")
    else:
        print(f"\n{'Station':<8}{'n':>6}{'50% CI cov':>12}{'80% CI cov':>12}{'% above 90th':>15}{'% below 10th':>15}")
        for station, (n, c50, c80, above90, below10) in gated_station_results.items():
            print(f"{station:<8}{n:>6}{c50:>11.1f}%{c80:>11.1f}%{above90:>14.1f}%{below10:>14.1f}%")

        actual_all = np.concatenate(all_actual)
        q10_all = np.concatenate(all_q10)
        q25_all = np.concatenate(all_q25)
        q75_all = np.concatenate(all_q75)
        q90_all = np.concatenate(all_q90)

        coverage_50_all = np.mean((actual_all >= q25_all) & (actual_all <= q75_all)) * 100.0
        coverage_80_all = np.mean((actual_all >= q10_all) & (actual_all <= q90_all)) * 100.0
        above_90_all = np.mean(actual_all > q90_all) * 100.0
        below_10_all = np.mean(actual_all < q10_all) * 100.0

        print("-" * 68)
        print(f"{'ALL':<8}{len(actual_all):>6}{coverage_50_all:>11.1f}%{coverage_80_all:>11.1f}%{above_90_all:>14.1f}%{below_10_all:>14.1f}%")

    print("\n✨ GATED-ONLY CALIBRATION CHECK COMPLETE!")


####################################################################
#                                                                  #
#                TdAI PROBABILISTIC PERFORMANCE EVALUATION         #
#          (Calibration Check, FULLY UNGATED - No Operational      #
#              Gate, No Moist-Bust Filter, All Cycles Pooled)      #
#                                                                  #
####################################################################

if do_ungated_coverage_check:
    print("\n" + "=" * 70)
    print(f" UNGATED CALIBRATION CHECK ({HOLDOUT_YEAR}, No Operational Gate, No Moist-Bust Filter)")
    print(" (Matches the full population the quantile models were actually trained on - the")
    print("  fairest train/test consistency check, distinct from the gated-only operational check above)")
    print("=" * 70)

    ungated_station_results = {}
    all_actual, all_q10, all_q25, all_q75, all_q90 = [], [], [], [], []

    for station in STATIONS:
        actual_list, q10_list, q25_list, q75_list, q90_list = [], [], [], [], []

        for c_name in CYCLE_NAMES:
            loaded = load_ungated_probabilistic(station, c_name)
            if loaded is None:
                continue
            X_test, y_test, models_dict = loaded

            actual_list.append(y_test.values)
            q10_list.append(models_dict['q10'].predict(X_test))
            q25_list.append(models_dict['q25'].predict(X_test))
            q75_list.append(models_dict['q75'].predict(X_test))
            q90_list.append(models_dict['q90'].predict(X_test))

        if not actual_list:
            continue

        actual = np.concatenate(actual_list)
        q10 = np.concatenate(q10_list)
        q25 = np.concatenate(q25_list)
        q75 = np.concatenate(q75_list)
        q90 = np.concatenate(q90_list)

        coverage_50 = np.mean((actual >= q25) & (actual <= q75)) * 100.0
        coverage_80 = np.mean((actual >= q10) & (actual <= q90)) * 100.0
        pct_above_90 = np.mean(actual > q90) * 100.0
        pct_below_10 = np.mean(actual < q10) * 100.0

        ungated_station_results[station] = (len(actual), coverage_50, coverage_80, pct_above_90, pct_below_10)

        all_actual.append(actual)
        all_q10.append(q10)
        all_q25.append(q25)
        all_q75.append(q75)
        all_q90.append(q90)

    if not ungated_station_results:
        print("⚠️ No stations had usable data for the ungated coverage check.")
    else:
        print(f"\n{'Station':<8}{'n':>6}{'50% CI cov':>12}{'80% CI cov':>12}{'% above 90th':>15}{'% below 10th':>15}")
        for station, (n, c50, c80, above90, below10) in ungated_station_results.items():
            print(f"{station:<8}{n:>6}{c50:>11.1f}%{c80:>11.1f}%{above90:>14.1f}%{below10:>14.1f}%")

        actual_all = np.concatenate(all_actual)
        q10_all = np.concatenate(all_q10)
        q25_all = np.concatenate(all_q25)
        q75_all = np.concatenate(all_q75)
        q90_all = np.concatenate(all_q90)

        coverage_50_all = np.mean((actual_all >= q25_all) & (actual_all <= q75_all)) * 100.0
        coverage_80_all = np.mean((actual_all >= q10_all) & (actual_all <= q90_all)) * 100.0
        above_90_all = np.mean(actual_all > q90_all) * 100.0
        below_10_all = np.mean(actual_all < q10_all) * 100.0

        print("-" * 68)
        print(f"{'ALL':<8}{len(actual_all):>6}{coverage_50_all:>11.1f}%{coverage_80_all:>11.1f}%{above_90_all:>14.1f}%{below_10_all:>14.1f}%")

    print("\n✨ UNGATED CALIBRATION CHECK COMPLETE!")


####################################################################
#                                                                  #
#                TdAI PROBABILISTIC PERFORMANCE EVALUATION         #
#     (PIT Histogram + Aggregate CRPS-Style Score, Operational      #
#            Gate Only - No Moist-Bust Filter, Per Station)        #
#                                                                  #
####################################################################
#
# PIT (Probability Integral Transform): for every observation, find where
# the actual value falls within the model's predicted CDF (built by
# interpolating across the 5 trained quantiles, q10-q90, and linearly
# extrapolating beyond q10/q90 using the nearest segment's slope). If the
# model is perfectly calibrated, PIT values are uniformly distributed on
# [0, 1] across the population - this generalizes the 50%/80% coverage
# check above to every quantile level simultaneously, instead of just two.
# A U-shaped histogram means the intervals are too narrow (underdispersive);
# a hump in the middle means they're too wide (overdispersive).
#
# CRPS-style aggregate score: mean pinball loss averaged across all 5
# trained quantiles. This approximates CRPS (the continuous ranked
# probability score), the standard single-number proper scoring rule for a
# full probabilistic forecast - lower is better, and unlike coverage alone
# it penalizes both miscalibration AND excess width (poor sharpness).
#
# Uses the gated-only population (operational gate, no moist-bust filter) -
# the moist-bust-only subset is a biased population (see the gated-only
# coverage check above) and would distort both diagnostics.

if do_pit_and_crps_check:
    pit_output_path = os.path.join(base_path, f"model_training/{HOLDOUT_YEAR}_evaluation_OFFICIAL/")
    os.makedirs(pit_output_path, exist_ok=True)

    print("\n" + "=" * 70)
    print(f" PIT HISTOGRAM + AGGREGATE CRPS-STYLE SCORE ({HOLDOUT_YEAR}, Ungated, No Moist-Bust Filter)")
    print("=" * 70)

    QUANTILE_LEVELS = np.array([0.10, 0.25, 0.50, 0.75, 0.90])

    def pinball_loss(actual_vals, predicted_vals, q):
        diff = actual_vals - predicted_vals
        return np.mean(np.maximum(q * diff, (q - 1.0) * diff))

    def estimate_pit(actual_vals, q10, q25, q50, q75, q90):
        """Vectorized PIT estimate per sample via linear interpolation
        across the 5 trained quantile levels, with linear extrapolation
        beyond q10/q90 using the outermost segment's slope. Clipped to
        [0, 1] since extreme extrapolated values can otherwise overshoot."""
        n = len(actual_vals)
        pit = np.empty(n)
        q_values = np.vstack([q10, q25, q50, q75, q90]).T
        q_values = np.sort(q_values, axis=1)  # guard against occasional quantile crossing between separately-trained models

        for i in range(n):
            y = actual_vals[i]
            qv = q_values[i]
            if y <= qv[0]:
                denom = qv[1] - qv[0]
                slope = (QUANTILE_LEVELS[1] - QUANTILE_LEVELS[0]) / denom if denom > 0 else 0.0
                pit[i] = QUANTILE_LEVELS[0] - slope * (qv[0] - y)
            elif y >= qv[-1]:
                denom = qv[-1] - qv[-2]
                slope = (QUANTILE_LEVELS[-1] - QUANTILE_LEVELS[-2]) / denom if denom > 0 else 0.0
                pit[i] = QUANTILE_LEVELS[-1] + slope * (y - qv[-1])
            else:
                pit[i] = np.interp(y, qv, QUANTILE_LEVELS)

        return np.clip(pit, 0.0, 1.0)

    pit_station_results = {}
    crps_station_results = {}

    for station in STATIONS:
        actual_list, q10_list, q25_list, q50_list, q75_list, q90_list = [], [], [], [], [], []

        for c_name in CYCLE_NAMES:
            loaded = load_ungated_probabilistic(station, c_name)
            if loaded is None:
                continue
            X_test, y_test, models_dict = loaded

            actual_list.append(y_test.values)
            q10_list.append(models_dict['q10'].predict(X_test))
            q25_list.append(models_dict['q25'].predict(X_test))
            q50_list.append(models_dict['q50'].predict(X_test))
            q75_list.append(models_dict['q75'].predict(X_test))
            q90_list.append(models_dict['q90'].predict(X_test))

        if not actual_list:
            print(f"⚠️ No {HOLDOUT_YEAR} samples across any cycle for K{station}. Skipping.")
            continue

        actual = np.concatenate(actual_list)
        q10 = np.concatenate(q10_list)
        q25 = np.concatenate(q25_list)
        q50 = np.concatenate(q50_list)
        q75 = np.concatenate(q75_list)
        q90 = np.concatenate(q90_list)

        pit_station_results[station] = estimate_pit(actual, q10, q25, q50, q75, q90)

        per_quantile_pinball = [
            pinball_loss(actual, pred, q)
            for pred, q in zip([q10, q25, q50, q75, q90], QUANTILE_LEVELS)
        ]
        crps_station_results[station] = (len(actual), np.mean(per_quantile_pinball), per_quantile_pinball)

    if not pit_station_results:
        print("⚠️ No stations had usable data for the PIT histogram / CRPS check.")
    else:
        # --- CRPS-STYLE AGGREGATE SCORE TABLE ---
        print(f"\n{'Station':<8}{'n':>6}{'CRPS-approx':>13}{'q10':>8}{'q25':>8}{'q50':>8}{'q75':>8}{'q90':>8}")
        for station, (n, crps_approx, per_q) in crps_station_results.items():
            per_q_str = "".join(f"{v:>8.3f}" for v in per_q)
            print(f"{station:<8}{n:>6}{crps_approx:>13.3f}{per_q_str}")

        all_pit = np.concatenate(list(pit_station_results.values()))
        all_n = sum(n for n, _, _ in crps_station_results.values())
        all_crps = np.average(
            [crps_approx for _, crps_approx, _ in crps_station_results.values()],
            weights=[n for n, _, _ in crps_station_results.values()],
        )
        print("-" * 61)
        print(f"{'ALL':<8}{all_n:>6}{all_crps:>13.3f}")

        # --- PIT HISTOGRAM PLOT ---
        n_stations = len(pit_station_results)
        n_cols = 2
        n_rows = int(np.ceil(n_stations / n_cols))
        n_bins = 10

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5.3 * n_rows), squeeze=False)
        axes_flat = axes.flatten()

        for ax, (station, pit_vals) in zip(axes_flat, pit_station_results.items()):
            n_obs = len(pit_vals)
            expected_count = n_obs / n_bins

            # Soft background tint so the "which half am I in" read is
            # instant, before even looking at the bars
            ax.axvspan(0.0, 0.5, color='salmon', alpha=0.10, zorder=0)
            ax.axvspan(0.5, 1.0, color='dodgerblue', alpha=0.10, zorder=0)
            ax.axvline(0.5, color='gray', linestyle=':', linewidth=1.0, zorder=1)

            ax.hist(pit_vals, bins=np.linspace(0, 1, n_bins + 1), color='dodgerblue', alpha=0.8,
                    edgecolor='white', zorder=2)
            ax.axhline(expected_count, color='crimson', linestyle='--', linewidth=1.6, zorder=3)
            ax.text(0.99, expected_count, ' Expected if calibrated', ha='right', va='bottom',
                    fontsize=12, color='crimson', fontweight='bold', transform=ax.get_yaxis_transform())

            ax.set_xlim(0, 1)

            # Station label lives inside the axes itself (upper right) instead
            # of an external title/subtitle band - removes the need to reserve
            # any vertical margin above the plot at all.
            ax.text(0.97, 0.95, f'K{station} (n={n_obs})', ha='right', va='top',
                    fontsize=11, fontweight='bold', color='black', transform=ax.transAxes,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.85))

            # Plain-English direction arrows, kept snug against the axes top
            # (small axes-fraction offset, no title above them competing for
            # space) so they don't reintroduce the large white gap.
            ax.annotate('', xy=(0.02, 1.05), xytext=(0.47, 1.05), xycoords='axes fraction',
                        arrowprops=dict(arrowstyle='->', color='firebrick', lw=1.6), annotation_clip=False)
            ax.text(0.245, 1.06, 'Model OVER-predicted',
                    ha='center', va='bottom', fontsize=12, color='firebrick', fontweight='bold',
                    transform=ax.transAxes)

            ax.annotate('', xy=(0.98, 1.05), xytext=(0.53, 1.05), xycoords='axes fraction',
                        arrowprops=dict(arrowstyle='->', color='steelblue', lw=1.6), annotation_clip=False)
            ax.text(0.755, 1.06, 'Model UNDER-predicted',
                    ha='center', va='bottom', fontsize=12, color='steelblue', fontweight='bold',
                    transform=ax.transAxes)

            ax.set_xlabel('PIT Value', fontsize=15)
            ax.set_ylabel('Count', fontsize=15)
            ax.grid(axis='y', linestyle='--', alpha=0.4)

        for ax in axes_flat[len(pit_station_results):]:
            ax.axis('off')

        fig.suptitle(
            f'PIT Histogram by Station ({HOLDOUT_YEAR}, Ungated, All Cycles Pooled)\n'
            f'Flat/even bars = well calibrated   |   Taller on left = model runs too high   |   Taller on right = model runs too low',
            fontsize=12, fontweight='bold'
        )
        fig.subplots_adjust(hspace=0.32, wspace=0.18, top=0.92, bottom=0.05, left=0.06, right=0.98)

        pit_plot_path = os.path.join(pit_output_path, "probabilistic_pit_histogram_by_station.png")
        fig.savefig(pit_plot_path, dpi=150)
        plt.close(fig)
        print(f"🖼️  Saved -> {pit_plot_path}")

    print("\n✨ PIT HISTOGRAM + CRPS CHECK COMPLETE!")
