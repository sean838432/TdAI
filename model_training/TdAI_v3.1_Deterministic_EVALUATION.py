"""
TdAI v3.1 Deterministic - POST-TRAINING EVALUATION

Loads the already-trained per-station-per-cycle models and compiled datasets
produced by TdAI_v3.1_Deterministic_TRAINING.py (no retraining happens here)
and generates:
  - Top 25 NBM moist bust day bar charts (per station/cycle)
  - Actual vs. TdAI-predicted scatter plots (per station, all cycles pooled)
  - SHAP (TreeExplainer) feature importance (per station, averaged across all cycles)

All three sections use the same operational-gate + moist-bust-only
methodology as TdAI_v3.1_Deterministic_TRAINING.py's Section 5.
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import mean_absolute_error

################################## INPUTS ####################################
base_path = "/home/sean834/TdAI/"
training_dataset_path = os.path.join(base_path, "model_training/training_dataset/")

# This script scores models against HOLDOUT_YEAR as a strictly-unseen test
# set, so it must ONLY ever load from trained_models_EVALUATION/ - the folder
# TdAI_v3.1_Deterministic_TRAINING.py writes to when PRODUCTION_MODE=False
# (HOLDOUT_YEAR excluded from training). The live trained_models/ folder holds
# PRODUCTION_MODE=True models trained on ALL years including HOLDOUT_YEAR, so
# evaluating against those would silently score a model on data it already
# saw during training.
models_output_path = os.path.join(base_path, "model_training/trained_models_EVALUATION/")

STATIONS = ['FVE', 'CAR', 'HUL', 'MLT', 'GNR', 'BGR']
CYCLE_NAMES = ['03z_Day1', '03z_Day2', '15z_Day1', '15z_Day2']

# Must match HOLDOUT_YEAR in TdAI_v3.1_Deterministic_TRAINING.py - the models
# were trained with this year excluded, so it's the only year safe to score here.
HOLDOUT_YEAR = 2025

do_top25 = False
do_scatter_plot = True
do_feature_importance = False

N_TOP_FEATURES = 10
###############################################################################


def load_gated_moist_bust(station, c_name):
    """Loads a station/cycle's compiled dataset + trained model, isolates
    HOLDOUT_YEAR, and returns (X_moist, y_moist, gb_model) after applying the
    operational gate + moist-bust-only filter. Returns None if unavailable."""
    dataset_full_path = os.path.join(training_dataset_path, f"TdAI_Training_Data_{station}_{c_name}.csv")
    model_full_path = os.path.join(models_output_path, station, f"tdai_deterministic_model_{station}_{c_name}.joblib")

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

    gb_model = joblib.load(model_full_path)
    return X_moist, y_moist, gb_model


####################################################################
#                                                                  #
#                     TdAI PERFORMANCE EVALUATION                  #
#                      (TOP 25 NBM Moist Busts)                    #
#                                                                  #
####################################################################

if do_top25:
    top25_output_path = os.path.join(base_path, f"model_training/{HOLDOUT_YEAR}_evaluation_OFFICIAL/top25_plots/")
    os.makedirs(top25_output_path, exist_ok=True)

    print("\n" + "=" * 70)
    print(f" TOP 25 NBM MOIST BUST DAY EVALUATION ({HOLDOUT_YEAR}, Moist Bust + Operational Gate)")
    print("=" * 70)

    for station in STATIONS:
        for c_name in CYCLE_NAMES:
            loaded = load_gated_moist_bust(station, c_name)
            if loaded is None:
                continue
            X_moist, y_moist, gb_model = loaded

            # Make the model predictions on the moist-bust + gated dataset
            y_pred = gb_model.predict(X_moist)
            y_pred = np.where(y_pred < 0, 0.0, y_pred)  # Make sure TdAI never suggests raising Td

            # --- ISOLATE THE TOP 25 DAILY MOIST BUSTS ---
            results_df = pd.DataFrame({
                'valid_time': X_moist.index,
                'date': pd.to_datetime(X_moist.index).date,
                'nbm_error_f': y_moist.values,
                'tdai_pred_f': y_pred,
            })

            daily_summary = results_df.groupby('date').agg({
                'nbm_error_f': lambda x: x.iloc[np.abs(x).argmax()],
                'tdai_pred_f': lambda x: x.iloc[np.abs(x).argmax()],
            }).reset_index()

            num_busts = min(25, len(daily_summary))
            top_busts = daily_summary.sort_values(by='nbm_error_f', ascending=False).head(num_busts)
            top_busts = top_busts.sort_values(by='date')

            # --- COMPACT PER-STATION-PER-CYCLE PERFORMANCE LINE ---
            # Scored on the TOP 25 bust days shown in the plot below, not the
            # full moist-bust population - so this number matches what the
            # chart is actually showing.
            tdai_mae_top25 = mean_absolute_error(top_busts['nbm_error_f'], top_busts['tdai_pred_f'])
            nbm_mae_top25 = top_busts['nbm_error_f'].abs().mean()
            skill_top25 = (1.0 - tdai_mae_top25 / nbm_mae_top25) * 100.0 if nbm_mae_top25 > 0 else float('nan')
            if top_busts['nbm_error_f'].std() > 0 and top_busts['tdai_pred_f'].std() > 0:
                r2_top25 = np.corrcoef(top_busts['nbm_error_f'], top_busts['tdai_pred_f'])[0, 1] ** 2
            else:
                r2_top25 = float('nan')

            # % of the actual bust magnitude TdAI's prediction captures, on
            # average - the skill score alone can look decent even when TdAI
            # is systematically undershooting the biggest busts, since any
            # reduction vs. the raw NBM error counts as "skill."
            mean_actual_top25 = top_busts['nbm_error_f'].mean()
            mean_pred_top25 = top_busts['tdai_pred_f'].mean()
            pct_captured_top25 = (mean_pred_top25 / mean_actual_top25) * 100.0 if mean_actual_top25 > 0 else float('nan')

            # print(f"K{station:<4} {c_name:<9} Top{num_busts:<2d}  NBM MAE={nbm_mae_top25:5.2f}F  "
            #       f"TdAI MAE={tdai_mae_top25:5.2f}F  Skill={skill_top25:+6.1f}%  R²={r2_top25:.2f}  "
            #       f"Magnitude Captured={pct_captured_top25:5.1f}%")

            # --- PLOT ---
            plt.figure(figsize=(15, 6))
            x = np.arange(len(top_busts))
            width = 0.35

            plt.bar(x - width / 2, top_busts['nbm_error_f'], width, label='Actual NBM Error', color='salmon', alpha=0.85)
            plt.bar(x + width / 2, top_busts['tdai_pred_f'], width, label='TdAI Predicted Error', color='dodgerblue', alpha=0.85)
            plt.axhline(0, color='black', linewidth=1.2)
            plt.xticks(x, top_busts['date'], rotation=45, ha='right', fontsize=9)
            plt.ylabel('Dewpoint Error Magnitude (°F)', fontsize=11)
            plt.xlabel('Bust Date (Chronological Order)', fontsize=11)
            plt.title(f'Actual NBM Error vs. TdAI Predicted Error — K{station} {c_name} (Top {num_busts} Moist Busts, {HOLDOUT_YEAR})',
                      fontsize=13, fontweight='bold')
            plt.legend(fontsize=10, loc='upper left')
            plt.grid(axis='y', linestyle='--', alpha=0.5)

            for i, row in enumerate(top_busts.itertuples()):
                same_direction = np.sign(row.nbm_error_f) == np.sign(row.tdai_pred_f)
                if same_direction:
                    y_pos = max(row.nbm_error_f, row.tdai_pred_f) + 0.4
                    plt.text(i, y_pos, '✓', ha='center', color='forestgreen', fontweight='bold', fontsize=10)
                else:
                    y_pos = row.tdai_pred_f - 1.2
                    plt.text(i, y_pos, '✗', ha='center', color='crimson', fontweight='bold', fontsize=10)

            plt.tight_layout()
            plot_path = os.path.join(top25_output_path, f"{station}_{c_name}_top25_bust_evaluation.png")
            plt.savefig(plot_path, dpi=150)
            plt.close()

    print("\n✨ TOP 25 BUST DAY EVALUATION COMPLETE!")


####################################################################
#                                                                  #
#                     TdAI PERFORMANCE EVALUATION                  #
#                           (Scatter Plots)                        #
#                                                                  #
####################################################################

if do_scatter_plot:
    scatter_output_path = os.path.join(base_path, f"model_training/{HOLDOUT_YEAR}_evaluation_OFFICIAL/")
    os.makedirs(scatter_output_path, exist_ok=True)

    print("\n" + "=" * 70)
    print(f" GENERATING ACTUAL VS. TdAI-PREDICTED SCATTER PLOTS ({HOLDOUT_YEAR}, All Cycles Pooled)")
    print("=" * 70)

    station_pooled = {}

    for station in STATIONS:
        actual_list, pred_list = [], []

        for c_name in CYCLE_NAMES:
            loaded = load_gated_moist_bust(station, c_name)
            if loaded is None:
                continue
            X_moist, y_moist, gb_model = loaded

            y_pred = gb_model.predict(X_moist)
            y_pred = np.where(y_pred < 0, 0.0, y_pred)  # TdAI never suggests raising Td

            actual_list.append(y_moist.values)
            pred_list.append(y_pred)

        if not actual_list:
            print(f"⚠️ No {HOLDOUT_YEAR} moist-bust + gated samples across any cycle for K{station}. Skipping.")
            continue

        station_pooled[station] = (np.concatenate(actual_list), np.concatenate(pred_list))

    if not station_pooled:
        print("⚠️ No stations had usable data for the scatter plot.")
    else:
        # Skill within a single NBM-error magnitude bin (contiguous, no gaps:
        # 0-3F, 3-5F, 5-8F, 8F+) - the aggregate skill is an average across
        # all bust sizes and can hide much weaker performance on the biggest,
        # most operationally significant busts. R2 is deliberately NOT
        # computed per bin: restricting the range of actual values (as any
        # magnitude bin does by construction) mechanically compresses
        # variance and drives correlation-based R2 toward zero regardless of
        # how good the predictions actually are ("range restriction") - it
        # produced misleading results like R2=0.00 alongside Skill=+63% in
        # the same bin. R2 stays meaningful only on the full, unrestricted
        # range (the main title line above).
        def bin_skill(actual, predicted, lo, hi):
            mask = (actual >= lo) & (actual < hi)
            if mask.sum() < 2:
                return float('nan')
            a, p = actual[mask], predicted[mask]
            nbm_mae = np.abs(a).mean()
            tdai_mae = mean_absolute_error(a, p)
            return (1.0 - tdai_mae / nbm_mae) * 100.0 if nbm_mae > 0 else float('nan')

        magnitude_bins = [(0, 3, '0-3F'), (3, 5, '3-5F'), (5, 8, '5-8F'), (8, np.inf, '8+F')]

        n_stations = len(station_pooled)
        n_cols = 2
        n_rows = int(np.ceil(n_stations / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5.6 * n_rows), squeeze=False)
        axes_flat = axes.flatten()

        for ax, (station, (actual, predicted)) in zip(axes_flat, station_pooled.items()):
            if actual.std() > 0 and predicted.std() > 0:
                r_squared = np.corrcoef(actual, predicted)[0, 1] ** 2
            else:
                r_squared = float('nan')

            tdai_mae = mean_absolute_error(actual, predicted)
            nbm_mae = np.abs(actual).mean()
            skill_score = (1.0 - tdai_mae / nbm_mae) * 100.0 if nbm_mae > 0 else float('nan')

            ax.scatter(actual, predicted, s=18, alpha=0.5, color='dodgerblue', edgecolor='none')

            lo = min(actual.min(), predicted.min()) - 1
            hi = max(actual.max(), predicted.max()) + 1
            ax.plot([lo, hi], [lo, hi], color='black', linestyle='--', linewidth=1.2, label='Perfect Prediction (1:1)')

            # Least-squares line of best fit through the actual points
            if actual.std() > 0:
                slope, intercept = np.polyfit(actual, predicted, 1)
                fit_x = np.array([lo, hi])
                ax.plot(fit_x, slope * fit_x + intercept, color='crimson', linewidth=1.4,
                        label=f'Best Fit (slope={slope:.2f})')

            ax.axhline(0, color='gray', linewidth=0.8)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)

            ax.set_xlabel('Actual NBM Error (°F)', fontsize=12)
            ax.set_ylabel('Model Predicted Error (°F)', fontsize=12)

            bin_strs = []
            for bin_lo, bin_hi, label in magnitude_bins:
                bin_skill_val = bin_skill(actual, predicted, bin_lo, bin_hi)
                bin_strs.append(f"{label}: Skill={bin_skill_val:+.0f}%")
            breakdown_block = "   ".join(bin_strs)

            ax.set_title(f'K{station} (n={len(actual)}, R²={r_squared:.2f}, Skill={skill_score:+.1f}%)', #\n{breakdown_block}',
                         fontsize=15, fontweight='bold')
            ax.grid(linestyle='--', alpha=0.4)
            ax.legend(fontsize=12, loc='upper left')

        for ax in axes_flat[len(station_pooled):]:
            ax.axis('off')

        fig.suptitle(f'Actual vs. TdAI-Predicted Error by Station (All Cycles Pooled, {HOLDOUT_YEAR}, Moist Bust Days Only)',
                     fontsize=15, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.99])

        scatter_plot_path = os.path.join(scatter_output_path, "residual_scatter_by_station.png")
        fig.savefig(scatter_plot_path, dpi=150)
        plt.close(fig)
        print(f"🖼️  Saved -> {scatter_plot_path}")

    print("\n✨ SCATTER PLOT GENERATION COMPLETE!")


####################################################################
#                                                                  #
#                     TdAI PERFORMANCE EVALUATION                  #
#              (Permutation Feature Importance, Avg. Cycles)       #
#                                                                  #
####################################################################

if do_feature_importance:
    importance_output_path = os.path.join(base_path, f"model_training/{HOLDOUT_YEAR}_evaluation_OFFICIAL/")
    os.makedirs(importance_output_path, exist_ok=True)

    print("\n" + "=" * 70)
    print(f" SHAP FEATURE IMPORTANCE ({HOLDOUT_YEAR}, Averaged Across Cycles)")
    print("=" * 70)

    importance_records = []

    for station in STATIONS:
        for c_name in CYCLE_NAMES:
            loaded = load_gated_moist_bust(station, c_name)
            if loaded is None:
                continue
            X_moist, y_moist, gb_model = loaded

            # TreeExplainer gives exact, per-prediction additive attributions
            # for tree ensembles - unlike permutation importance, it isn't
            # thrown off by interacting/routing features (e.g. season), and
            # it's deterministic (no repeats/seed needed).
            explainer = shap.TreeExplainer(gb_model)
            shap_values = explainer.shap_values(X_moist)

            n_rows = len(X_moist)
            abs_shap = np.abs(shap_values)
            imp_mean = abs_shap.mean(axis=0)
            # Standard error of the mean across rows - the per-cycle
            # uncertainty estimate, analogous to permutation importance's
            # across-repeat std but here driven by the row population itself.
            imp_std = abs_shap.std(axis=0, ddof=1) / np.sqrt(n_rows) if n_rows > 1 else np.zeros(abs_shap.shape[1])

            for feature, m, s in zip(X_moist.columns, imp_mean, imp_std):
                importance_records.append({
                    'station': station, 'cycle': c_name, 'feature': feature,
                    'importance_mean_f': m, 'importance_std_f': s,
                })

    if not importance_records:
        print("⚠️ No feature importance results were generated (missing models/datasets).")
    else:
        importance_df = pd.DataFrame(importance_records)

        # Define a function to compute the pooled standard deviation across cycles
        def pooled_std(stds):
            return np.sqrt(np.sum(np.square(stds))) / len(stds)

        avg_importance = importance_df.groupby(['station', 'feature']).agg(
            importance_mean_f=('importance_mean_f', 'mean'),
            importance_std_f=('importance_std_f', pooled_std),
        ).reset_index()

        # One plot per station, each showing that station's own top N
        # features (ranked by its own average importance, not a shared
        # cross-station ranking), with error bars showing the pooled
        # across-cycle uncertainty.
        for station in avg_importance['station'].unique():
            station_data = avg_importance[avg_importance['station'] == station].sort_values(
                'importance_mean_f', ascending=False
            ).head(N_TOP_FEATURES)

            plt.figure(figsize=(10, 7))
            y = np.arange(len(station_data))

            plt.barh(y, station_data['importance_mean_f'], xerr=station_data['importance_std_f'],
                     color='dodgerblue', capsize=3, error_kw={'ecolor': 'black', 'elinewidth': 1, 'alpha': 0.7})
            plt.yticks(y, station_data['feature'], fontsize=9)
            plt.gca().invert_yaxis()
            plt.axvline(0, color='gray', linewidth=0.8)
            plt.xlabel('Mean |SHAP value| (°F, averaged across cycles ± pooled std)', fontsize=11)
            plt.title(f'K{station} SHAP Feature Importance — Averaged Across All Cycles ({HOLDOUT_YEAR})', fontsize=14, fontweight='bold')
            plt.grid(axis='x', linestyle='--', alpha=0.5)
            plt.tight_layout()

            importance_plot_path = os.path.join(importance_output_path, f"shap_importance_{station}_avg_across_cycles.png")
            plt.savefig(importance_plot_path, dpi=150)
            plt.close()
            print(f"🖼️  Saved -> {importance_plot_path}")

    print("\n FEATURE IMPORTANCE EVALUATION COMPLETE!")
