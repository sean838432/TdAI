<table border="0">
  <tr>
    <td><img alt="TdAI Deterministic Logo" src="https://github.com/user-attachments/assets/81d2f099-5b12-4420-b163-02968b1e163a" /></td>
    <td><img alt="TdAI Probabilistic Logo" src="https://github.com/user-attachments/assets/bc11d753-bfe7-4347-83a1-4e3224560760" /></td>
  </tr>
</table>

TdAI is an NBM postprocessing model that uses a Gradient Boosted Decision Tree (GBDT) algorithm to bias correct NBM dewpoint forecasts, particularly on dry, well-mixed days. Its output is designed to be used by NWS forecasters as a fire weather situational tool, giving them confidence to decrease the forecast dewpoint, and thus RH, well below guidance. The overall goal of TdAI is to improve the quality of the fire weather products and services the NWS provides to its fire partners.

---

## TdAI Model Architecture

1. **Training Dataset**: TdAI is trained only on 21z NBM & HRRR sounding data from May 2021 to July 2026. Training strictly on 21z data maximizes performance by focusing exclusively on peak boundary layer mixing and minimum diurnal RH.
2. **Cycle Specialization**: Separate models for both deterministic and probabilistic versions were trained for each operational cycle of TdAI (run time and forecast hour) to prevent structural bias:
   * **02:45z TdAI Day 1**: 00z HRRR at f21 and 01z NBM at f20
   * **02:45z TdAI Day 2**: 00z HRRR at f45 and 01z NBM at f44
   * **14:45z TdAI Day 1**: 12z HRRR at f09 and 13z NBM at f08
   * **14:45z TdAI Day 2**: 12z HRRR at f33 and 13z NBM at f32
3. **Execution Gating Criteria**: TdAI runs only when **NBM RH ≤ 60%**, **T ≥ 50°F**, and **Cloud Cover ≤ 60%**. This prevents forecasters from seeing TdAI output on non-fire weather days where predictions carry little operational significance.

### Feature Variables
* NBM Temperature (°C)
* NBM RH (%)
* NBM Sky (%)
* NBM Mixing Height (100s of ft AGL)
* NBM Wind Speed (kts)
* NBM Wind Direction (deg)
* HRRR PWAT
* HRRR 1000mb–850mb Lapse Rate (°C/km)
* HRRR 850mb–500mb Lapse Rate (°C/km)
* HRRR RH at all levels sfc-500mb (%)
* Time of year

### Outcome Variable
* Dewpoint ($T_d$) error relative to the 13z NBM forecast

### Weighting Scheme
*(Focuses model optimization on the largest NBM moist busts)*
* **$T_d$ error 3–4°F**: Weight of 2
* **$T_d$ error ≥ 5°F**: Weight of 5

### TdAI Version History
* **TdAI Deterministic**: v3.0
* **TdAI Probabilistic**: v2.0

---

## Repository Overview

The repository consists of three main components:

1. **Deterministic Pipeline**: A Python script ingesting 00z/12z HRRR and 01z/13z NBM data at KCAR used to run the deterministic TdAI models twice daily.
2. **Probabilistic Pipeline**: A Python script ingesting 00z/12z HRRR and 01z/13z NBM data at KCAR used to run the probabilistic TdAI models twice daily.
3. **Web Visualization Dashboard**: An operational web dashboard displaying live TdAI forecast outputs.

> **Automated Execution**: Google Cloud Scheduler triggers a cron job executing TdAI daily at **02:45z** and **14:45z**.

---

## Future Additions

* Add HRRR soil moisture and/or recent precipitation to the training dataset feature matrix
* Add 15–21z average (rather than strictly 21z point value) NBM Sky Cover as a predictor
* Expand model training across additional WFO Caribou ASOS sites (*KBHB, KBGR, KGNR, KMLT, K40B, etc.*)
