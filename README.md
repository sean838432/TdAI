<table border="0">
  <tr>
    <td><img alt="TdAI Deterministic Logo" src="https://github.com/user-attachments/assets/81d2f099-5b12-4420-b163-02968b1e163a" /></td>
    <td><img alt="TdAI Probabilistic Logo" src="https://github.com/user-attachments/assets/bc11d753-bfe7-4347-83a1-4e3224560760" /></td>
  </tr>
</table>

TdAI is an NBM postprocessing model that uses a Gradient Boosted Decision Tree (GBDT) machine learning algorithm to bias correct NBM dew point forecasts, particularly on dry, well-mixed days. Its output is designed to be used by forecasters as a fire weather situational tool, giving them confidence to decrease the forecast dew point, and thus RH, well below guidance. The overall goal of TdAI is to improve the quality of the fire weather products and services the National Weather Service provides to its fire partners.

---

## TdAI Model Architecture

1. **Machine Learning Algorithm**: Gradient Boosted Decision Trees
2. **Training Dataset**: TdAI is trained on the most recent 6 years of 21z NBM & HRRR data from March to November using a sliding training window. Training strictly on 21z data maximizes performance by focusing exclusively on peak boundary layer mixing and minimum diurnal RH.
3. **Cycle Specialization**: Separate models for both the deterministic and probabilistic architectures were trained for each operational cycle of TdAI (run time and forecast hour) to prevent structural bias:
   * **02:45z TdAI Day 1**: 00z HRRR at f21 and 01z NBM at f20
   * **02:45z TdAI Day 2**: 00z HRRR at f45 and 01z NBM at f44
   * **14:45z TdAI Day 1**: 12z HRRR at f09 and 13z NBM at f08
   * **14:45z TdAI Day 2**: 12z HRRR at f33 and 13z NBM at f32
4. **Execution Gating Criteria**: TdAI runs only when **NBM RH ≤ 60%**, **T ≥ 50°F**, and **Cloud Cover ≤ 60%**. This prevents forecasters from seeing TdAI output on non-fire weather days where predictions carry little operational significance.

### Feature Variables
* NBM Temperature (°C)
* NBM RH (%)
* NBM 15z-21z Avg Sky (%)
* NBM Mixing Height (100s of ft AGL)
* NBM Wind Speed (kts)
* NBM Wind Direction (deg)
* HRRR PWAT
* HRRR 1000mb–850mb Lapse Rate (°C/km)
* HRRR 850mb–500mb Lapse Rate (°C/km)
* HRRR RH at all levels sfc-500mb (%)
* Time of year

### Outcome Variable
* Dew point ($T_d$) error relative to the 01z/13z NBM forecast

### Weighting Scheme
*(Focuses model optimization on the largest NBM moist busts)*
* **$T_d$ error < 3°F**: Weight of 1
* **$T_d$ error 3–4°F**: Weight of 2
* **$T_d$ error ≥ 5°F**: Weight of 5

---

## Repository Overview

The repository consists of 6 main components:

1. **Deterministic Forecast Pipeline**: ```TdAI_deterministic_operational.py``` ingests 00z/12z HRRR and 01z/13z NBM data at KCAR to run the deterministic TdAI models operationally. ```TdAI_cron.yml``` runs this pipeline automatically twice daily.

2. **Probabilistic Forecast Pipeline**: ```TdAI_probabilistic_operational.py``` ingests 00z/12z HRRR and 01z/13z NBM data at KCAR to run the probabilistic TdAI models operationally. ```TdAI_cron.yml``` runs this pipeline automatically twice daily.

3. **Deterministic Verification Pipeline**: ```TdAI_deterministic_verification.py``` verifies the current day's TdAI forecast and NDFD forecast against 21 UTC ASOS data. ```TdAI_verify.yml``` runs this pipeline automatically twice daily.

4. **Web Visualization Dashboard**: An operational web dashboard displaying live TdAI forecast outputs. See the link in the about section.
 
5. **Automatic TdAI Retraining Pipeline**: ```TdAI_auto_retrain.py``` uses an automated 6 year sliding window retraining algorithm which, once a month, brings the current training dataset up to current, drops data older than 6 years, and retrains the operational deterministic and probabilistic models. ```TdAI_auto_retrain.yml``` runs this pipeline automatically on the first of the month, placing freshly trained models in ```/model_training_STATIC/trained_models```.

6. **Model Training & Evaluation (static framework)**: Under ```/model_training_STATIC```, ```TdAI_Deterministic_TRAINING.py``` and ```TdAI_Deterministic_EVALUATION.py``` are used for the initial training and evaluation of the deterministic TdAI models while ```TdAI_Probabilistic_TRAINING.py``` and ```TdAI_Probabilistic_EVALUATION.py``` are used for the initial training and evaluation of the probabilistic TdAI models. ```TdAI_Training_Dataset_Compilation.py``` compiles all HRRR, NBM, and ASOS data into the training dataset used for training the deterministic and probabilistic models.

> **Automated Execution**: Google Cloud Scheduler triggers a cron job producing TdAI forecasts daily at **02:45z** and **14:45z** and an additional daily cron job at 21:30z verifies TdAI predictions. A monthly cron (on the 1st of each month) extends the training dataset by one month and drops any data exceeding 6 years before retraining the operational models.

---
