TdAI is an NBM postprocessing model that uses a Gradient Boosted Decision Tree (GBDT) algorithm to bias correct NBM dewpoint forecasts, particularly on dry, well-mixed days. Its output is designed to be used by NWS forecasters as a fire weather situational tool, giving them confidence to decrease the forecast dewpoint, and thus RH, well below guidance. The overall goal of TdAI is to improve the quality of the fire weather products and services the NWS provides to its fire partners.

---------------------

TdAI MODEL ARCHITECTURE:
1) We train only on the 21z data. This maximizes performance because we are only focusing on the time of maximum mixing and lowest RH
2) No re-distribution of the dataset is done (no random removal of quiet days and no sky/temp/RH/LPW filtering of the dataset)
3) Training data from 2020-2024 & 2026 is used. 2025 is the validation dataset
4) A smaller number of trees and smaller max depth of trees is used to account for the smaller dataset (resulting from only using 21z data). This prevents overfitting
5) An operational validation framework has been added to assess model performance if TdAI was run only under set weather conditions (i.e. when a bust was most likely)

    FEATURE VARIABLES:
        NBM Temperature (C)
        NBM Dewpoint (C)
        NBM RH (%)
        NBM Sky (%)
        NBM Mixing Height (100s of ft AGL)
        NBM Wind Speed (kts)
        NBM Wind Direction (deg)
        HRRR RH at all levels (%)
        HRRR PWAT
        HRRR 1000mb-850mb Lapse Rate (C/km)
        HRRR 850mb-500mb Lapse Rate (C/km)
        Time of year

    OUTCOME VARIABLE:
        Td error from the 12z NBM forecast

    WEIGHTING SCHEME:
        Td error 3-4 F: Weight of 2
        Td error >= 5 F: Weight of 5

----------------------

THE REPOSITORY CONSISTS OF TWO MAIN PARTS:

1) A Python script that ingests 12z HRRR and 13z NBM data at KCAR which is used to run the TdAI model
2) A web visualization dashboard for TdAI forecast output

Google Cloud Scheduler was used to set up a cron that runs TdAI at 14:45z every day

------------------------

STILL TO DO:

1) Set up an operational filter that runs the model only on fire weather days (i.e. T > 50 F, RH < 50 %, Sky < 60%)
2) Improve the dashboard to include verification statistics
3) Tweak feature variables - Remove Td and add HRRR soil moisture
4) Add a probabilistic distribution to TdAI using an ensemble of quantile mapping runs (i.e. 10th, 25th, 50th, 75th, 90th
5) Train the model on more ASOS sites
