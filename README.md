<div style="display: flex; gap: 10px;">
  <img height="158" alt="TdAI Deterministic Logo" src="https://github.com/user-attachments/assets/81d2f099-5b12-4420-b163-02968b1e163a" />
  <img height="158" alt="TdAI Probabilistic Logo" src="https://github.com/user-attachments/assets/bc11d753-bfe7-4347-83a1-4e3224560760" />
</div>

TdAI is an NBM postprocessing model that uses a Gradient Boosted Decision Tree (GBDT) algorithm to bias correct NBM dewpoint forecasts, particularly on dry, well-mixed days. Its output is designed to be used by NWS forecasters as a fire weather situational tool, giving them confidence to decrease the forecast dewpoint, and thus RH, well below guidance. The overall goal of TdAI is to improve the quality of the fire weather products and services the NWS provides to its fire partners.

---------------------

TdAI MODEL ARCHITECTURE:
1) TdAI is trained only on 21z NBM & HRRR data from May 2020 to May 2026. Training only on 21z data maximizes performance because we are only focusing on the time of maximum mixing and lowest RH
2) TdAI runs only when NBM RH <= 60%, T >= 50 degrees, and Cloud Cover <=60%. This is done to prevent the user from seeing TdAI output on non-fire weather days where its prediction likely doesn't hold any significant value given it is trained to perform best on very dry days.

    FEATURE VARIABLES:
        NBM Temperature (C)
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
        Td error from the 13z NBM forecast

    WEIGHTING SCHEME (tells the model to focus more on the largest NBM moist busts):
        Td error 3-4 F: Weight of 2
        Td error >= 5 F: Weight of 5

----------------------

THE REPOSITORY CONSISTS OF THREE MAIN PARTS:

1) A Python script that ingests 00z/12z HRRR and 01z/13z NBM data at KCAR which is used to run the DETERMINISTIC TdAI model twice a day
2) A Python script that ingests 00z/12z HRRR and 01z/13z NBM data at KCAR which is used to run the PROBABILISTIC TdAI model twice a day
3) A web visualization dashboard for TdAI forecast output

Google Cloud Scheduler was used to set up a cron that runs TdAI at 02:45z and 14:45z every day

------------------------

FUTURE STEPS:

1) Add HRRR soil moisture and/or recent precipitation to the training dataset
2) Train the model on 00z HRRR runs so TdAI runs with the 13z and 01z NBM crons
3) Train the model on more ASOS sites (KBHB, KBGR, KGNR, KMLT, K40B)
