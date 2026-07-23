<table border="0">
  <tr>
    <td><img alt="TdAI Deterministic Logo" src="https://github.com/user-attachments/assets/81d2f099-5b12-4420-b163-02968b1e163a" /></td>
    <td><img alt="TdAI Probabilistic Logo" src="https://github.com/user-attachments/assets/bc11d753-bfe7-4347-83a1-4e3224560760" /></td>
  </tr>
</table>

TdAI is an NBM postprocessing model that uses a Gradient Boosted Decision Tree (GBDT) algorithm to bias correct NBM dewpoint forecasts, particularly on dry, well-mixed days. Its output is designed to be used by NWS forecasters as a fire weather situational tool, giving them confidence to decrease the forecast dewpoint, and thus RH, well below guidance. The overall goal of TdAI is to improve the quality of the fire weather products and services the NWS provides to its fire partners.

---------------------

TdAI MODEL ARCHITECTURE:
1) TdAI is trained only on 21z NBM & HRRR sounding data from May 2020 to July 2026. Training only on 21z data maximizes performance because we are only focusing on the time of peak mixing and lowest RH
2) Separate models for both the deterministic and probabilistic versions were trained for each operational cycle of TdAI (run time and forecast hour) to prevent structural bias. These are as follows:
   a) 02:45z TdAI forecast for day 1 (00z HRRR at f21 and 01z NBM at f20)
   b) 02:45z TdAI forecast for day 2 (00z HRRR at f45 and 01z NBM at f44)
   c) 14:45z TdAI forecast for day 1 (12z HRRR at f09 and 13z NBM at f08)
   d) 14:45z TdAI forecast for day 2 (12z HRRR at f33 and 13z NBM at f32)
5) TdAI runs only when NBM RH <= 60%, T >= 50 degrees, and Cloud Cover <=60%. This is done to prevent the user from seeing TdAI output on non-fire weather days where its prediction likely doesn't hold any significance given it is trained to perform best on very dry days.

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
        Td error from the 13z NBM forecast

    WEIGHTING SCHEME (tells the model to focus more on the largest NBM moist busts):
        Td error 3-4 F: Weight of 2
        Td error >= 5 F: Weight of 5

    TdAI VERSIONS:
        TdAI Deterministic: v3.0
        TdAI Probabilistic: v2.0

----------------------

THE REPOSITORY CONSISTS OF THREE MAIN PARTS:

1) A Python script that ingests 00z/12z HRRR and 01z/13z NBM data at KCAR which is used to run the DETERMINISTIC TdAI models twice a day
2) A Python script that ingests 00z/12z HRRR and 01z/13z NBM data at KCAR which is used to run the PROBABILISTIC TdAI models twice a day
3) A web visualization dashboard for TdAI forecast output

Google Cloud Scheduler was used to set up a cron that runs TdAI at 02:45z and 14:45z every day

------------------------

FUTURE ADDITIONS TO TdAI:

1) Add HRRR soil moisture and/or recent precipitation to the training dataset
2) Add 15-21z average, rather than just 21z, NBM Sky as a predictor
3) Train the model on more ASOS sites (KBHB, KBGR, KGNR, KMLT, K40B, etc.)
