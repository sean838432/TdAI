# How to Train TdAI on New ASOS Sites #
The following is a comprehensive guide for training TdAI on your own ASOS sites. Before going into detail, here are a few general guidelines:

1. I recommend choosing inland ASOS sites to train on. Those too close to the ocean will have a heavy marine influence that will result in less high end moist NBM errors attributed to under mixing which will decrease model performance. If in doubt, compare the percentage of samples at the ASOS with NBM error >= 5F with those further inland and make sure it isn't too drastic a difference. 
2. The deterministic and probabilistic models should be trained on the 6 most recent years of data to ensure stability
3. I recommend not including dates well outside your fire season (e.g. winter) in the training dataset. We want to make sure the models are primarily learning the typical fire weather patterns favorable for NBM Td to be too moist
4. The model trains to correct NBM error at 21 UTC as this is the approximate time of the driest conditions, but this can be changed based on your time zone and local effects.


## Step 1: Data Download

TdAI requires NBM, ASOS, and HRRR data at the desired ASOS location you want to train the model on. Navigate to /data_download and download data to your local machine using the following scripts:
```
TdAI/
├── data_download/
    ├── ASOS_download.py
    ├── NBM_download.py
    ├── hrrr_soundings_download.py
    ├── hrrr_vvel_soilw_mslma_hgt_download.py
    └── sounding_compilation_parquet.py
```

**ASOS_download.py**: Python script for downloading ASOS data. It will output a csv file combining all ASOS Td observations within a specified date range 

**NBM_download.py**: Python script for downloading NBM T, Td, sky, wind speed, wind direction, and mixing height. It will output a csv file combining all NBM day 1 and day 2 forecasts at 15, 18, and 21 UTC for a specified init time (should be 01z and 13z to align with the NBM forecasters edit)

**hrrr_soundings_download.py**: Python script for downloading HRRR soundings at the nearest grid point to the ASOS site for 4 cycles (combinations of model initialization time and forecast hour). Individual csv files will be saved for each forecast.

Finally, use **sounding_compilation_parquet.py** to combine the individual HRRR csv files into one lightweight parquet file which the training script will expect

An extra script called **hrrr_vvel_soilw_mslma_hgt_download.py** was not used for the operational TdAI but can be used to add HRRR vertical velocity, MSLP, and soil moisture to the training dataset. Adding these to the training dataset for 6 ASOS sites in the CAR CWA offered minimal performance improvement but it may be worth testing for your area.

## Step 2: Model Training & Evaluation

**You must have the following at your desired ASOS sites to train the TdAI deterministic and probabilistic models:**
1. A CSV file containing 01z NBM T, Td, Sky, wind speed, wind direction, and mixing height data
2. A CSV file containing 13z NBM T, Td, Sky, wind speed, wind direction, and mixing height data
3. A CSV file containing ASOS Td data
4. A parquet file containing all HRRR sounding data

After all data is downloaded, navigate to \model_training_STATIC and follow these steps to train the deterministic and probabilistic TdAI models:
```
TdAI/
├── model_training_STATIC/
    ├── trained_models/
    ├── TdAI_Deterministic_EVALUATION.py
    ├── TdAI_Deterministic_TRAINING.py
    ├── TdAI_Probabilistic_EVALUATION.py
    ├── TdAI_Probabilistic_TRAINING.py
    └── TdAI_Training_Dataset_Compilation.py
```

**a. Compile the Training Dataset**

```
Python TdAI_Training_Dataset_Compilation.py
```

Running this script will combine NBM - ASOS error (the target variable) with NBM station and HRRR sounding data (the predictors) along a constant valid time index.


**b. Train the Deterministic Models**

```
Python TdAI_Deterministic_TRAINING.py
```

Running this script will train separate deterministic models on each forecast cycle (combos of HRRR and NBM init time and forecast hour). It is recommended not to significantly alter the GBDT model hyperparameters 
as these have been show to provide good performance while minimizing the risk of overfitting to the dataset but the experienced developed may consider employing a hyperparameter optimization function. 
A weighting scheme is applied to the training dataset to ensure the model focuses most on correcting the large NBM errors. To evaluate model performance set PRODUCTION_MODE = False and define a holdout year for testing. 
When operationalizing the model set PRODUCTION_MODE = True to train on the entire dataset (note that this means an evaluation is not possible).

```
Output Trained Deterministic Models:
---------------
| TdAI Run | Target | NBM Input | HRRR Input |
|   0300z  |  Day 1 | 01z (f20) |  00z (f21) |
|   0300z  |  Day 2 | 01z (f44) |  00z (f45) |
|   1500z  |  Day 1 | 13z (f08) |  12z (f09) |
|   1500z  |  Day 2 | 13z (f32) |  12z (f33) |
```

**c. Evaluate the Deterministic Models**

```
Python TdAI_Deterministic_EVALUATION.py
```

_Only run this script if you held out a year for model evaluation._ You can choose from multiple different evaluation techniques do_top25 will show model performance on the top 25 largest NBM moist errors at each site from the evaluation year, do_scatter_plot will show a scatter plot of observed NBM error vs. model predicted NBM error at each site (which includes a coefficient of determination and skill score calculation), 
do_feature_importance will produce a feature importance plot showing the most importance variables contributing to the models' predictions at each site, and do_bust_threshold_count will compute a table counting the number of samples above a certain BUST_EXCEEDANCE_THRESHOLD_F value. Evaluation results from all TdAI models will be pooled together.

**d. Train the Probabilistic Models**

```
Python TdAI_Probabilistic_TRAINING.py
```

Run this script to train an array of probabilistic models (10th, 25th, 50th, 75th, 90th) for each forecast cycle (combos of HRRR and NBM init time and forecast hour). See below for what this framework looks like. As with the deterministic models, it is recommended not to significantly alter the GBDT model hyperparameters as these have been show to provide good performance while minimizing the risk of overfitting to the dataset (especially since overfitting is especially a concern at the 10th/90th tails) but the experienced developed may consider employing a hyperparameter optimization function. A weighting scheme is applied to the training dataset to ensure the model focuses most on correcting the large NBM errors. To evaluate model performance set PRODUCTION_MODE = False and define a holdout year for testing. When operationalizing the model set PRODUCTION_MODE = True to train on the entire dataset (note that this means an evaluation is not possible).

```
Output Trained Probabilistic Models:
---------------
| Percentile | TdAI Run | Target | NBM Input | HRRR Input |
|    10th    |   0300z  |  Day 1 | 01z (f20) |  00z (f21) |
|    25th    |   0300z  |  Day 1 | 01z (f20) |  00z (f21) |
|    50th    |   0300z  |  Day 1 | 01z (f20) |  00z (f21) |
|    75th    |   0300z  |  Day 1 | 01z (f20) |  00z (f21) |
|    90th    |   0300z  |  Day 1 | 01z (f20) |  00z (f21) |

|    10th    |   0300z  |  Day 2 | 01z (f44) |  00z (f45) |
|    25th    |   0300z  |  Day 2 | 01z (f44) |  00z (f45) |
|    50th    |   0300z  |  Day 2 | 01z (f44) |  00z (f45) |
|    75th    |   0300z  |  Day 2 | 01z (f44) |  00z (f45) |
|    90th    |   0300z  |  Day 2 | 01z (f44) |  00z (f45) |

|    10th    |   1500z  |  Day 1 | 13z (f08) |  12z (f09) |
|    25th    |   1500z  |  Day 1 | 13z (f08) |  12z (f09) |
|    50th    |   1500z  |  Day 1 | 13z (f08) |  12z (f09) |
|    75th    |   1500z  |  Day 1 | 13z (f08) |  12z (f09) |
|    90th    |   1500z  |  Day 1 | 13z (f08) |  12z (f09) |

|    10th    |   1500z  |  Day 2 | 13z (f32) |  12z (f33) |
|    25th    |   1500z  |  Day 2 | 13z (f32) |  12z (f33) |
|    50th    |   1500z  |  Day 2 | 13z (f32) |  12z (f33) |
|    75th    |   1500z  |  Day 2 | 13z (f32) |  12z (f33) |
|    90th    |   1500z  |  Day 2 | 13z (f32) |  12z (f33) |
```

**e. Evaluate the Probabilistic Models**
```
Python TdAI_Probabilistic_EVALUATION.py
```
_Only run this script if you held out a year for model evaluation._ You can choose from multiple different evaluation techniques as follows. Note that the script pools all forecast cycles together (e.g. the 50th percentile 03z Day 1, 03z Day 2, 15z Day 1, and 15z Day 2 forecast models are pooled together). 

**do_scatter_plot:** Produces a scatter plot of NBM observed vs median (50th) predicted error at each ASOS site. Use this to ensure the performance of the 50th percentile models is similar to the deterministic models

**do_ci_band_plot:** Produces a confidence interval band plot for each station. This shows where each individual observed NBM error sample fell in the probabilistic distribution over the evaluation year.

**do_gated_coverage_check:** Checks if the percentage of samples within the 50% and 80% confidence interval matches what is expected from a well calibrated distribution (e.g. do about 50% of the observed NBM-ASOS error fall within the model's 25th and 75th percentiles?) if a fire weather model run filter (to prevent the model running on non-fire weather days) is applied 

**do_ungated_coverage_check:** Checks if the percentage of samples within the 50% and 80% confidence interval matches what is expected from a well calibrated distribution (e.g. do about 50% of the observed NBM-ASOS error fall within the model's 25th and 75th percentiles?) on the raw, unfiltered dataset (truest calibration check)

**do_ungated_coverage_check:** Creates a probability integral transform (PIT) histogram for assessing calibration of the entire probabilistic TdAI forecast distribution (highly recommend looking at this to assess skew). The PIT histogram should appear left skewed (skewed towards predicting higher NBM errors). This is an intentional result of the weighting scheme and provides more protection against extreme errors. This option also produces a Continuous Ranked Probability Score (CRPS) table, measuring both the calibration and sharpness of the TdAI forecast distribution.

## STEP 3. Operationalizing TdAI

```
TdAI/
├── .github/
    └── workflows/
        ├── TdAI_cron.yml
        └── TdAI_verify.yml       
├── model_training_STATIC/
    └── trained_models/
├── deterministic_output/
├── probabilistic_output/
├── ABOUT.txt
├── CHANGELOG.txt
├── TdAI_deterministic_operational.py
├── TdAI_deterministic_verification.py
├── TdAI_probabilistic_operational.py
├── TdAI_dashboard.html
└── requirements.txt
```

**a. Set up ```TdAI_deterministic_operational.py``` for running the TdAI deterministic models operationally**

**b. Set up ```TdAI_probabilistic_operational.py``` for running the TdAI probabilistic models operationally**

**c. Set up ```TdAI_deterministic_verification.py``` for verifying the TdAI deterministic model forecasts operationally**

**d. Customize ```TdAI_dashboard.html``` to your liking for viewing TdAI forecasts and verification**

This script controls the TdAI dashboard webpage where you will display and view all TdAI forecasts and verification. It should be pretty functional right out of the box but a few tweaks may be necessary depending on how many ASOS sites you are training. ```ABOUT.txt``` and ```CHANGELOG.txt``` are plug-in text files showing how TdAI works and what changes have been made to the model design respectively.

**e. Set up GitHub Pages to host your dashboard webpage**

Navigate to "Repository Settings" -> "Pages" and ensure that under "Build and Deployment" -> "Source" is selected as "Deploy from Branch". Under "Branch", select the branch as "main" and the folder as "/ (root)". The webpage should be built in the next couple of minutes and be accessible at https://<YOUR_GITHUB_USERNAME>.github.io/<YOUR_REPO_NAME>/TdAI_dashboard

**f. Set up ```TdAI_cron.yml``` as the TdAI forecast automated workflow**

Modify this script as needed for updated filenames but it otherwise should be ready out of the box. Make sure you have a ```requirements.txt``` file in your repository containing all the Python libraries needed by ```TdAI_deterministic_operational.py``` ```TdAI_probabilistic_operational.py```

You can test this workflow by going to GitHub Actions, selecting the workflow name on the right, and then clicking "Run Workflow".

**g. Set up a job in Google Cloud Scheduler to initiate the forecast automated workflow**

Create an account for free in Google Cloud (I recommend using a personal Google account as it may not work otherwise). You will likely have to start a free trial but you get 3 free scheduled jobs in Google Cloud. Once the trial ends you will have to upgrade to a paid account but you still get the 3 free scheduled jobs. Search for "Cloud Scheduler" and then select "Create Job" of the top option bar. Enter the following information for each option:

---

**Name:** Enter what you want to call the job (e.g. tdai-forecast-pinger)

**Frequency:** Determines when the cron activates. It needs to be written in unix-cron format

**Timezone:** Use UTC

**Target Type:** HTTP

**URL:** https://api.github.com/repos/<YOUR_GITHUB_USERNAME>/<YOUR_REPO_NAME>/actions/workflows/<YOUR-YML-FILENAME>/dispatches 

**HTTP Headers:**

**Name 1:** Accept | **Value 1:** application/vnd.github+json

**Name 2:** Authorization | **Value 2:** Bearer <YOUR-GITHUB-PERSONAL-ACCESS-TOKEN>

**Name 3:** Content-Type | **Value 3:** application/octet-stream

**Name 4:** User-Agent | **Value 4:** Google-Cloud-Scheduler

**Body:** {"ref": "main"}

**Auth header:** None

---

When done you can test the job by clicking the 3 dots under the actions column of the job and selecting "Force Run."

**h. Set up ```TdAI_verify.yml``` as the TdAI forecast verification automated workflow**

Modify this script as needed for updated filenames but it otherwise should be ready out of the box. Make sure you have a ```requirements.txt``` file in your repository containing all the Python libraries needed by ```TdAI_deterministic_verification.py```

You can test this workflow by going to GitHub Actions, selecting the workflow name on the right, and then clicking "Run Workflow"

**i. Set up a job in Google Cloud Scheduler to initiate the forecast verification automated workflow**

Follow the same instructions in step 6.g. but ensure that when setting up the job the url option points to ```TdAI_verify.yml```

## STEP 4. Implementing an Automated, Sliding Window Retraining Workflow (Optional Advanced Feature)

**Note:** I recommend AGAINST implementing this until you are fully comfortable with running the TdAI automated workflow regularly

Adding this feature will result in an automated retraining of the TdAI deterministic and probabilistic models every month (or at a frequency of your choosing) to bring the training dataset up to current. The retraining will operate on a sliding window meaning it will add data to the current date and remove data older than a defined number of years. This helps ensure that TdAI is trained on the most recent versions of the NBM. 


## Questions? 
Ask in the TdAI discussion thread under the discussions tab of the repository or email me at seanmelanson12@gmail.com. I am more than happy to help in any way possible! 



