# How to Train TdAI on New ASOS Sites #
The following is a comprehensive guide for training TdAI on your own ASOS sites. Before going into detail, here are a few general guidelines:

1. The deterministic and probabilistic models should be trained on the 6 most recent years of data to ensure stability
2. I recommend not including dates well outside your fire season (e.g. winter) in the training dataset. We want to make sure the models are primarily learning the typical fire weather patterns favorable for NBM Td to be too moist
3. The model trains to correct NBM error at 21 UTC as this is the approximate time of the driest conditions, but this can be changed based on your time zone and local effects.


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

**ASOS_download.py**: Python script for downloading ASOS data. It will output a csv file combining all ASOS observations within a specified date range 

**NBM_download.py**: Python script for downloading NBM T, Td, sky, wind speed, wind direction, and mixing height. It will output a csv file combining all NBM day 1 and day 2 forecasts at 15, 18, and 21 UTC for a specified init time (should be 01z and 13z to align with the NBM forecasters edit)

**hrrr_soundings_download.py**: Python script for downloading HRRR soundings at the nearest grid point to the ASOS site for 4 cycles (combinations of model initialization time and forecast hour). Individual csv files will be saved for each forecast.

Finally, use **sounding_compilation_parquet.py** to combine the individual HRRR csv files into one lightweight parquet file which the training script will expect

An extra script called **hrrr_vvel_soilw_mslma_hgt_download.py** was not used for the operational TdAI but can be used to add HRRR vertical velocity, MSLP, and soil moisture to the training dataset. Adding these to the training dataset for 6 ASOS sites in the CAR CWA offered minimal performance improvement but it may be worth testing for your area.

## Step 2: Model Training

**You must have the following at your desired ASOS sites to train the TdAI deterministic and probabilistic models:**
1. A CSV file containing 01z NBM T, Td, Sky, wind speed, wind direction, and mixing height data
2. A CSV file containing 13z NBM T, Td, Sky, wind speed, wind direction, and mixing height data
3. A CSV file containing ASOS Td data
4. A parquet file containing all HRRR sounding data

After all data is downloaded, navigate to \model_training_STATIC and follow these steps to train the deterministic and probabilistic TdAI models:
```
TdAI/
├── model_training_STATIC/
    ├── TdAI_Deterministic_EVALUATION.py
    ├── TdAI_Deterministic_TRAINING.py
    ├── TdAI_Probabilistic_EVALUATION.py
    ├── TdAI_Probabilistic_TRAINING.py
    └── TdAI_Training_Dataset_Compilation.py
```

**1. Compile the Training Dataset**

```
Python TdAI_Training_Dataset_Compilation.py
```

Running this script will combine NBM - ASOS error (the target variable) with NBM and HRRR sounding data (the predictors) along a constant valid time.


**2. Train the Deterministic Models**

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

**3. Evaluate the Deterministic Models**

```
Python TdAI_Deterministic_EVALUATION.py
```

Running this script will only work if you held out a year for model evaluation. You can choose from multiple different evaluation techniques do_top25 will show model performance on the top 25 largest NBM moist errors at each site
from the evaluation year, do_scatter_plot will show a scatter plot of observed NBM error vs. model predicted NBM error at each site (which includes a coefficient of determination and skill score calculation), 
do_feature_importance will produce a feature importance plot showing the most importance variables contributing to the models' predictions at each site, and do_bust_threshold_count will compute a table counting the number of samples
above a certain BUST_EXCEEDANCE_THRESHOLD_F value. Evaluation results from all TdAI models will be pooled together.

**4. Train the Probabilistic Models**

```
Python TdAI_Probabilistic_TRAINING.py
```

Run this script to train an array of probabilistic models (10th, 25th, 50th, 75th, 90th) for each forecast cycle (combos of HRRR and NBM init time and forecast hour). 

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


## Questions? 
Ask in the TdAI discussion thread under the discussions tab of the repository or email me at seanmelanson12@gmail.com. I am more than happy to help in any way possible! 



