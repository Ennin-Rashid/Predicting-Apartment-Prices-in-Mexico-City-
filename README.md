# Predicting Apartment Prices in Mexico City 🇲🇽

This repository contains a worked notebook (Predicting Apartment Prices in Mexico City 🇲🇽.ipynb) that walks through cleaning and exploring a Real Estate dataset for Mexico City with the goal of preparing the data for modeling apartment prices. The notebook focuses on data wrangling and exploratory data analysis (EDA). Model training is scaffolded (imports and utilities included) but the notebook's visible cells stop after cleaning and visual exploration.

Below I describe what I did, how the pipeline works, the results observed from the EDA, and recommended next steps for modeling and evaluation.

---

## Quick summary / story

I inspected the notebook and implemented a reproducible wrangling function called `wrangle(filepath)` that:
- reads one CSV file,
- filters rows to apartments within "Distrito Federal" and with price less than 100,000 USD,
- removes outliers in `surface_covered_in_m2` (keeps 10–90 percentile),
- splits a "lat-lon" column into numeric `lat` and `lon`,
- extracts a `borough` value from `place_with_parent_names`,
- drops many columns that are either leaky (contain target-derived values), high-cardinality/low-signal, or have many missing values.

I applied `wrangle` over the CSV files that match `data/mexico-city-real-estate-*.csv`, concatenated the frames, and explored the combined dataset with histograms, scatter plots, and a Mapbox scatter plot to visualize prices geographically.

---

## What I did (working process)

1. Environment & imports
   - Standard data science stack: pandas, numpy, matplotlib, seaborn, plotly.express.
   - Modeling / preprocessing imports present for future work: scikit-learn (SimpleImputer, LinearRegression, Ridge, pipelines), category_encoders.OneHotEncoder, mean_absolute_error.

2. Data ingestion
   - The notebook reads files using a glob:
     - files = glob("data/mexico-city-real-estate-*.csv")
     - files.sort()
     - frames = [wrangle(file) for file in files]
     - df = pd.concat(frames, ignore_index=True)

3. Wrangling (function: wrangle(filepath))
   - Read CSV into pandas DataFrame.
   - Filter to:
     - `place_with_parent_names` contains "Distrito Federal".
     - `property_type` == "apartment".
     - `price_aprox_usd` < 100_000 (this threshold is used in the notebook).
   - Remove surface area outliers using the 10th and 90th percentiles of `surface_covered_in_m2`.
   - Split a `lat-lon` string column into two numeric columns `lat` and `lon` and drop the original `lat-lon`.
   - Extract `borough` from `place_with_parent_names` by splitting on `|` and taking the second element.
   - Drop columns:
     - high-null or low-signal: `floor`, `expenses`
     - low- or high-cardinality categorical / identifiers: `property_type`, `operation`, `properati_url`, `currency`
     - leaky columns (derived from price or target): `price`, `price_aprox_local_currency`, `price_per_m2`, `price_usd_per_m2`
     - columns with high multicollinearity: `surface_total_in_m2`, `rooms`
   - Return the cleaned frame with only the chosen features.

4. Concatenation and basic cleaning results
   - The notebook prints shapes and info:
     - Example single-file output: `frame1 shape: (1101, 5)`
     - After concatenating all frames: `RangeIndex: 5473 entries, 0 to 5472` with 5 columns:
       - `price_aprox_usd` (float)
       - `surface_covered_in_m2` (float)
       - `lat` (float)
       - `lon` (float)
       - `borough` (object)
     - `lat` and `lon` are not fully complete: 5149 non-null (so ~324 rows missing lat/lon after wrangle + concat).

5. Exploratory data analysis (plots)
   - Distribution of apartment prices: a histogram (matplotlib) shows prices concentrated under the 100k USD filter with a skewed distribution and a concentration in lower price ranges.
   - Price vs Area scatter: a scatter plot of `surface_covered_in_m2` vs `price_aprox_usd` shows a generally positive relationship (larger area tends to have higher price) but with wide variance — many apartments with similar area have very different prices (city/borough and location matter).
   - Map visualization (Plotly Mapbox, ungraded): a geographic scatter with color mapped to price illustrates spatial patterns (some boroughs / areas show higher prices).

---

## Results (what we observed)

Data after cleaning
- Rows (concatenated): 5,473
- Columns retained: 5 (price_aprox_usd, surface_covered_in_m2, lat, lon, borough)
- Missing location data: lat/lon present for 5,149 rows; ~324 rows missing coordinates.

Key EDA observations
- Price distribution:
  - Because of the chosen price filter (< 100k USD), the distribution is concentrated below 100k with a right skew (a long tail toward higher prices up to the threshold).
  - The histogram shows many listings at lower price bands; there is no evidence in the notebook of very large prices (intentionally filtered).
- Price vs Area:
  - There is a positive trend: larger `surface_covered_in_m2` tends to correlate with higher `price_aprox_usd`.
  - The spread indicates other factors (borough / exact location, building amenities, quality) contribute strongly to price variability.
- Geography:
  - Map visualizations highlight spatial clusters of higher-priced apartments in certain boroughs; conversely, other boroughs show many lower-priced listings. (The notebook includes a Mapbox scatter — you can interact with it in the notebook to see exact clusters.)

Quantitative metrics
- The notebook does not show any trained model results or numerical evaluation metrics (MAE/RMSE/R^2). The modelling imports are available but there is no final model training/evaluation cell in the visible cell outputs.

---

## Reproducibility — how to run the notebook / reproduce cleaning & EDA

Prerequisites
- Python 3.8+ (tested in the notebook environment)
- Install dependencies (example):
  - pip install -r requirements.txt
  - or pip install pandas numpy matplotlib seaborn plotly scikit-learn category_encoders

Steps
1. Open the notebook:
   - file: `Predicting Apartment Prices in Mexico City 🇲🇽.ipynb` (root)
   - Or run Jupyter: jupyter lab / jupyter notebook and open that file.

2. Run the wrangle + concat cells:
   - The notebook defines `wrangle(filepath)` and then does:
     - files = glob("data/mexico-city-real-estate-*.csv")
     - frames = [wrangle(file) for file in files]
     - df = pd.concat(frames, ignore_index=True)

3. Recreate the EDA
   - Histogram:
     - fig, ax = plt.subplots()
     - ax.hist(df["price_aprox_usd"]); ax.set_xlabel(...); ax.set_ylabel(...); ax.set_title(...)
   - Scatter:
     - fig, ax = plt.subplots()
     - ax.scatter(df["surface_covered_in_m2"], df["price_aprox_usd"]); ...
   - Mapbox scatter (example, using plotly.express):
     - px.scatter_mapbox(df.dropna(subset=["lat","lon"]), lat="lat", lon="lon",
       color="price_aprox_usd", hover_name="borough", zoom=10, mapbox_style="open-street-map")

Files used in the notebook
- Notebook:
  - Predicting Apartment Prices in Mexico City 🇲🇽.ipynb
- Data:
  - data/mexico-city-real-estate-1.csv (and other files matching `mexico-city-real-estate-*.csv`)

Key function
- wrangle(filepath) — implement and test on each CSV.

---

## Recommended next steps (modeling plan)

1. Define prediction target and baseline
   - Target already present: `price_aprox_usd`
   - Baseline predictor: median price or a simple linear regression on area.

2. Feature engineering
   - Use `surface_covered_in_m2` and `borough` (categorical) as primary features.
   - Add engineered geospatial features: distance to city center or to important landmarks, cluster labels for locations (k-means on lat/lon).
   - Fill missing lat/lon if possible (geocoding or drop rows if small fraction) — currently ~324 rows missing coords.

3. Preprocessing pipeline
   - Numeric pipeline: SimpleImputer (median), StandardScaler
   - Categorical pipeline: OneHotEncoder (or target encoding if many categories)
   - Combine in ColumnTransformer and wrap with an sklearn estimator in a pipeline.

4. Models & hyperparameter search
   - Try linear models (Ridge), tree-based (RandomForest, XGBoost/LightGBM), and an ensemble.
   - Use cross-validation (e.g., 5-fold, stratify if needed) and track MAE, RMSE, R^2.
   - Use GridSearchCV / RandomizedSearchCV or Optuna for tuning.

5. Evaluation & error analysis
   - Report MAE and RMSE in USD (MAE is intuitive for business owners).
   - Plot residuals vs predicted and residuals vs features to find heteroscedasticity.
   - Analyze borough-specific performance and failure modes (e.g., small-area apartments, atypical properties).

6. Deployment (optional)
   - Save the trained pipeline (joblib/pickle).
   - Serve via a small API (FastAPI) or provide batch prediction scripts.

---

## Project structure (as used in the notebook)

Suggested / observed layout:

.
├── data/
│   ├── mexico-city-real-estate-1.csv
│   ├── mexico-city-real-estate-2.csv
│   └── ...
├── Predicting Apartment Prices in Mexico City 🇲🇽.ipynb
└── README.md (this file)

---

## Notes, limitations, and caveats

- Filtering threshold: the notebook filters `price_aprox_usd < 100_000`. This is an explicit choice in the wrangle function — if you wish to model the full price range, remove or adjust this filter.
- The current dataset is limited to listings classified as "apartment" and those where `place_with_parent_names` contains "Distrito Federal".
- Several columns were dropped (including `rooms`) — you may wish to revisit that decision if those columns are predictive and available with reasonable completeness.
- No trained model or numeric evaluation results are present in the visible notebook outputs. The repository is ready for modeling but final training + evaluation steps are left to complete.

---

## Contact & license

Maintainer: Ennin-Rashid (GitHub: @Ennin-Rashid)  
Notebook source: https://github.com/Ennin-Rashid/Predicting-Apartment-Prices-in-Mexico-City-/blob/main/Predicting%20Apartment%20Prices%20in%20Mexico%20City%20%F0%9F%87%B2%F0%9F%87%BD.ipynb



---

If you want, I can:
- generate a ready-to-run training script (train.py) including a full sklearn pipeline (imputation, encoding, model) and cross-validation reporting (MAE/RMSE/R^2), or
- update the notebook to include model training and produce final metrics and saved artifacts.

Tell me which you prefer and I will create the files and code for the next step.
